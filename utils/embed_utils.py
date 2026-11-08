from jax import random, numpy as jnp, jit
import jax
from functools import partial
from ngclearn.utils.optim import get_opt_init_fn, get_opt_step_fn
from ngclearn.components.jaxComponent import JaxComponent
from ngclearn import Compartment
from ngclearn import compilable
from ngclearn.utils import tensorstats
from ngclearn.utils.distribution_generator import DistributionGenerator as dist
import os
from pathlib import Path

@partial(jit, static_argnums=[0,1])
def _create_sinusoidal_embeddings(seq_len, embed_dim):
    """
    Create fixed absolute sinusoidal positional embeddings.

    Returns:
        Shape: (seq_len, embed_dim)
    """
    position = jnp.arange(seq_len)[:, None]
    div_term = jnp.exp(jnp.arange(0, embed_dim, 2) * 
                      (-jnp.log(10000.0) / embed_dim))
    
    embeddings = jnp.zeros((seq_len, embed_dim))
    embeddings = embeddings.at[:, 0::2].set(jnp.sin(position * div_term))
    embeddings = embeddings.at[:, 1::2].set(jnp.cos(position * div_term))
    return embeddings

@partial(jit, static_argnums=[1])
def _calc_prior_term(weights, prior_type, prior_lmbda):
    """
    Computes an L1/L2/Elastic-Net regularization term to be added to an
    embedding update before the optimizer step. Returns zeros if disabled.
    """
    dW_reg = jnp.zeros_like(weights)

    if prior_type == "l2" or prior_type == "ridge":
        dW_reg = -weights * prior_lmbda
    elif prior_type == "l1" or prior_type == "lasso":
        dW_reg = -jnp.sign(weights) * prior_lmbda
    elif prior_type == "l1l2" or prior_type == "elastic_net":
        prior_scale, l1_ratio = prior_lmbda
        dW_reg = -jnp.sign(weights) * l1_ratio - weights * (1 - l1_ratio) / 2
        dW_reg = dW_reg * prior_scale

    return dW_reg

@partial(jit, static_argnums=[4, 5, 6, 7, 9])
def _compute_embedding_updates(inputs, post, word_weights, pos_weights, 
                              vocab_size, seq_len, embed_dim, batch_size, pos_learnable,
                              prior_type, prior_lmbda):
    """
    Compute updates for word embeddings and positional embeddings
    """
    
    # Flatten for processing
    flat_tokens = inputs.reshape(-1)
    flat_errors = post.reshape(batch_size * seq_len, embed_dim)
     
    # Word embeddings update - accumulate gradients for each token
    d_word_weights = jnp.zeros((vocab_size, embed_dim))
    d_word_weights = d_word_weights.at[flat_tokens].add(flat_errors)


    # positional embededings update

    d_pos_weights = jnp.zeros((seq_len, embed_dim))
    
    batch_positions = jnp.tile(jnp.arange(seq_len), batch_size).astype(jnp.int32)
    d_pos_weights =  jax.lax.cond(pos_learnable, lambda: d_pos_weights.at[batch_positions].add(flat_errors), lambda: d_pos_weights)
            
    return d_word_weights, d_pos_weights

@partial(jit, static_argnums=[1, 2])
def _enforce_constraints(weights, w_bound, is_nonnegative=False):
    """
    Applies a hard clip to weights after the optimizer step.
    w_bound <= 0 disables clipping.
    """
    _W = weights
    if w_bound > 0.:
        if is_nonnegative:
            _W = jnp.clip(_W, 0., w_bound)
        else:
            _W = jnp.clip(_W, -w_bound, w_bound)
    return _W

class EmbeddingSynapse(JaxComponent):
    """
    A synaptic cable that handles word and positional embeddings.

    | --- Synapse Compartments: ---
    | inputs - input token indices (takes in external signals)
    | outputs - output embedding signals (only word embeddings)
    | word_weights - word embedding matrix
    | pos_weights - position embedding matrix  
    | post - post-synaptic signals for learning (takes in external signals)
    | key - JAX PRNG key
    | --- Synaptic Plasticity Compartments: ---
    | dWordWeights - current delta matrix for word embedding changes
    | dPosWeights - current delta matrix for position embedding changes
    | word_opt_params - optimizer statistics for word embeddings
    | pos_opt_params - optimizer statistics for position embeddings

    Args:
        name: the string name of this component

        vocab_size: size of vocabulary for word embeddings

        seq_len: sequence length

        embed_dim: dimensionality of embeddings

        batch_size: batch size dimension

        eta: global learning rate 

        optim_type: optimization scheme (Default: "sgd")

        pos_learnable: whether position embeddings are learnable or fixed

        position_encoding: type of position encoding (Default: "rope")

        weight_scale: scaling factor for weight initialization (Default: 0.02)

        w_bound: maximum absolute value to hard-clip word_weights/pos_weights
            to after each evolve() step; <= 0 disables clipping (Default: 0.)

        is_nonnegative: if True and w_bound > 0, clip to [0, w_bound] instead
            of [-w_bound, w_bound] (Default: False)

        prior: regularization prior applied to both word_weights and
            pos_weights; tuple of (name, lambda): ("constant", 0.) [off,
            default], ("l2"/"ridge", lmbda), ("l1"/"lasso", lmbda),
            ("l1l2"/"elastic_net", (scale, l1_ratio))

       weight_init: initialization kernel (shape, key) -> array used to
                    initialize word_weights and learnable pos_weights.
                    If None, embeddings are initialized to zeros.
    """

    def __init__(
            self, name, vocab_size, seq_len, embed_dim, batch_size,
            eta, optim_type,position_encoding="rope", pos_learnable=False, weight_scale=0.02, sign_value=-1.,
            w_bound=1., is_nonnegative=False, prior=("constant", 0.),
            weight_init=None,
            **kwargs
    ):
        super().__init__(name, **kwargs)

        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.eta = eta
        self.weight_scale = weight_scale
        self.optim_type = optim_type
        self.position_encoding = position_encoding
        self.use_positional = (self.position_encoding == "positional")
        self.pos_learnable = pos_learnable
                self.sign_value = sign_value

        # Regularization/bounding configuration
        self.w_bound = w_bound
        self.is_nonnegative = is_nonnegative
        prior_type, prior_lmbda = prior
        if prior_type is None:
            prior_type = "constant"
        if prior_type.lower() == "gaussian":
            prior_type = "ridge"
        elif prior_type.lower() == "laplacian":
            prior_type = "lasso"
        self.prior_type = prior_type
        self.prior_lmbda = prior_lmbda

        key = random.PRNGKey(1234)
        word_key, pos_key = random.split(key, 2)
        
        # Fallback initialization: zeros, since scatter-add updates break
        # symmetry per-token/position without needing random initialization
        if weight_init is not None:
            word_weights = weight_init((self.vocab_size, self.embed_dim), word_key)
        else:
            word_weights = jnp.zeros((self.vocab_size, self.embed_dim))
        if self.use_positional:
            if self.pos_learnable:
            if weight_init is not None:
                    pos_weights = weight_init((self.seq_len, self.embed_dim), pos_key)
            else:
                pos_weights = jnp.zeros((self.seq_len, self.embed_dim))
            else:
                pos_weights = _create_sinusoidal_embeddings(self.seq_len, self.embed_dim)
        else:
            # unused in RopE mode
            pos_weights = jnp.zeros((seq_len, embed_dim))
        
        ## Compartments
        self.inputs = Compartment(jnp.zeros((self.batch_size, self.seq_len), dtype=jnp.int32))
        self.outputs = Compartment(jnp.zeros((self.batch_size, self.seq_len, self.embed_dim)))
        self.word_weights = Compartment(word_weights)
        self.pos_weights = Compartment(pos_weights)
        self.post = Compartment(jnp.zeros((self.batch_size, self.seq_len, self.embed_dim)))
        
        self.dWordWeights = Compartment(jnp.zeros((self.vocab_size, self.embed_dim)))
        self.dPosWeights = Compartment(jnp.zeros((self.seq_len, self.embed_dim)))
        
        # Optimization
        self.opt = get_opt_step_fn(optim_type, eta=self.eta)
        self.word_opt_params = Compartment(
            get_opt_init_fn(optim_type)([self.word_weights.get()])
        )
        #position optimizer for learned only
        if(self.use_positional and self.pos_learnable):
            self.pos_opt_params = Compartment(
                get_opt_init_fn(optim_type)([self.pos_weights.get()])
            )
        else:
            self.pos_opt_params = Compartment(None)
    @compilable
    def advance_state(self):
        """
        Forward pass: output = word_embedding[inputs] + position_embedding[positions]
        """
        inputs=self.inputs.get()
        word_weights=self.word_weights.get()
        seq_len=self.seq_len.get()
        embed_dim=self.embed_dim.get()
        batch_size = inputs.shape[0]
        
        flat_tokens = inputs.reshape(-1).astype(jnp.int32)
        word_embeds_flat = word_weights[flat_tokens]
        word_embeds = word_embeds_flat.reshape(batch_size, self.seq_len, self.embed_dim)
        
        if self.use_positional:
            pos_weights = self.pos_weights.get()
            positions = jnp.arange(seq_len)
            pos_embeds= pos_weights[positions]
            pos_embeds_batch = jnp.broadcast_to(pos_embeds, (batch_size, seq_len, embed_dim))

            combined_embeddings = word_embeds + pos_embeds_batch
        else:
            #RoPE mode
            combined_embeddings = word_embeds
        self.outputs.set(combined_embeddings)

  
    @compilable
    def evolve(self):
        """
        Learning step: Hebbian updates for word embeddings and update positional embeddings only when using learned absolute positional embeddings
        """
        opt = self.opt.get()
        vocab_size = self.vocab_size.get()
        seq_len = self.seq_len.get()
        embed_dim = self.embed_dim.get()
        batch_size = self.batch_size.get()
        inputs = self.inputs.get()
        post = self.post.get()
        word_weights = self.word_weights.get()
        pos_weights = self.pos_weights.get()
        word_opt_params = self.word_opt_params.get()
        pos_opt_params = self.pos_opt_params.get()
        batch_size = inputs.shape[0]
        # Compute embedding updates
        inputs= inputs.astype(jnp.int32)
        d_word_weights, d_pos_weights = _compute_embedding_updates(
            inputs, post, word_weights, pos_weights, self.vocab_size, self.seq_len,
            self.embed_dim, batch_size, self.pos_learnable, self.prior_type, self.prior_lmbda
        )

        # Optional soft weight-bound scaling of the update
        if self.w_bound > 0.:
            d_word_weights = d_word_weights * (self.w_bound - jnp.abs(word_weights))
            d_pos_weights = d_pos_weights * (self.w_bound - jnp.abs(pos_weights))

        d_word_weights = d_word_weights * self.sign_value
        d_pos_weights = d_pos_weights * self.sign_value

        word_opt_params, [new_word_weights] = opt(
            word_opt_params, [word_weights], [d_word_weights]
        )
        
        self.word_weights.set(new_word_weights)
        self.word_opt_params.set(word_opt_params)
        self.dWordWeights.set(d_word_weights)

        new_pos_weights = pos_weights
        new_pos_opt_params = pos_opt_params

        if (self.use_positional and self.pos_learnable):
            
            pos_opt_params, [new_pos_weights] = opt(
                pos_opt_params, [pos_weights], [d_pos_weights]
            )
            new_pos_opt_params = pos_opt_params
        
        # Optional hard clip of weights post-update
        new_word_weights = _enforce_constraints(new_word_weights, self.w_bound, is_nonnegative=self.is_nonnegative)
        if self.pos_learnable:
            new_pos_weights = _enforce_constraints(new_pos_weights, self.w_bound, is_nonnegative=self.is_nonnegative)

        # return new_word_weights, new_pos_weights, d_word_weights, d_pos_weights, word_opt_params, new_pos_opt_params
        self.word_weights.set(new_word_weights)
        self.pos_weights.set(new_pos_weights)
        self.dWordWeights.set(d_word_weights)
        self.dPosWeights.set(d_pos_weights)
        self.word_opt_params.set(word_opt_params)
        self.pos_opt_params.set(new_pos_opt_params)
    @compilable
    def reset(self):
        """
        Reset compartments to zeros
        """

        inputs = jnp.zeros((self.batch_size, self.seq_len), dtype=jnp.int32)
        outputs = jnp.zeros((self.batch_size, self.seq_len, self.embed_dim))
        post = jnp.zeros((self.batch_size, self.seq_len, self.embed_dim))
        dWordWeights = jnp.zeros((self.vocab_size, self.embed_dim))
        dPosWeights = jnp.zeros((self.seq_len, self.embed_dim))
        # return inputs, outputs, post, dWordWeights, dPosWeights
        self.inputs.set(inputs)
        self.outputs.set(outputs)
        self.post.set(post)
        self.dWordWeights.set(dWordWeights)
        self.dPosWeights.set(dPosWeights)


    @classmethod
    def help(cls):
        """
        Return metadata describing the embedding component.

        EmbeddingSynapse supports two position-encoding configurations:

        - rope: returns word embeddings only. RoPE is applied later to
        the query and key tensors in the attention component.
        - positional: adds learned or fixed absolute positional
        embeddings to the word embeddings.
        """
        properties = {
            "synapse_type": (
                "EmbeddingSynapse - performs token embedding lookup and "
                "optionally adds absolute positional embeddings. In RoPE "
                "mode, rotation is applied later to attention queries and keys."
            )
            }

        compartment_props = {
            "inputs": {
                "inputs": (
                    "Input token indices "
                    "(batch_size, seq_len)"
                ),
                "post": (
                    "Post-synaptic error signals used for learning "
                    "(batch_size, seq_len, embed_dim)"
                ),
            },
            "states": {
                "word_weights": (
                    "Word embedding matrix "
                    "(vocab_size, embed_dim)"
                ),
                "pos_weights": (
                    "Absolute positional embedding matrix "
                    "(seq_len, embed_dim); used only in positional mode"
                ),
                "word_opt_params": (
                    "Optimizer state for the word embedding matrix"
                ),
                "pos_opt_params": (
                    "Optimizer state for learned positional embeddings; "
                    "inactive in RoPE and fixed positional modes"
                ),
            },
            "analytics": {
                "dWordWeights": (
                    "Word embedding update matrix "
                    "(vocab_size, embed_dim)"
                ),
                "dPosWeights": (
                    "Positional embedding update matrix "
                    "(seq_len, embed_dim); zero unless learned "
                    "positional embeddings are enabled"
                ),
            },
            "outputs": {
                "outputs": (
                    "Embedding output "
                    "(batch_size, seq_len, embed_dim)"
                ),
            },
        }

        hyperparams = {
            "vocab_size": "Size of the token vocabulary",
            "seq_len": "Maximum input sequence length",
            "embed_dim": "Dimensionality of each embedding",
            "batch_size": "Batch size dimension",
            "position_encoding": (
                "Position-encoding mode: 'rope' or 'positional'"
            ),
            "pos_learnable": (
                "Whether absolute positional embeddings are trainable; "
                "used only when position_encoding='positional'"
            ),
            "eta": "Optimizer learning rate",
            "optim_type": "Optimization algorithm",
            "weight_scale": "Embedding initialization scale",
        }

        info = {
            cls.__name__: properties,
            "compartments": compartment_props,
            "dynamics": (
                "rope: outputs = word_embedding[inputs]; "
                "positional: outputs = word_embedding[inputs] + "
                "position_embedding[positions]"
            ),
            "hyperparameters": hyperparams,
        }

        return info    
    def __repr__(self):
        # FIX: Replaced the non-existent Compartment.is_compartment with isinstance(..., Compartment)
        comps = [varname for varname in dir(self) if isinstance(getattr(self, varname), Compartment)]
        
        if not comps:
            # Handle the case where no compartments are found to avoid max() on an empty sequence
            return f"[{self.__class__.__name__}] PATH: {self.name}\n  No Compartments Found"

        maxlen = max(len(c) for c in comps) + 5
        lines = f"[{self.__class__.__name__}] PATH: {self.name}\n"
        
        # Iterate over the valid compartment names
        for c in comps:
            # Get the actual Compartment object
            compartment_obj = getattr(self, c) 
            compartment_value = compartment_obj.get()
            
            # This check is necessary because pos_opt_params contains None
            # in RoPE mode and fixed positional-embedding mode.
            # tensorstats() should not be called on None.
            if compartment_value is None:
                line = "None"

            else:
                try:
                    # Get tensor statistics (assuming tensorstats is correctly imported)
                    stats = tensorstats(compartment_value)

                except (TypeError, ValueError, AttributeError):
                    # Optimizer-state compartments may contain lists,
                    # tuples, or other nested structures instead of one
                    # tensor. Represent their type without changing them.
                    stats = None
                    line = (
                        f"type: "
                        f"{type(compartment_value).__name__}"
                    )

                else:
                    if stats is not None:
                        line = [
                            f"{k}: {v}"
                            for k, v in stats.items()
                        ]
                        line = ", ".join(line)
                    else:
                        line = "None"
                
            lines += f"  {f'({c})'.ljust(maxlen)}{line}\n"
            
        return lines


    def save(self, directory, **kwargs):
        """Save word embedding parameters and learned positional embedding parameters to disk."""
        
        Path(directory).mkdir(parents=True, exist_ok=True)
        file_name = os.path.join(directory, f"{self.name}.npz")
        
        if (self.use_positional and self.pos_learnable):
            jnp.savez(
                file_name,
                word_weights=(self.word_weights.get()),
                pos_weights=(self.pos_weights.get()),
            )
        else:
            jnp.savez(
                file_name,
                word_weights=( self.word_weights.get()),
            )
      

    def load(self, directory, **kwargs):
        """Load word embedding parameters from disk."""
        import os
        file_name = os.path.join(directory, f"{self.name}.npz")
        data = jnp.load(file_name)
        
        self.word_weights.set(data['word_weights'])

        if (self.use_positional and self.pos_learnable ):
            if "pos_weights" not in data:
                raise ValueError(
                    "The checkpoint has no "
                    "pos_weights. It may have "
                    "been trained using RoPE."
                )

            self.pos_weights.set(
                data["pos_weights"]
            )