# NGC-PC-Transformers
Implementation of a Predictive Coding Transformer built on the Neural Generative Coding framework.

## Installation

Clone the Repository

```bash
git clone https://github.com/iCog-Labs-Dev/NGC-PC-Transformers.git
```
Navigate to the Project Directory

```bash
cd NGC-PC-Transformers
```
Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
``` 
Install required packages:

```bash
pip install -r requirements.txt
```
Run the tokenization
```bash
python -m data_preprocess.tokenizer
```
Run the training
```bash
python train.py
```
Run the evaluation
```bash
python eval.py
```
Run the generation
```bash
python generation.py
``` 

## Fast KV-Cache Decoder & Generation

Text generation supports an optimized, stand-alone JIT-compiled KV-cache decoder (`utils/fast_decode.py`):
- **Toggle**: Configure `use_kv_cache = True` (fast path) or `False` (legacy full-context loop fallback) in `config.py`.
- **Architecture**: Extracts weights directly from the loaded `NGCTransformer` into a JAX PyTree and executes only the feed-forward projection path without graph compilation overhead, unused position passes, or intermediate compartments.
- **Sliding-Window Policy**:
  - `t < seq_len`: Pure incremental $O(1)$ matmul step writing key/value slices into fixed-shape cache arrays via `jax.lax.dynamic_update_slice`.
  - `t >= seq_len`: Window slides by 1 token; absolute position embeddings shift, so the last `seq_len` tokens are re-prefilled in a single batched causal pass, keeping outputs strictly equivalent to full context.
- **Eval Mode**: Dropout is disabled (`dropout_rate=0.0`) during inference.
- **Sampling**: Temperature and top-k are applied to unnormalized logits / log-probabilities.

### Running Equivalence Tests
To verify exact mathematical equivalence between `fast_decode` and `model.process()`:
```bash
python test_fast_decode.py
```
This tests incremental steps ($t < S$), sliding-window re-prefill ($t \ge S$), config mismatches, and greedy end-to-end token match.
