import jax
from jax import numpy as jnp, random
from model import NGCTransformer
from ngclearn.utils.metric_utils import measure_CatNLL
from data_preprocess.data_loader import DataLoader
from config import Config as config
from eval import eval_model
import time
import os
from datetime import datetime

jax.config.update("jax_default_matmul_precision", "high")
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

class TrainingLogger:
    """Logger to save training logs to a .txt file"""
    
    def __init__(self, log_dir="logs"):
        """Initialize logger with directory to save logs"""
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"training_log_{timestamp}.txt")
        
        # Initialize log file with header
        with open(self.log_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("NGC TRANSFORMER TRAINING LOG\n")
            f.write("="*80 + "\n")
            f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            # Write configuration
            f.write("CONFIGURATION:\n")
            f.write("-"*40 + "\n")
            f.write(f"  seq_len: {config.seq_len}\n")
            f.write(f"  batch_size: {config.batch_size}\n")
            f.write(f"  n_embed: {config.n_embed}\n")
            f.write(f"  vocab_size: {config.vocab_size}\n")
            f.write(f"  n_layers: {config.n_layers}\n")
            f.write(f"  n_heads: {config.n_heads}\n")
            f.write(f"  n_iter: {config.n_iter}\n")
            f.write(f"  epochs: {config.epoch}\n")
            f.write(f"  tau_m: {config.tau_m}\n")
            f.write(f"  eta: {config.eta}\n")
            f.write(f"  eta_o: {config.eta_o}\n")
            f.write(f"  dropout_rate: {config.dropout_rate}\n")
            f.write(f"  pos_learnable: {config.pos_learnable}\n")
            f.write(f"  optim_type: {config.optim_type}\n")
            f.write(f"  wub: {config.wub}\n")
            f.write(f"  wlb: {config.wlb}\n")
            f.write("-"*40 + "\n\n")
    
    def log_batch(self, epoch, batch_idx, metrics):
        """Log batch-level metrics"""
        with open(self.log_file, 'a') as f:
            f.write(f"  Epoch {epoch} | Batch {batch_idx:3d}: ")
            f.write(f"EFE = {metrics['efe']:.4f}, ")
            f.write(f"CE = {metrics['ce']:.4f}, ")
            f.write(f"PPL = {metrics['ppl']:.4f}")
            if 'l1' in metrics:
                f.write(f" | L1 = {metrics['l1']:.4f}")
                f.write(f" | L2 = {metrics['l2']:.4f}")
                f.write(f" | L3 = {metrics['l3']:.4f}")
                f.write(f" | L4 = {metrics['l4']:.4f}")
                f.write(f" | L5 = {metrics['l5']:.4f}")
            f.write("\n")
    
    def log_epoch_summary(self, epoch, train_metrics, val_metrics):
        """Log epoch-level summary"""
        with open(self.log_file, 'a') as f:
            f.write(f"\n{'='*40}\n")
            f.write(f"EPOCH {epoch} SUMMARY\n")
            f.write(f"{'='*40}\n")
            f.write(f"Training:   CE = {train_metrics['ce']:.4f}, ")
            f.write(f"PPL = {train_metrics['ppl']:.4f}, ")
            f.write(f"Avg EFE = {train_metrics['efe']:.4f}\n")
            f.write(f"Validation: CE = {val_metrics['ce']:.4f}, ")
            f.write(f"PPL = {val_metrics['ppl']:.4f}\n")
            f.write(f"{'='*40}\n\n")
    
    def log_training_complete(self, total_time, final_train_metrics, final_val_metrics):
        """Log training completion summary"""
        with open(self.log_file, 'a') as f:
            f.write("\n" + "="*80 + "\n")
            f.write("TRAINING COMPLETE\n")
            f.write("="*80 + "\n")
            f.write(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Training Time: {total_time:.2f} seconds ({total_time/60:.2f} min)\n")
            f.write("-"*40 + "\n")
            f.write("FINAL METRICS:\n")
            f.write(f"  Train CE: {final_train_metrics['ce']:.4f}\n")
            f.write(f"  Train PPL: {final_train_metrics['ppl']:.4f}\n")
            f.write(f"  Train Avg EFE: {final_train_metrics['efe']:.4f}\n")
            f.write(f"  Val CE: {final_val_metrics['ce']:.4f}\n")
            f.write(f"  Val PPL: {final_val_metrics['ppl']:.4f}\n")
            f.write("="*80 + "\n")

def main():
    seq_len, batch_size, n_embed, vocab_size, n_layers, n_heads, n_iter, optim_type = config.seq_len, config.batch_size, config.n_embed, config.vocab_size, config.n_layers, config.n_heads, config.n_iter, config.optim_type
    pos_learnable = config.pos_learnable
    epoch = config.epoch
    wub = config.wub 
    wlb = config.wlb
    eta = config.eta
    T = n_iter
    tau_m = config.tau_m
    act_fx = config.act_fx
    dropout_rate = config.dropout_rate
    dkey = random.PRNGKey(1234)
    
    # Initialize logger
    logger = TrainingLogger(log_dir="logs")
    
    data_loader = DataLoader(seq_len=seq_len, batch_size=batch_size)
    train_loader, valid_loader, _ = data_loader.load_and_prepare_data()
    
    model = NGCTransformer(dkey, batch_size=batch_size, seq_len=seq_len, n_embed=n_embed, vocab_size=vocab_size, n_layers=n_layers, n_heads=n_heads,
                          T=T, dt=1., tau_m=tau_m , act_fx=act_fx, eta=eta, dropout_rate=dropout_rate, exp_dir="exp",
                  loadDir=None, pos_learnable=pos_learnable, optim_type=optim_type, wub=wub, wlb=wlb, model_name="ngc_transformer", generate=False)

    num_params = model.count_parameters() / 1e6
    print(f"{num_params:.2f} M parameters")
    
    # Log model parameters
    with open(logger.log_file, 'a') as f:
        f.write(f"Model Parameters: {num_params:.2f} M\n\n")
        f.write("TRAINING PROGRESS:\n")
        f.write("="*80 + "\n")

    def train_model(data_loader, epoch_num):
        train_EFE = 0.
        total_nll, total_tokens = 0., 0

        for batch_idx, batch in enumerate(data_loader):
            inputs = batch[0][1]
            targets = batch[1][1]

            targets_flat = jax.nn.one_hot(targets, vocab_size).reshape(-1, vocab_size)

            _, y_mu, _EFE, L1, L2, L3, L4, L5 = model.process(obs=inputs, lab=targets_flat, adapt_synapses=True)
            train_EFE += _EFE

            y_pred = y_mu.reshape(-1, vocab_size)
            batch_ce_loss = measure_CatNLL(y_pred, targets_flat).mean()
            total_nll += batch_ce_loss * targets_flat.shape[0]
            total_tokens += targets_flat.shape[0]

            if batch_idx % 10 == 0:
                batch_ppl = jnp.exp(batch_ce_loss)
                # Print to console
                print(f"  Batch {batch_idx}: EFE = {_EFE:.4f}, CE = {batch_ce_loss:.4f}, PPL = {batch_ppl:.4f} | L1 = {L1:.4f} | L2 = {L2:.4f} | L3 = {L3:.4f} | L4 = {L4:.4f} | L5 = {L5:.4f}")
                
                # Log to file
                metrics = {
                    'efe': float(_EFE),
                    'ce': float(batch_ce_loss),
                    'ppl': float(batch_ppl),
                    'l1': float(L1),
                    'l2': float(L2),
                    'l3': float(L3),
                    'l4': float(L4),
                    'l5': float(L5)
                }
                logger.log_batch(epoch_num, batch_idx, metrics)

        num_batches = batch_idx + 1
        avg_train_EFE = train_EFE / num_batches
        ce_loss = total_nll / total_tokens
        ppl = jnp.exp(ce_loss)
        return avg_train_EFE, ce_loss, ppl

    start_time = time.time()
    
    # Store metrics for final summary
    final_train_metrics = {}
    final_val_metrics = {}

    for i in range(epoch):
        print(f"\nEpoch {i}:")
        
        # Log epoch start
        with open(logger.log_file, 'a') as f:
            f.write(f"\n--- EPOCH {i} START ---\n")

        avg_train_EFE, train_ce, train_ppl = train_model(train_loader, i)
        
        # Convert to Python floats for logging
        train_ce_val = float(train_ce)
        train_ppl_val = float(train_ppl)
        avg_train_EFE_val = float(avg_train_EFE)
        
        # Evaluate on validation set
        dev_ce, dev_ppl, _ = eval_model(model, valid_loader, vocab_size)
        dev_ce_val = float(dev_ce)
        dev_ppl_val = float(dev_ppl)
        
        # Print to console
        print(f"Epoch {i} Summary: Train CE = {train_ce_val:.4f}, Train PPL = {train_ppl_val:.4f}, Val CE = {dev_ce_val:.4f}, Val PPL = {dev_ppl_val:.4f}, Avg EFE = {avg_train_EFE_val:.4f}")
        
        # Log epoch summary to file
        train_metrics = {'ce': train_ce_val, 'ppl': train_ppl_val, 'efe': avg_train_EFE_val}
        val_metrics = {'ce': dev_ce_val, 'ppl': dev_ppl_val}
        logger.log_epoch_summary(i, train_metrics, val_metrics)
        
        # Store final metrics
        if i == epoch - 1:
            final_train_metrics = train_metrics
            final_val_metrics = val_metrics
        
        # Save model checkpoint (optional - save every epoch or just final)
        # if i % 5 == 0 or i == epoch - 1:
        #     model.save_to_disk(params_only=False)
        
        if i == (epoch - 1):
            model.save_to_disk(params_only=False)  # save final state of model to disk
    
    total_time = time.time() - start_time
    
    # Print to console
    print(f"\nTraining finished.")
    print(f"Total training time: {total_time:.2f} seconds ({total_time/60:.2f} min)")
    
    # Log training completion
    logger.log_training_complete(total_time, final_train_metrics, final_val_metrics)
    
    print(f"\nLog saved to: {logger.log_file}")

if __name__ == "__main__":
    main()