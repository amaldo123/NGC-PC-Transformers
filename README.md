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

## Fast Standalone Inference Decoder

Generation historically called `model.process()`, which executed the full training-time predictive coding circuit: `reset.run()`, 52 parameter copies from the training model into the projection model, error cell updates, transposed feedback synapse updates, and $T=26$ iterative relaxation steps over a full padded `batch_size x seq_len` window for every single generated token.

The standalone inference decoder in `inference/decoder.py` isolates the exact feedforward predictive coding projection path (`y_mu_inf = projection.q_target_Ratecell.zF`) in pure JAX with $O(1)$ KV-cached decoding and zero imports of `model.py` at runtime.

### 1. One-Time Weight Export
Extract parameters and architecture metadata from a trained checkpoint into compact standalone files:
```bash
python export_generation_weights.py --exp-dir exp --model-name ngc_transformer
```
This produces:
- `generation_weights.npz`: packed parameter arrays (float32).
- `generation_meta.json`: architecture hyper-parameters and activation configurations.

*(Note: `generation.py` will automatically trigger this export if the files are not already present.)*

### 2. Fast Text Generation
```bash
# Fast generation using the standalone decoder (default)
python generation.py --max-new-tokens 200 --temperature 0.8 --top-k 50

# Deterministic greedy generation
python generation.py --max-new-tokens 100 --temperature 0.0

# Legacy fallback (calls NGCTransformer.process)
python generation.py --legacy --max-new-tokens 50
```

### 3. Benchmarking
Benchmark token throughput over multiple runs:
```bash
# Benchmark the fast standalone decoder
python generation.py --benchmark --max-new-tokens 200 --runs 3

# Benchmark the legacy model
python generation.py --legacy --benchmark --max-new-tokens 20 --runs 3
```

**Measured CPU Performance (TinyShakespeare Checkpoint, 4 Layers, 8 Heads, Dim 128):**
| Decoder Implementation | Throughput (Median) | Peak Throughput | Complexity per Token | Memory Stability |
| :--- | :--- | :--- | :--- | :--- |
| **Legacy `model.process()`** | **0.78 tok/s** | 0.78 tok/s | $O(T \cdot B \cdot S^2)$ with 52 weight copies | OOM after ~35 steps |
| **Fast Standalone Decoder** | **149.79 tok/s** | **155.34 tok/s** | $O(1)$ KV Cache step | 0 leaks, stable for 1000s of tokens |

*Achieves a **~199x speedup** on CPU while guaranteeing 100% numerical and token-for-token parity with the projection model.*

### 4. Running the Equivalence Test Suite
Run the comprehensive numerical verification suite covering per-node intermediate activations, softmax probabilities, incremental KV cache parity, sliding-window boundary transitions, and multi-architecture configurations:
```bash
python -m unittest tests/test_decoder_equivalence.py -v
```
All 6 test cases enforce a strict numerical difference tolerance of $\le 10^{-4}$.

