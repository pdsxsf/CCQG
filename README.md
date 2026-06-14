# CCQG: Multi-Task BERT-BiLSTM-CRF for Chinese NER

This repository implements a multi-task Named Entity Recognition (NER) system for Chinese text, built on a **BERT-BiLSTM-CRF** architecture with a **Label Attention** mechanism and a gating-based multi-task fusion strategy. The model jointly learns two labeling tasks (e.g., NER tags and POS/syntactic tags) and fuses their outputs through a learned gate to improve primary task performance.

## Architecture Overview

The core model (`BERT_BiLSTM_CRF_MUL` in `models.py`) consists of the following components:

- **BERT Encoder**: Uses `bert-base-chinese` as the backbone to produce contextual token representations.
- **Optional BiLSTM Layer**: A bidirectional LSTM layer (`rnn_dim=128` by default) that captures sequential dependencies on top of BERT outputs.
- **Label Attention**: A custom attention module (`LabelAttention` in `attentions.py`) that computes label-aware representations by projecting hidden states against learnable label embeddings.
- **Multi-Task Heads**: Two separate CRF heads — one for the primary NER task (`crf1`) and one for the auxiliary task (`crf2`). A secondary projection (`l2tol1`) maps auxiliary predictions back to the primary label space.
- **Gating Fusion**: A softmax-gated combination of three expert signals (primary emission, label attention output, and auxiliary-derived emission) produces the final emission scores for the primary CRF decoder.

The total training loss is: `loss = loss1 + lambda * loss2`, where `lambda` (default `0.2`) controls the contribution of the auxiliary task.

## Project Structure

```
CCQG-main/
├── ner.py               # Main training, evaluation, and testing entry point
├── models.py            # Model definitions (BERT_BiLSTM_CRF, BERT_BiLSTM_CRF_MUL)
├── attentions.py        # Attention mechanism implementations
├── utils.py             # Data loading, feature conversion, NerProcessor
├── conlleval.py         # CoNLL-style evaluation script (precision, recall, F1)
├── clue_process.py      # CLUE benchmark data preprocessing utility
├── run.sh               # Shell script for full training + evaluation + testing
├── only_test.sh         # Shell script for testing with a pre-trained model
├── requirements.txt     # Python dependency specifications
└── data/
    └── ccqg/            # Dataset directory (train/dev/test splits)
```

## Data Format

The input data follows a **BIO-style tab-separated format**, with one character per line and blank lines separating sentences:

```
character    label1    label2
```

For example:

```
课	O	B-NN
外	O	M-NN
活	O	M-NN
动	O	E-NN
```

- **label1**: The primary task label (e.g., binary indicator tags such as `O` and `B`).
- **label2**: The auxiliary task label (e.g., BMES-tagged POS/syntactic categories like `B-NN`, `M-NN`, `E-NN`, `S-VC`).

Sentences are separated by empty lines. The data files in `data/ccqg/` include:

| File | Description |
|------|-------------|
| `train_mul.txt` | Training set |
| `dev_mul.txt` | Development/validation set |
| `test_mul.txt` | Test set |
| `train_mul2.txt` | Training set (alternate split) |
| `dev_mul2.txt` | Development set (alternate split) |
| `test_mul2.txt` | Test set (alternate split) |
| `toy.txt` | Small toy dataset for quick testing |

> **Note**: The shell scripts (`run.sh`, `only_test.sh`) reference `train.txt`, `dev.txt`, and `test.txt`. You may need to rename the `*_mul.txt` files accordingly or update the paths in the scripts to match the actual filenames.

## Requirements

### Python Dependencies

All dependencies are listed in `requirements.txt`:

```
numpy~=1.21.2
torch~=1.10.0
tqdm~=4.64.0
tensorboardX~=2.5.1
pytorchcrf~=1.2.0
pytorch-transformers==1.2.0
```

Install them with:

```bash
pip install -r requirements.txt
```

> **Important**: This project uses `pytorch-transformers` (v1.2.0), which is the legacy Hugging Face Transformers library. This version is significantly older than the current `transformers` package and may not be compatible with newer Python versions (3.10+). It is recommended to use Python 3.7–3.9.

### Pre-trained Model

The model requires the `bert-base-chinese` pre-trained weights. Download them from Hugging Face and place them in the `./pretrained_bert/bert-base-chinese/` directory. The directory should contain:

- `pytorch_model.bin` (or `model.safetensors`)
- `config.json`
- `vocab.txt`

### Hardware

Training requires a CUDA-compatible GPU. The code is designed to run on a single GPU by default (`device = torch.device("cuda")`). Multi-GPU support via `DataParallel` is available when multiple GPUs are detected.

## Usage

### Full Training Pipeline (Train + Eval + Test)

```bash
python ner.py \
    --model_name_or_path ./pretrained_bert/bert-base-chinese \
    --do_train True \
    --do_eval True \
    --do_test True \
    --max_seq_length 256 \
    --train_file ./data/ccqg/train.txt \
    --eval_file ./data/ccqg/dev.txt \
    --test_file ./data/ccqg/test.txt \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --num_train_epochs 50 \
    --do_lower_case \
    --logging_steps 200 \
    --need_birnn True \
    --rnn_dim 256 \
    --clean True \
    --output_dir ./output
```

Or simply run the provided script:

```bash
bash run.sh
```

### Testing Only (with a Pre-trained Model)

```bash
CUDA_VISIBLE_DEVICES=6 python ner.py \
    --model_name_or_path ./pretrained_bert/bert-base-chinese \
    --do_train False \
    --do_eval False \
    --do_test True \
    --max_seq_length 128 \
    --train_file ./data/ccqg/train.txt \
    --eval_file ./data/ccqg/dev.txt \
    --test_file ./data/ccqg/test.txt \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --num_train_epochs 100 \
    --do_lower_case \
    --logging_steps 200 \
    --need_birnn True \
    --rnn_dim 256 \
    --clean True \
    --output_dir ./output
```

Or:

```bash
bash only_test.sh
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max_seq_length` | 256 | Maximum input sequence length |
| `--train_batch_size` | 8 | Training batch size |
| `--learning_rate` | 3e-5 | Initial learning rate (AdamW) |
| `--num_train_epochs` | 10 | Number of training epochs |
| `--need_birnn` | False | Whether to include the BiLSTM layer |
| `--rnn_dim` | 128 | BiLSTM hidden dimension (per direction) |
| `--lbd` | 0.2 | Weight of the auxiliary task loss |
| `--warmup_steps` | 0 | Number of linear warmup steps |
| `--seed` | 2019 | Random seed (parsed but not used; see [Reproducibility](#reproducibility)) |

### Monitoring Training

Training and evaluation metrics are logged via TensorBoard (using `tensorboardX`). Logs are saved in `./log/log<timestamp>/`. View them with:

```bash
tensorboard --logdir ./log/
```

### Model Output

After training, the best model (based on dev F1) is saved to `--output_dir` along with:

- `pytorch_model.bin` — Model weights
- `config.json` — Model configuration
- `vocab.txt` — Tokenizer vocabulary
- `training_args.bin` — Serialized training arguments
- `label2id.pkl` — Label-to-index mapping
- `label_list.pkl` — Label sets for both tasks
- `token_labels_.txt` — Per-token predictions on the test set

## Reproducibility

This section provides detailed instructions and notes for reproducing the experimental results.

### Random Seed Configuration

> **Important**: The `set_seed()` function defined in `ner.py` is **never called** during execution. The `--seed` argument (default `2019`) is parsed but has **no effect** on the random state.

The only seed that is actually applied is a **hardcoded seed** (`1024`) at the beginning of `main()`, before device initialization:

```python
seed = 1024
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

This means:

- Python's `random` module and NumPy's RNG are **not seeded**, so data loading order (if any shuffling depends on them) may vary between runs.
- The `--seed` CLI argument is accepted but unused — changing it will not affect results.
- The hardcoded seed `1024` controls PyTorch's CPU and CUDA RNG at the point of model weight initialization.

To make the `--seed` argument effective, add a call to `set_seed(args)` after argument parsing in `main()`:

```python
args = parser.parse_args()
set_seed(args)  # Add this line
```

Or, to ensure full reproducibility with a single fixed seed, add the following at the top of `main()`:

```python
random.seed(1024)
np.random.seed(1024)
torch.manual_seed(1024)
torch.cuda.manual_seed_all(1024)
```

### cuDNN Determinism

The current code does **not** enforce cuDNN deterministic behavior. If you require bitwise reproducibility across runs, add the following to `ner.py`:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Note that enabling deterministic mode may reduce training speed.

### Environment Setup for Reproduction

To create a reproducible environment:

```bash
# Create a conda environment with a compatible Python version
conda create -n ccqg python=3.8 -y
conda activate ccqg

# Install CUDA-compatible PyTorch (adjust for your CUDA version)
pip install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# Install remaining dependencies
pip install -r requirements.txt
```

The key version constraints that affect model behavior are:

- `torch==1.10.0` — Different PyTorch versions may produce different results due to changes in CUDA kernels, default initializations, and CRF implementations.
- `pytorch-transformers==1.2.0` — This legacy library loads BERT weights differently from modern `transformers`. Using a different version may result in incompatible weight loading or different tokenizer behavior.
- `pytorchcrf~=1.2.0` — The CRF layer implementation; version differences could affect decoding results.

### Step-by-Step Reproduction

1. **Prepare the environment** following the instructions above.

2. **Download `bert-base-chinese`** and place it in `./pretrained_bert/bert-base-chinese/`.

3. **Prepare the data**: Ensure the data files are in `./data/ccqg/`. If using the provided scripts, either rename `train_mul.txt` → `train.txt`, `dev_mul.txt` → `dev.txt`, `test_mul.txt` → `test.txt`, or update the file paths in `run.sh`.

4. **Run training**:

   ```bash
   bash run.sh
   ```

   Or run `ner.py` directly with the arguments shown in the Usage section.

5. **Check results**: Evaluation metrics (precision, recall, F1) are printed to stdout during both validation (per epoch) and testing phases. The best model checkpoint is saved based on dev F1 score.

### Known Reproducibility Issues

- **`set_seed()` is never invoked**: The `set_seed()` function is defined in `ner.py` but never called. The `--seed` argument is parsed but has no effect. Only the hardcoded seed `1024` applies. See [Random Seed Configuration](#random-seed-configuration) for details and suggested fixes.
- **Python version sensitivity**: The `pytorch-transformers==1.2.0` package may fail to install on Python 3.10+. Use Python 3.7–3.9 for best compatibility.
- **GPU architecture**: Results may vary slightly across different GPU architectures (e.g., V100 vs. A100) due to differences in CUDA kernel implementations, even with the same seed and deterministic mode enabled.
- **Data file naming**: The mismatch between script paths (`train.txt`) and actual data filenames (`train_mul.txt`) must be resolved before running.
- **Label ordering**: The label-to-id mapping is generated dynamically from the training data (via set iteration) on first run and cached as `label_list.pkl` and `label2id.pkl`. If these files are deleted and regenerated, the mapping order may differ, potentially affecting results. Always reuse the cached files or provide them alongside the model for consistent evaluation.

### Expected Resource Usage

- **Training time**: Approximately several hours on a single GPU (depends on dataset size and GPU model). The default configuration uses batch size 64, 50 epochs, and max sequence length 256.
- **GPU memory**: With `bert-base-chinese` (768-dim), BiLSTM (`rnn_dim=256`), batch size 64, and max sequence length 256, expect to use approximately 10–14 GB of GPU memory.
- **Disk space**: The pre-trained BERT model requires ~400 MB. Training logs, model checkpoints, and output files require additional space depending on the number of saved checkpoints.
