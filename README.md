# CS6315E Neural Networks and Deep Learning — RTE Challenge

## 1. Overview and Architecture

This repository solves the Recognizing Textual Entailment (RTE) binary classification task (**entailment** vs. **not_entailment**) using a custom **DeBERTa-v3-base Cross-Encoder** (183,833,090 total and trainable parameters).

- **Backbone:** `microsoft/deberta-v3-base` (raw pre-trained weights; zero external NLI/RTE checkpoints or LLM augmentation used).
- **Pooling and Classification Head:** Attention-masked Mean Pooling across all sequence tokens + 5x Multi-Sample Dropout (p = 0.1, 0.2, 0.3, 0.4, 0.5) sharing a single linear projection (768 to 2).
- **Regularization and Optimization:** Symmetric KL-Divergence R-Drop (alpha = 1.0–2.0), Label Smoothing (0.05), Layer-wise Learning Rate Decay (decay_rate = 0.95), and Cosine learning rate schedule.

## 2. Repository Contents

The GitHub repository contains the following files:

- **README.md** — Project overview, architecture, extraction instructions, and execution guide.
- **CS6315E_RTE_Submission.zip** — Complete project submission containing notebooks, trained model checkpoints, validation data, tokenizer, configuration, report, and model card.
- **Dataset for Challenge-20261017.zip** — Training dataset archive.

### Contents of CS6315E_RTE_Submission.zip

Download and extract `CS6315E_RTE_Submission.zip`. The extracted folder contains:

```text
CS6315E_RTE_Submission/
│
├── MODEL CARD.pdf
├── WRITTEN REPORT.pdf
│
├── train_rte.ipynb
├── demo_eval_rte.ipynb
│
├── val.jsonl
│
├── submission_base_seed42_lr2e5/
├── submission_base_seed2026_lr25e5/
├── submission_base_seed100_lr15e5/
│
├── final_submission/
├── saved_config/
└── saved_tokenizer/
```

### Extraction Instructions

1. Download `CS6315E_RTE_Submission.zip` from the GitHub repository.
2. Extract the ZIP archive using Windows File Explorer, 7-Zip, or WinRAR.
3. Open the extracted `CS6315E_RTE_Submission` folder.
4. Keep the original folder structure and filenames unchanged.
5. Open the required Jupyter notebook in Google Colab.

The `README.md` file is maintained separately in the GitHub repository and is not included inside the submission ZIP.

## 3. Submitted Weight Variants and Previously Obtained Validation Results

All three submitted weight checkpoints share the exact same 183,833,090-parameter `DebertaV3EntailmentMSD` architecture.

1. **submission_base_seed42_lr2e5/best_model_weights.pt** — Validation Accuracy: **85.20%** | Validation F1: **0.8551**
2. **submission_base_seed100_lr15e5/best_model_weights.pt** — Validation Accuracy: **83.39%** | Validation F1: **0.8392**
3. **submission_base_seed2026_lr25e5/best_model_weights.pt** — Validation Accuracy: **84.12%** | Validation F1: **0.8571**

These are the previously obtained validation results for the three trained model variants.

## 4. How to Reproduce Training Using `train_rte.ipynb` (Google Colab)

### Step A: Open `train_rte.ipynb` and Enable T4 GPU Runtime

1. Open the extracted **train_rte.ipynb** notebook in Google Colab.
2. In the top menu bar, click **Runtime → Change runtime type**.
3. Under **Hardware accelerator**, select **T4 GPU** and click **Save**.

### Step B: Upload Dataset Files and Run `train_rte.ipynb`

1. In the top menu bar, click **Runtime → Run all** (or run the cells sequentially from top to bottom).
2. When Cell 1 prompts **Click 'Choose Files'**, select both of the following files together from your local machine:
   - The training ZIP archive (**Dataset for Challenge-20261017.zip**, which automatically extracts `train.jsonl` into `./rte_data/train.jsonl`).
   - The validation file (`val.jsonl`).
3. Cell 2 trains all three weight variants strictly on `./rte_data/train.jsonl`, evaluates them on the validation file, and saves each `best_model_weights.pt` and `valid_predictions.jsonl`.
4. Cell 3 mounts Google Drive and backs up all three trained model weight files, tokenizer, configuration, and validation predictions into:

`/content/drive/MyDrive/CS6315E_RTE_Submission`

## 5. How to Run Evaluation / Demo Inference Using `demo_eval_rte.ipynb` (Google Colab)

### Step A: Open `demo_eval_rte.ipynb` and Enable T4 GPU Runtime

1. Open the extracted **demo_eval_rte.ipynb** notebook in Google Colab.
2. Click **Runtime → Change runtime type → select T4 GPU → Save**.

### Step B: Prepare the Saved Model Files

1. Ensure the Google Drive folder `/content/drive/MyDrive/CS6315E_RTE_Submission` containing the three saved `best_model_weights.pt` files is accessible.
2. Ensure the saved tokenizer and configuration files are available in their expected locations.

### Step C: Upload Validation and Test Files and Run `demo_eval_rte.ipynb`

1. Click **Runtime → Run all**.
2. When prompted by `files.upload()`, select both:
   - `val.jsonl` (or `valid.jsonl`)
   - `test.jsonl`
3. The notebook automatically performs the following operations:
   - Mounts Google Drive and displays the **Pre-Test Validation Accuracy and F1** for all three saved models.
   - Loads all three saved models (`best_model_weights.pt`) and evaluates them on `test.jsonl`.
   - Prints **Test Loss, Test Accuracy, and Test F1** for each model.
   - Saves all three prediction files into `./demo_test_predictions/`.

**Note:** The test file `test.jsonl` must be provided separately when running the evaluation notebook.

## 6. Important Notes

- The complete project submission is available in `CS6315E_RTE_Submission.zip`.
- The training dataset is provided separately as `Dataset for Challenge-20261017.zip`.
- The validation file `val.jsonl` is included in the submission folder.
- The three trained model checkpoints, tokenizer, and configuration files are included in the submission.
- Use the saved checkpoints in `demo_eval_rte.ipynb` to evaluate the previously trained models.
- Running `train_rte.ipynb` again performs a new training run. Exact reproduction of the previously obtained metrics is not guaranteed because of possible differences in execution environments and GPU computation.
- Google Colab with a T4 GPU is recommended for training and evaluation.
