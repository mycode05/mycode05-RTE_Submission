# CS6315E Neural Networks and Deep Learning — RTE Challenge

## 1. Overview and Architecture

This repository solves the Recognizing Textual Entailment (RTE) binary classification task (**entailment** vs. **not_entailment**) using a custom **DeBERTa-v3-base Cross-Encoder** with 183,833,090 total and trainable parameters.

- **Backbone:** `microsoft/deberta-v3-base` (raw pre-trained weights; no external NLI/RTE checkpoints or LLM augmentation used).
- **Pooling and Classification Head:** Attention-masked Mean Pooling across all sequence tokens, followed by 5× Multi-Sample Dropout (p = 0.1, 0.2, 0.3, 0.4, 0.5) sharing a single linear projection (768 → 2).
- **Regularization and Optimization:** Symmetric KL-Divergence R-Drop (alpha = 1.0–2.0), Label Smoothing (0.05), Layer-wise Learning Rate Decay (decay_rate = 0.95), and Cosine learning rate schedule.

## 2. Repository Contents

The GitHub repository contains the following files:

- **README.md** — Project overview, architecture, extraction instructions, and execution guide.
- **train_rte.ipynb** — Training notebook.
- **demo_eval_rte.ipynb** — Evaluation and demo inference notebook.
- **MODEL CARD.pdf** — Model architecture and configuration details.
- **WRITTEN REPORT.pdf** — Project report and experimental analysis.
- **CS6315E_RTE_Models.zip** — Archive containing all three trained model directories.
- **CS6315E_RTE_Final_Submission.zip** — Archive containing final submission artifacts, saved configuration, and tokenizer.
- **Dataset for Challenge-20261017.zip** — Training dataset archive.
- **val.jsonl** — Validation dataset.

## 3. Submitted Weight Variants and Previously Obtained Validation Results

All three submitted weight checkpoints share the same 183,833,090-parameter `DebertaV3EntailmentMSD` architecture.

| Model Checkpoint | Validation Accuracy | Validation F1 |
|---|---:|---:|
| `submission_base_seed42_lr2e5/best_model_weights.pt` | 85.20% | 0.8551 |
| `submission_base_seed100_lr15e5/best_model_weights.pt` | 83.39% | 0.8392 |
| `submission_base_seed2026_lr25e5/best_model_weights.pt` | 84.12% | 0.8571 |

These are the previously obtained validation results for the three trained model variants.

## 4. Project Folder Structure and Extraction Instructions

The submission contains two separate archives for the trained models and final submission artifacts.

### Archive 1: `CS6315E_RTE_Models.zip`

Contains all three trained model directories:

```text
CS6315E_RTE_Models.zip
│
├── submission_base_seed42_lr2e5/
├── submission_base_seed100_lr15e5/
└── submission_base_seed2026_lr25e5/
```

### Archive 2: `CS6315E_RTE_Final_Submission.zip`

Contains the final submission artifacts and supporting files:

```text
CS6315E_RTE_Final_Submission.zip
│
├── final_submission/
├── saved_config/
└── saved_tokenizer/
```

### Extraction Instructions

Follow these steps carefully to arrange the project files correctly.

1. Download both ZIP archives.
2. Extract `CS6315E_RTE_Models.zip` separately. This produces the three trained model directories.
3. Extract `CS6315E_RTE_Final_Submission.zip` separately. This produces the `final_submission`, `saved_config`, and `saved_tokenizer` directories.
4. Create a main folder named `CS6315E_RTE_Submission`.
5. Move all the extracted folders from both archives into this same main folder.

**Important:** Do not place one extracted archive inside the other. Both archives must be extracted separately, and their extracted folders must be placed together in the same directory.

The final project structure should look like this:

```text
CS6315E_RTE_Submission/
│
├── README.md
├── train_rte.ipynb
├── demo_eval_rte.ipynb
│
├── MODEL CARD.pdf
├── WRITTEN REPORT.pdf
│
├── Dataset for Challenge-20261017.zip
├── val.jsonl
│
├── submission_base_seed42_lr2e5/
├── submission_base_seed100_lr15e5/
├── submission_base_seed2026_lr25e5/
│
├── final_submission/
├── saved_config/
└── saved_tokenizer/
```

Preserve all original directory names and their internal contents. The evaluation notebook uses directory paths to locate the model files. Changing the folder names or arrangement may cause file-loading errors.

## 5. How to Reproduce Training Using `train_rte.ipynb` (Google Colab)

### Step A: Open the Notebook and Enable T4 GPU Runtime

1. Open `train_rte.ipynb` in Google Colab.
2. In the top menu bar, click **Runtime → Change runtime type**.
3. Under **Hardware accelerator**, select **T4 GPU** and click **Save**.

### Step B: Upload Dataset Files and Run `train_rte.ipynb`

1. Run the notebook cells sequentially from top to bottom, or select **Runtime → Run all**.
2. When Cell 1 prompts **Click 'Choose Files'**, select the following files from your local machine:
   - `Dataset for Challenge-20261017.zip` — the training dataset archive, which extracts `train.jsonl` into `./rte_data/train.jsonl`.
   - `val.jsonl` — the validation dataset.
3. Cell 2 trains all three model variants using `./rte_data/train.jsonl`, evaluates them on the validation dataset, and saves each `best_model_weights.pt` and `valid_predictions.jsonl`.
4. Cell 3 mounts Google Drive and backs up the trained model weights, tokenizer, configuration, and validation predictions into:

`/content/drive/MyDrive/CS6315E_RTE_Submission`

## 6. How to Run Evaluation / Demo Inference Using `demo_eval_rte.ipynb` (Google Colab)

### Step A: Open the Notebook and Enable T4 GPU Runtime

1. Open `demo_eval_rte.ipynb` in Google Colab.
2. Click **Runtime → Change runtime type → select T4 GPU → Save**.

### Step B: Prepare the Saved Model Files

1. Ensure the Google Drive folder `/content/drive/MyDrive/CS6315E_RTE_Submission` containing all three saved `best_model_weights.pt` files is accessible.
2. Ensure the saved tokenizer and configuration files are available at their expected locations.
3. Preserve the three model directory names exactly as shown in Section 3.

### Step C: Upload Validation and Test Files and Run `demo_eval_rte.ipynb`

1. Select **Runtime → Run all**.
2. When prompted by `files.upload()`, select:
   - `val.jsonl`
   - `test.jsonl`
3. The notebook is intended to perform the following operations:
   - Mount Google Drive and display the **Pre-Test Validation Accuracy and F1** for all three saved models.
   - Load all three saved models (`best_model_weights.pt`) and evaluate them on `test.jsonl`.
   - Print **Test Loss, Test Accuracy, and Test F1** for each model.
   - Save all three prediction files into `./demo_test_predictions/`.

**Note:** The test file `test.jsonl` must be provided separately when running the evaluation notebook.

The evaluation notebook's paths and execution workflow should be verified in Google Colab before final submission.

## 7. Reproducibility and Documentation

All three trained model variants are retained to support evaluation and comparison.

The submission also includes the saved configuration, tokenizer, validation predictions, and final submission artifacts.

The `.ipynb_checkpoints` directory is excluded because it contains automatically generated Jupyter Notebook backup files and is not required for reproducing the experiments.

For additional details, refer to:

- **MODEL CARD.pdf** — Model architecture, training configuration, and other relevant details.
- **WRITTEN REPORT.pdf** — Methodology, experiments, results, and analysis.

---

**Course:** CS6315E – Neural Networks and Deep Learning  
**Task:** Recognizing Textual Entailment (RTE)
