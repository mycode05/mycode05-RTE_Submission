
# CS6315E – Recognizing Textual Entailment (RTE)

A Neural Networks and Deep Learning project implementing binary Recognizing Textual Entailment using a custom **DeBERTa-v3-base Cross-Encoder** architecture.

## 1. Project Overview

Given a premise and a hypothesis, the model determines whether the hypothesis is entailed by the premise.

| Class | Description |
|---|---|
| `entailment` | The hypothesis follows from the premise. |
| `not_entailment` | The hypothesis does not follow from the premise. |

Three model configurations were trained using different random seeds and learning rates.

## 2. Model Architecture

**Base model:** `microsoft/deberta-v3-base`

| Component | Configuration |
|---|---|
| Backbone | DeBERTa-v3-base |
| Pooling | Attention-mask-aware mean pooling |
| Classification head | Linear layer (768 → 2) |
| Multi-Sample Dropout | 5 branches: 0.1, 0.2, 0.3, 0.4, 0.5 |
| Regularization | Symmetric KL-divergence-based R-Drop |
| Label smoothing | 0.05 |
| Layer-wise LR decay | 0.95 |
| LR scheduler | Cosine learning-rate schedule |

The model uses pretrained base-model weights. No external fine-tuned NLI/RTE checkpoints or LLM-generated training data were used.

## 3. Model Configurations and Validation Results

| Model | Seed | Learning Rate | Accuracy | F1 Score |
|---|---:|---:|---:|---:|
| Model 1 | 42 | 2e-5 | 85.20% | 0.8551 |
| Model 2 | 100 | 1.5e-5 | 83.39% | 0.8392 |
| Model 3 | 2026 | 2.5e-5 | 84.12% | 0.8571 |

## 4. Repository Structure

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
├── CS6315E_RTE_Final_Submission.zip
├── val.jsonl
│
├── submission_base_seed42_lr2e5/
├── submission_base_seed100_lr15e5/
└── submission_base_seed2026_lr25e5/
```

**Note:** The three trained model checkpoint directories are stored in Google Drive rather than this GitHub repository because of their large size.

## 5. Trained Model Storage – Google Drive

The three trained model directories are available in the following Google Drive location:

```text
MyDrive/
└── CS6315E_RTE_Submission/
    ├── submission_base_seed42_lr2e5/
    ├── submission_base_seed100_lr15e5/
    └── submission_base_seed2026_lr25e5/
```

Each directory contains the corresponding trained model weights and associated files.

The evaluation notebook accesses these directories directly after Google Drive is mounted in Colab.

**Important:** Preserve the directory names and folder structure. The evaluation notebook uses these paths to load the models.

## 6. Training

Notebook: `train_rte.ipynb`

1. Open the notebook in Google Colab.
2. Select a GPU runtime.
3. Upload the training dataset ZIP and `val.jsonl` when prompted.
4. Execute the notebook cells in order.
5. Train the three model configurations.
6. Save the trained model checkpoints and validation predictions.
7. Back up the trained model directories to Google Drive.

A GPU runtime is recommended.

## 7. Evaluation

Notebook: `demo_eval_rte.ipynb`

1. Open the notebook in Google Colab.
2. Mount Google Drive.
3. Ensure the three trained model directories are present in `MyDrive/CS6315E_RTE_Submission/`.
4. Provide the required `val.jsonl` and `test.jsonl` files when prompted.
5. Execute the notebook cells in order.
6. The notebook loads all three models and performs evaluation.
7. Test predictions are saved in the output directory specified in the notebook.

The notebook also displays the validation results for comparison.

## 8. Final Submission Archive

**Archive:** `CS6315E_RTE_Final_Submission.zip`

The archive contains:

```text
CS6315E_RTE_Final_Submission/
├── final_submission/
├── saved_config/
└── saved_tokenizer/
```

These are the final submission components, including the model submission files, configuration, and tokenizer.

## 9. Reproducibility

The repository provides the following resources:

- Training and evaluation notebooks.
- Training dataset archive and validation data.
- Model card and written report.
- Final submission archive.
- Trained model checkpoints stored separately in Google Drive.

The training notebook can be used to retrain the models, while the evaluation notebook loads the saved checkpoints from Google Drive.

---

**Course:** CS6315E – Neural Networks and Deep Learning
**Task:** Recognizing Textual Entailment (RTE)

This version keeps the final submission ZIP and does not mention the models ZIP anywhere.
