RTE (Recognizing Textual Entailment) – Neural Networks and Deep Learning

1. Project Overview

This project implements a binary classification system for the Recognizing Textual Entailment (RTE) task.

Given a premise and a hypothesis, the model predicts whether the hypothesis is entailed by the premise.

Classes:

- "entailment"
- "not_entailment"

The project uses a custom DeBERTa-v3-base Cross-Encoder architecture and evaluates three independently trained model configurations.

2. Model Architecture

The proposed architecture is based on "microsoft/deberta-v3-base".

Key components:

- Backbone: DeBERTa-v3-base, initialized with pretrained base-model weights.
- Pooling: Attention-mask-aware mean pooling.
- Multi-Sample Dropout: Five dropout branches with probabilities 0.1, 0.2, 0.3, 0.4, and 0.5, sharing a common linear classification layer.
- Classification Head: Linear projection from 768 dimensions to 2 classes.
- Regularization: Symmetric KL-divergence-based R-Drop.
- Additional techniques: Label smoothing, layer-wise learning-rate decay, and cosine learning-rate scheduling.

No external NLI/RTE fine-tuned checkpoints or LLM-generated training data were used.

3. Trained Model Configurations and Validation Results

Three model configurations were trained using different random seeds and learning rates.

Model| Random Seed| Learning Rate| Validation Accuracy| Validation F1
Model 1| 42| 2e-5| 85.20%| 0.8551
Model 2| 100| 1.5e-5| 83.39%| 0.8392
Model 3| 2026| 2.5e-5| 84.12%| 0.8571

These are the validation results obtained during training.

4. Repository Structure

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

The three trained model checkpoint directories are stored separately in Google Drive because of their large size. They are not included in this GitHub repository.

5. Google Drive – Trained Model Storage

All three trained model checkpoint folders are stored in Google Drive at:

MyDrive/
└── CS6315E_RTE_Submission/
    │
    ├── submission_base_seed42_lr2e5/
    ├── submission_base_seed100_lr15e5/
    └── submission_base_seed2026_lr25e5/

Each folder contains the corresponding trained model weights and associated files required for evaluation.

To reproduce the evaluation:

1. Open Google Colab.
2. Mount Google Drive.
3. Ensure the three model directories are available at the path shown above.
4. Upload the required "val.jsonl" and "test.jsonl" files when prompted by the evaluation notebook.
5. Run "demo_eval_rte.ipynb".

Important: Keep the directory names and folder structure unchanged because the evaluation notebook uses these paths to load the models.

6. Training Instructions

The training notebook is:

"train_rte.ipynb"

Steps:

1. Open the notebook in Google Colab.
2. Select a GPU runtime.
3. Upload the training dataset ZIP and "val.jsonl" when prompted.
4. Run the notebook cells in order.
5. The notebook trains the three model configurations.
6. The best model weights and validation predictions are saved.
7. The trained model folders can be backed up to Google Drive.

A GPU runtime is recommended for training.

7. Evaluation Instructions

The evaluation notebook is:

"demo_eval_rte.ipynb"

Steps:

1. Open the notebook in Google Colab.
2. Mount Google Drive.
3. Confirm that all three trained model folders are present in "MyDrive/CS6315E_RTE_Submission/".
4. Provide the required validation and test JSONL files.
5. Run the notebook cells in order.
6. The notebook loads all three trained models and performs evaluation.
7. Test predictions are saved in the output directory specified in the notebook.

The notebook also displays the stored validation results for comparison.

8. Final Submission Artifacts

The repository includes:

"CS6315E_RTE_Final_Submission.zip"

This archive contains:

- "final_submission/"
- "saved_config/"
- "saved_tokenizer/"

These files contain the final submission components and the configuration and tokenizer files required by the submission workflow.

9. Reproducibility

For reproducibility, the repository provides:

- Training notebook ("train_rte.ipynb").
- Evaluation notebook ("demo_eval_rte.ipynb").
- Training dataset archive.
- Validation dataset ("val.jsonl").
- Model card and written report.
- Final submission archive ("CS6315E_RTE_Final_Submission.zip").
- Three trained model checkpoint folders stored in Google Drive.

The training notebook can be used to retrain the models, while the evaluation notebook can load the saved checkpoints from Google Drive to reproduce the evaluation workflow.

---

Course: CS6315E – Neural Networks and Deep Learning
Task: Recognizing Textual Entailment (RTE)9
