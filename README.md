# CS6315E Neural Networks and Deep Learning — RTE Challenge

## 1\. Overview and Architecture

This repository solves the Recognizing Textual Entailment (RTE) binary classification task (**entailment** vs. **not\_entailment**) using a custom **DeBERTa-v3-base Cross-Encoder** (**183,833,090** total and trainable parameters).

- **Backbone:** **microsoft/deberta-v3-base** (raw pre-trained weights; zero external NLI/RTE checkpoints or LLM augmentation used)  
- **Pooling and Classification Head:** Attention-masked Mean Pooling across all sequence tokens \+ 5x Multi-Sample Dropout (**p \= 0.1, 0.2, 0.3, 0.4, 0.5**) sharing a single linear projection (**768 to 2**)  
- **Regularization and Optimization:** Symmetric KL-Divergence R-Drop (**alpha \= 1.0 \- 2.0**), Label Smoothing (**0.05**), Layer-wise Learning Rate Decay (**decay\_rate \= 0.95**), and Cosine learning rate schedule

## 2\. Submitted Weight Variants (Same Architecture)

All three submitted weight checkpoints share the exact same **183,833,090**\-parameter **DebertaV3EntailmentMSD** architecture:

1. **submission\_base\_seed42\_lr2e5/best\_model\_weights.pt** — **Valid Accuracy: 85.20% \| Valid F1: 0.8551**  
2. **submission\_base\_seed100\_lr15e5/best\_model\_weights.pt** — **Valid Accuracy: 83.39% \| Valid F1: 0.8392**  
3. **submission\_base\_seed2026\_lr25e5/best\_model\_weights.pt** — **Valid Accuracy: 84.12% \| Valid F1: 0.8571**

&nbsp;

## 3\. How to Reproduce Training (Google Colab)

### Step A: Enable T4 GPU Runtime

1. Open the training notebook (**train\_rte.ipynb**) in **Google Colab**.  
2. In the top menu bar, click **Runtime** \-\> **Change runtime type**.  
3. Under **Hardware accelerator**, select **T4 GPU** and click **Save**.

### Step B: Upload Dataset Files and Run All Cells

1. In the top menu bar, click **Runtime** \-\> **Run all** (or run the cells sequentially from top to bottom).  
2. When **Cell 1** prompts **Click 'Choose Files'**, select **both** of the following files together from your local machine:  
   - The training zip archive (**Dataset for Challenge-20261017.zip**, which automatically extracts **train.jsonl** into **./rte\_data/train.jsonl**)  
   - The validation file (**val.jsonl**)  
3. **Cell 2** trains all 3 weight variants strictly on **./rte\_data/train.jsonl**, evaluates them on **val.jsonl**, and saves each **best\_model\_weights.pt** and **valid\_predictions.jsonl**.  
4. **Cell 3** mounts Google Drive and backs up all 3 trained model weight files, tokenizer, config, and validation predictions into **/content/drive/MyDrive/CS6315E\_RTE\_Submission**.

&nbsp;

## 4\. How to Run Evaluation / Demo Inference on Test Data (Google Colab)

### Step A: Enable T4 GPU Runtime

1. Open the evaluation/demo notebook (**demo\_eval\_rte.ipynb**) in **Google Colab**.  
2. Click **Runtime** \-\> **Change runtime type** \-\> select **T4 GPU** \-\> click **Save**.

### Step B: Upload Validation and Test Files and Run All Cells

1. Ensure the Google Drive folder **/content/drive/MyDrive/CS6315E\_RTE\_Submission** containing the 3 saved **best\_model\_weights.pt** files is accessible.  
2. Click **Runtime** \-\> **Run all**.  
3. When prompted by **files.upload()**, select **both** **val.jsonl** (or **valid.jsonl**) and **test.jsonl**.  
4. The notebook automatically performs the following steps:  
   - Mounts Google Drive and displays the **Pre-Test Validation Accuracy and F1** for all 3 saved models.  
   - Loads all 3 saved models (**best\_model\_weights.pt**) and evaluates them on **test.jsonl**.  
   - Prints **Test Loss, Test Accuracy, and Test F1** for each model and saves all 3 prediction files into **./demo\_test\_predictions/**.
