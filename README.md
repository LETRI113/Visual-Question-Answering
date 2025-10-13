🧠 Visual Question Answering (VQA) – ResNet50 + LSTM + Attention
📘 Introduction

This project tackles the Visual Question Answering (VQA) problem — combining Computer Vision and Natural Language Processing to predict textual answers from an image-question pair.

Goal: build a baseline VQA model using

ResNet-50 for visual features

LSTM for question encoding

Attention mechanism for multimodal fusion

📂 Dataset:
Visual Question Answering – Computer Vision & NLP (Kaggle)

⚙️ Methodology
🖼️ Image Encoder – ResNet-50

Pretrained on ImageNet.

Removed final classification layer.

Extracted 2048-D visual features representing key image regions.

💬 Question Encoder – LSTM

Text preprocessing: tokenization, padding, vocabulary building.

Embedding size: 256.

Encoded by LSTM with 512 hidden units to capture sequential dependencies.

🎯 Attention Mechanism

Computes similarity between question and image features.

Produces attention weights over image regions.

Weighted visual representation is concatenated with the question vector.

🧩 Answer Decoder

Combines multimodal vector (image + text).

Predicts the answer sequence or class using a fully connected layer.

🧪 Experiment Results

| Metric                        |          Result          |
| :---------------------------- | :----------------------: |
| **Top-1 Validation Accuracy** | **59.04% (3164 / 5359)** |
| **Token-level Accuracy**      |        **0.5962**        |
| **Average F1-score**          |        **0.2834**        |
| **Best Eval Loss**            |        **1.9746**        |
| **Train Loss (final)**        |        **1.1255**        |

🧠 The model shows stable convergence and moderate performance for a baseline.
The attention mechanism improves interpretability and focus on relevant image regions, though gains over the baseline are modest.

⚙️ Training Details

Optimizer: AdamW

Learning rate: 1e-4

Dropout: 0.3

Batch size: 64

Epochs: up to 50 (early stopping after 41)

Loss function: Cross-Entropy

🚀 Future Work

Integrate Transformer-based encoders (ViT + BERT)

Improve F1-score using better text generation decoders

Apply multimodal fusion layers (e.g., MLP + attention heads)

Evaluate using BLEU / ROUGE / CIDEr for long-form answers


   [ẢNH] ─► CNN (ResNet50) ─► image_feat (512)
                         │
                         ▼
   [CÂU HỎI] ─► LSTM Encoder ─► question_feat (512)
                         │
                         ▼
               Concatenate ─► combined_feat (1024)
                         │
                         ▼
        [Decoder hidden state] ───┐
                                  │
                                  ▼
                         ┌───────────────┐
                         │ Attention     │
                         │  (Linear + σ) │
                         └──────┬────────┘
                                ▼
                        context vector (512)
                                │
                                ▼
                      Answer Decoder (LSTM)
                                │
                                ▼
                         Sinh ra câu trả lời
