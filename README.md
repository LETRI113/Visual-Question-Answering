# 🧠 Visual Question Answering (VQA) – ResNet50 + LSTM + Attention

## 📘 Introduction
This project implements the problem **Visual Question Answering (VQA)** — a natural question answering system based on images.

Objective: Build a baseline model using **ResNet-50** for images and **LSTM** for questions, then combine it with **Attention mechanism** to predict the answer.

Dataset: [Visual Question Answering – Computer Vision & NLP (Kaggle)](https://www.kaggle.com/datasets/bhavikardeshna/visual-question-answering-computer-vision-nlp)

---

## ⚙️ Method
- **Image feature extraction:** ResNet-50 pretrained on ImageNet, omitting the last fully-connected layer.

- **Question processing:** Tokenization, embedding (256), and encoding with LSTM (512 hidden units).

- **Attention mechanism:** Combine image and question features to select relevant image regions.

- **Decoding:** LSTM decoder predicts the answer sequence.

- **Training:** AdamW optimizer, learning rate 1e-4, dropout 0.3, early stopping.

---

## 🧪 Experiment
- Accuracy (token-level): **≈ 0.59**
- Average F1-score: **≈ 0.44**
- Using baseline model and attention-based model, the results are similar.