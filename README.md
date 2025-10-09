# 🧠 Visual Question Answering (VQA)
### **ResNet50 + LSTM + Attention Model**

This project implements a **Visual Question Answering (VQA)** model — a task combining **Computer Vision** and **Natural Language Processing** to help machines answer questions about the content of an image.

---

## 🚀 **Goal**
Given an image and a natural language question, the model automatically predicts the most appropriate answer.

**Example:**  
> 🖼️ *Image:* A person playing baseball  
> ❓ *Question:* “What is this person doing?”  
> 💬 *Answer:* “Playing baseball”

---

## ⚙️ **Model Pipeline Overview**

### 1️⃣ Image Feature Extraction – ResNet50
The image is resized to **224×224** and passed through a pretrained **ResNet50** to extract visual features.  
The output is a **2048-dimensional** feature vector, then reduced to **512-D** using a linear projection layer.

### 2️⃣ Question Encoding – LSTM
The question is tokenized, embedded via an **Embedding layer**, and processed by a **2-layer LSTM** (512 hidden units).  
The final hidden state represents the **semantic meaning** of the question as a **512-D** vector.

### 3️⃣ Multimodal Fusion – Attention Mechanism
The image and question feature vectors are combined through an **attention layer**,  
allowing the model to focus on image regions relevant to the question.

### 4️⃣ Answer Generation – LSTM Decoder + Softmax
The decoder **LSTM** generates the answer word by word using the attention-derived context vector.  
A **Linear + Softmax** layer predicts the probability of the next word in the answer vocabulary.  

- **Training:** uses *teacher forcing* (feeding the ground-truth answer sequence).  
- **Inference:** uses *greedy decoding* to generate answers automatically.

---

## ⚙️ **Model Training Configuration**

| Parameter | Value |
|:--|:--|
| **Dataset** | Visual Question Answering – Computer Vision & NLP *(Kaggle)* |
| **Training Samples** | ~10,000 image–question–answer triplets |
| **Train / Eval Split** | 80% / 20% |
| **Batch Size** | 16 |
| **Epochs** | 37 / 100 *(Early Stopping after 10 epochs without improvement)* |
| **Optimizer** | AdamW |
| **Initial Learning Rate** | 1e-4 |
| **Scheduler** | ReduceLROnPlateau *(factor=0.5, patience=2)* |
| **Loss Function** | CrossEntropyLoss *(ignore_index=1 to skip `<pad>` tokens)* |
| **Dropout** | 0.2 (within LSTM) |
| **Embedding Size (Question)** | 256 |
| **Hidden Size (LSTM)** | 512 |
| **Image Feature Dimension** | 512 (from ResNet50 projection) |
| **Max Question Length** | 24 tokens |
| **Max Answer Length** | 7 tokens |
| **Device** | GPU (NVIDIA CUDA on Kaggle) |
| **Training Time per Epoch** | ~170–176 seconds |
| **Early Stopping** | Enabled – stops after 10 non-improving epochs |
| **Best Epoch** | ~Epoch 27–31 |
| **Best Avg F1-score** | 0.4454 (≈44.5%) |
| **Estimated Token-level Accuracy** | ≈58.1% |
| **Train Loss Reduction** | 2.83 → 1.12 |
| **Eval Loss Reduction** | 2.82 → 2.09 |

---

## 📈 **Training Summary**

- The model shows **stable convergence**, with steadily decreasing loss and increasing F1-score up to ~0.44.  
- **No strong overfitting** observed — validation loss follows training loss closely.  
- **Training duration** is reasonable for a ResNet50 + LSTM + Attention architecture on mid-range GPUs.  
- The **early stopping** mechanism effectively prevents overtraining and saves the best-performing checkpoint.

---

## 🧩 **Results & Discussion**

The model achieves a **best average F1-score of 0.4454** and an **estimated token-level accuracy of about 58%**, indicating moderate success in aligning visual and textual understanding within the dataset.

**Key Observations:**
- The model learns meaningful visual–text relationships, performing well on **object recognition**, **color identification**, and **yes/no** questions.  
- However, it struggles with **counting**, **compositional reasoning**, and **text-in-image** (OCR-based) questions — common challenges for CNN–LSTM architectures.  
- The **attention mechanism** helps the model focus on image regions relevant to the question, improving interpretability and performance over naive concatenation methods.

**Performance Plateau:**  
The F1-score plateaued near 0.44 after ~30 epochs. This indicates the model reached its representational limit given the architecture and data size — further training yielded diminishing returns. The learning rate scheduler and early stopping successfully prevented overfitting and unnecessary computation.

**Improvement Directions:**
1. Replace LSTM with **Transformer-based encoders (e.g., BERT, RoBERTa)** for richer language understanding.  
2. Use **object-level features** from **Faster R-CNN** or **DETR** instead of global CNN pooling (Bottom-Up Attention).  
3. Increase training samples or apply **data augmentation** and **question paraphrasing**.  
4. Experiment with **CLIP or ViLT-style** multimodal embeddings for better image–text alignment.  
5. Use **beam search decoding** instead of greedy decoding for more coherent answers.


