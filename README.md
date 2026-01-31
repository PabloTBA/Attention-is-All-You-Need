# Attention Is All You Need: PyTorch Implementation

This repository contains my personal implementation of the Transformer model from scratch, based on the research paper **"Attention Is All You Need"**. 

The goal of this project is to demonstrate a code-level understanding of the mechanisms that drive modern LLMs, specifically how self-attention replaces traditional recurrence.

## 📄 The Paper
**Title:** Attention Is All You Need  
**Authors:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin  
**Link:** [ArXiv:1706.03762](https://arxiv.org/abs/1706.03762)

---

## 🏗️ Architecture Overview

The Transformer uses an Encoder-Decoder structure. Unlike RNNs, it processes the entire sequence simultaneously, allowing for massive parallelization.

![Transformer Architecture](https://i.sstatic.net/67ADh.png)

---

## 🧠 Key Mechanisms Implemented

### 1. Scaled Dot-Product Attention
This is the core of the model. We calculate attention scores by taking the dot product of Queries ($Q$) and Keys ($K$), scaling them by the square root of the dimension ($\sqrt{d_k}$), and applying a softmax. This determines how much focus to put on the Values ($V$).

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

![Scaled Dot Product Attention](https://miro.medium.com/v2/resize:fit:1024/1*oWcygI-NuySqM6apygn0YQ.png)

### 2. Multi-Head Attention
Instead of performing a single attention function, the model projects the queries, keys, and values into multiple "heads." This allows the model to focus on different positions and representation subspaces simultaneously.

![Multi-Head Attention](https://substackcdn.com/image/fetch/$s_!DpnB!,f_auto,q_auto:good,fl_progressive:steep/https://substack-post-media.s3.amazonaws.com/public/images/910a2088-ea19-4b3a-b016-b69294c36a0f_1000x305.png)

### 3. Layer Normalization & Residuals
To ensure the network can learn deeply without vanishing gradients, every sub-layer (Attention, Feed Forward) is surrounded by a residual connection (adding the input to the output) followed by Layer Normalization.

![Normalization Visual](https://consuledge.com.au/wp-content/uploads/2024/12/batchnorm.png)

### 4. Positional Encoding
Since the model has no recurrence or convolution, it has no inherent sense of order. I implemented sinusoidal positional encodings to inject information about the relative or absolute position of the tokens in the sequence.

![Positional Concept](https://media.licdn.com/dms/image/v2/D5622AQGYdJMGMXXrSw/feedshare-shrink_800/feedshare-shrink_800/0/1729013988538?e=2147483647&v=beta&t=WaXZcx7I0cn2wWBGsh0bVwCTIXtAC2qS7UDxPe0_IiM)

---

## 💭 Personal Implementation Notes
* **Masking:** One of the trickiest parts was implementing the look-ahead mask in the decoder to ensure predictions for position $i$ can only depend on known outputs at positions less than $i$.
* **Tensor Shapes:** Tracking the tensor dimensions $(Batch, Sequence, D\_model)$ through the linear layers and split-heads required careful debugging.

---

## ⚠️ Disclaimer & Image Credits

This repository is for educational purposes. The images used in this README are not my own work and belong to their respective owners. They are used here solely to illustrate the concepts implemented in the code.

**Image Sources:**
* [Architecture Diagram](https://i.sstatic.net/67ADh.png)
* [Attention Mechanism](https://miro.medium.com/v2/resize:fit:1024/1*oWcygI-NuySqM6apygn0YQ.png)
* [Multi-Head Visualization](https://substackcdn.com/image/fetch/$s_!DpnB!,f_auto,q_auto:good,fl_progressive:steep/https://substack-post-media.s3.amazonaws.com/public/images/910a2088-ea19-4b3a-b016-b69294c36a0f_1000x305.png)
* [Normalization Diagram](https://consuledge.com.au/wp-content/uploads/2024/12/batchnorm.png)
* [Concept Summary](https://media.licdn.com/dms/image/v2/D5622AQGYdJMGMXXrSw/feedshare-shrink_800/feedshare-shrink_800/0/1729013988538?e=2147483647&v=beta&t=WaXZcx7I0cn2wWBGsh0bVwCTIXtAC2qS7UDxPe0_IiM)

**Reference Paper:**
* Vaswani, A., et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). 31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA.
