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

### 1. Self-Attention Mechanism (Scaled Dot-Product)
This is the core of the model. We calculate attention scores by taking the dot product of Queries ($Q$) and Keys ($K$), scaling them by the square root of the dimension ($\sqrt{d_k}$), and applying a softmax. This determines how much focus to put on the Values ($V$).

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

![Scaled Dot Product Attention](https://substackcdn.com/image/fetch/$s_!DpnB!,f_auto,q_auto:good,fl_progressive:steep/https://substack-post-media.s3.amazonaws.com/public/images/910a2088-ea19-4b3a-b016-b69294c36a0f_1000x305.png)

### 2. Multi-Head Attention
Instead of performing a single attention function, the model projects the queries, keys, and values into multiple "heads." This allows the model to focus on different positions and representation subspaces simultaneously.

![Multi-Head Attention](https://miro.medium.com/v2/resize:fit:1024/1*oWcygI-NuySqM6apygn0YQ.png)

### 3. Masked Attention
In the decoder, we must prevent positions from attending to subsequent positions. This "look-ahead mask" (setting future positions to $-\infty$ before softmax) ensures that predictions for position $i$ can depend only on the known outputs at positions less than $i$.

![Masked Attention](https://media.licdn.com/dms/image/v2/D5622AQGYdJMGMXXrSw/feedshare-shrink_800/feedshare-shrink_800/0/1729013988538?e=2147483647&v=beta&t=WaXZcx7I0cn2wWBGsh0bVwCTIXtAC2qS7UDxPe0_IiM)

### 4. Layer Normalization
To ensure the network can learn deeply without vanishing gradients, every sub-layer (Attention, Feed Forward) is surrounded by a residual connection (adding the input to the output) followed by Layer Normalization.

![Normalization Visual](https://i2.wp.com/syncedreview.com/wp-content/uploads/2018/03/image-11-2.png?resize=950%2C256&ssl=1)

### 5. Positional Encoding
Since the model has no recurrence or convolution, it has no inherent sense of order. I implemented sinusoidal positional encodings to inject information about the relative or absolute position of the tokens in the sequence.

![Positional Concept](https://encrypted-tbn1.gstatic.com/images?q=tbn:ANd9GcTxebeKGxo8-hv8PSsSYFAeI8AL6bUnuyZkN2NDZh1zqTPd8C-S)

---

## 💭 Personal Implementation Notes
* **Masking Logic:** Implementing the boolean mask for the decoder was crucial to preserve the auto-regressive property during training.
* **Tensor Shapes:** Tracking the tensor dimensions $(Batch, Sequence, D\_model)$ through the linear layers and split-heads required careful debugging.

---

## ⚠️ Disclaimer & Image Credits

This repository is for educational purposes. The images used in this README are **not my own work** and belong to their respective owners. They are used here solely to illustrate the concepts implemented in the code.

**Image Sources:**
* [Architecture Diagram](https://i.sstatic.net/67ADh.png)
* [Scaled Dot-Product Attention](https://substackcdn.com/image/fetch/$s_!DpnB!,f_auto,q_auto:good,fl_progressive:steep/https://substack-post-media.s3.amazonaws.com/public/images/910a2088-ea19-4b3a-b016-b69294c36a0f_1000x305.png)
* [Multi-Head Attention](https://miro.medium.com/v2/resize:fit:1024/1*oWcygI-NuySqM6apygn0YQ.png)
* [Masked Attention](https://media.licdn.com/dms/image/v2/D5622AQGYdJMGMXXrSw/feedshare-shrink_800/feedshare-shrink_800/0/1729013988538?e=2147483647&v=beta&t=WaXZcx7I0cn2wWBGsh0bVwCTIXtAC2qS7UDxPe0_IiM)
* [Layer Normalization](https://i2.wp.com/syncedreview.com/wp-content/uploads/2018/03/image-11-2.png?resize=950%2C256&ssl=1)
* [Positional Encoding](https://encrypted-tbn1.gstatic.com/images?q=tbn:ANd9GcTxebeKGxo8-hv8PSsSYFAeI8AL6bUnuyZkN2NDZh1zqTPd8C-S)

**Reference Paper:**
* Vaswani, A., et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). 31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA.
