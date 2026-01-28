# Domain-Adversarial Training of Neural Networks (DANN) 

### Deep Learning Educational Demo - Unsupervised Domain Adaptation

This repository contains a clean, modular PyTorch implementation of the paper **"Domain-Adversarial Training of Neural Networks"** (Ganin et al., 2016).

The goal of this project is to demonstrate how a neural network can learn to generalize to a **Target Domain** without having any labels for it, by using adversarial training to enforce **domain invariance**.

## The Problem: Domain Shift

When we train a model on "clean" data (e.g., **MNIST**) and test it on "noisy" or visually different data (e.g., **MNIST-M**, digits on colored backgrounds), performance usually drops significantly due to the difference in feature distributions.

* **Source Domain ():** Standard MNIST Digits (Labeled).
* **Target Domain ():** Synthetic MNIST-M (Unlabeled during training).

## The Solution: DANN & Gradient Reversal

This implementation uses an architecture with three key components:

1. **Feature Extractor:** Learns the deep representation of images.
2. **Label Predictor:** Classifies the digit (0-9).
3. **Domain Classifier:** Tries to predict if the image comes from the Source or the Target.

✨ **The Magic Trick:** We use a **Gradient Reversal Layer (GRL)** . During the backward pass, this layer multiplies the gradient flowing from the domain classifier by a negative constant (). This forces the Feature Extractor to learn features that are **indistinguishable** between domains (confusing the discriminator) while still being useful for digit classification.

## 📂 Project Structure

```text
DeepLearning-Demo/
│
├── data/
│   └── mnist_m.py          # On-the-fly synthetic MNIST-M generator
├── models/
│   ├── dann.py             # Main Architecture (CNN + Branches)
│   └── grl.py              # Gradient Reversal Layer implementation
├── notebooks/
│   └── tsne_animation.ipynb # Visual Demo: t-SNE training animation
├── utils/
│   ├── lambda_schedule.py  # Dynamic scheduling for the adaptation parameter
│   ├── tsne_utils.py       # Feature extraction helper for visualization
│   └── training_utils.py   # Metrics and accuracy calculation
├── train_baseline.py       # Control script (Source Only training)
├── train_dann.py           # Main experiment script (Adversarial training)
└── requirements.txt        # Dependencies

```

## Installation & Usage

### 1. Setup Environment

Cloning the repository and installing dependencies:

```bash
git clone https://github.com/agurban01/DeepLearning-Demo.git
cd DeepLearning-Demo

# Create virtual environment (Recommended)
python -m venv venv
# Activate (Windows): venv\Scripts\activate
# Activate (Mac/Linux): source venv/bin/activate

# Install requirements
pip install -r requirements.txt

```

### 2. Run Baseline (Control Group)

First, train a standard model without domain adaptation to establish a baseline.

```bash
python train_baseline.py

```

> **Expected Result:** High accuracy on Source (MNIST, >98%) but very poor accuracy on Target (MNIST-M, ~30-50%). This proves the domain shift problem exists.

### 3. Run DANN (The Experiment)

Now, train the model using the Gradient Reversal Layer.

```bash
python train_dann.py

```

> 
> **Expected Result:** As the adaptation parameter `alpha` increases , the network aligns the distributions. Target accuracy should improve significantly (often reaching 70-80%+) without using any target labels.
> 
> 

### 4. Interactive Visualization (Extension)

To see the "learning process" in action, run the Jupyter Notebook. It generates a **t-SNE animation** showing how Source (Blue) and Target (Red) feature distributions merge over time.

```bash
jupyter notebook notebooks/tsne_animation.ipynb

```

## Originality & Extension

Beyond the standard implementation, this project includes:

1. **Dynamic Data Generation:** Instead of relying on large, static dataset downloads (like BSDS500), `data/mnist_m.py` generates MNIST-M samples *on-the-fly* using randomized RGB noise and inversion. This makes the code lightweight and immediately reproducible.
2. **Live Latent Space Animation:** The project goes beyond static plots by implementing a training loop that captures latent space snapshots to animate the alignment process using t-SNE, providing a pedagogical view of *how* the network learns invariance.

## References

* **Original Paper:** Ganin, Y., et al. (2016). *Domain-Adversarial Training of Neural Networks*. Journal of Machine Learning Research. [PDF](https://arxiv.org/abs/1505.07818)
* **Theoretical Foundation:** Ben-David et al. (2010). *A theory of learning from different domains*.

