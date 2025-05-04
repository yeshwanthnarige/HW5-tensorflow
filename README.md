# HW5-tensorflow

Name : Yeshwanth Narige  
Id: 700764035

# MNIST GAN and Sentiment Classifier Data Poisoning Simulation

## Overview

This repository contains two self‑contained experiments demonstrating:

1. **Basic GAN on MNIST**: A PyTorch implementation of a vanilla Generative Adversarial Network (GAN) that generates handwritten digits from the MNIST dataset.
2. **Data Poisoning Simulation**: A scikit‑learn demonstration of how flipping labels for a specific entity (`"UC Berkeley"`) in a sentiment analysis task can degrade model performance and illustrate allocational harm.

Each subproject includes source code, instructions, and sample outputs (images and plots).

---

## Directory Structure

```
/
├── mnist_gan.py           # PyTorch GAN implementation
├── poisoning_simulation.py# Scikit‑learn sentiment poisoning demo
├── requirements.txt       # Python dependencies
├── mnist_gan_outputs/     # Generated MNIST images & loss curves
│   ├── epoch_0.png
│   ├── epoch_50.png
│   ├── epoch_100.png
│   └── loss_curve.png
└── poisoning_confusion.png# Confusion matrices before/after poisoning
```

---

## Requirements

* **Python 3.7+**
* **PyTorch**, **torchvision**
* **scikit-learn**
* **matplotlib**

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. MNIST GAN

1. Run the GAN script:

   ```bash
   python mnist_gan.py
   ```
2. The script will:

   * Train a Generator and Discriminator for 100 epochs.
   * Save sample grids of generated digits at epochs **0**, **50**, and **100** under `mnist_gan_outputs/`.
   * Output a `loss_curve.png` showing Generator vs Discriminator loss.

### 2. Data Poisoning Simulation

1. Run the poisoning demo:

   ```bash
   python poisoning_simulation.py
   ```
2. The script will:

   * Train a logistic regression on clean movie‑review data.
   * Flip labels for any sample containing “UC Berkeley” and retrain.
   * Save a side‑by‑side confusion matrix (`poisoning_confusion.png`) illustrating pre‑ and post‑poisoning performance.

---

## Sample Outputs

* **MNIST GAN**:

  * `mnist_gan_outputs/epoch_0.png`: Initial noise outputs
  * `mnist_gan_outputs/epoch_50.png`: Mid‑training digit approximations
  * `mnist_gan_outputs/epoch_100.png`: Near‑converged digit samples
  * `mnist_gan_outputs/loss_curve.png`: Adversarial loss trends

* **Data Poisoning**:

  * `poisoning_confusion.png`: Clean vs Poisoned confusion matrices

