# Autoencoder for Next Digit Creation on MNIST

This repository contains the implementation and evaluation of a PyTorch-based **Autoencoder** designed to solve the "next digit creation" problem. Instead of simply reconstructing the input, the neural network learns a mapping to generate the image of the consecutive digit.

This project explores generative deep learning techniques, latent space representations, and introduces an independent classifier network to act as a "judge" for the generated images, offering a comparison against a linear PCA baseline.

## 🎯 Project Purpose

The primary objective is to build a neural network that takes an image of a digit $n \in \{0,1,...,9\}$ as input and generates the image of the next digit $(n+1) \pmod{10}$. Specifically, the project aims to:

* Design and train an Autoencoder with varying bottleneck (latent space) dimensions.
* Prevent overfitting using techniques like Dropout, Early Stopping, and Learning Rate Scheduling.
* Evaluate the generative performance not just via traditional Mean Squared Error (MSE), but through **Classification Accuracy** using an independent model.
* Compare the non-linear deep learning approach against a linear dimensionality reduction technique (**PCA**).

## 📊 Dataset & Preprocessing

The project uses the classic **MNIST** dataset of handwritten digits (70,000 total images). To frame this as a next-digit generation task, a custom preprocessing pipeline was developed:

1. **Custom Pairing:** For every image of digit $n$, the pipeline randomly pairs it with an image of digit $n+1$ (with 9 looping back to 0) from the dataset.
2. **Flattening:** $28 \times 28$ grayscale images are flattened into 784-dimensional vectors.
3. **Normalization:** Pixel values are scaled from $[0, 255]$ to the $[-1, 1]$ range.
4. **Data Splitting:** The 70k pairs are shuffled and split into Training (60%), Validation (20%), and Testing (20%) sets.

## 🧠 Models & Architecture

### 1. The Autoencoder
* **Encoder:** 784 (Input) $\rightarrow$ 256 $\rightarrow$ `hidden_dim` (Latent Space).
* **Decoder:** `hidden_dim` $\rightarrow$ 256 $\rightarrow$ 784 (Output).
* **Configurations:** The latent dimension (`hidden_dim`) is tested across 64, 128, and 256 neurons to find the optimal compression-to-generation ratio.
* **Activations:** ReLU for hidden layers, Tanh for the final output (to match the $[-1, 1]$ normalization).

### 2. The Digit Classifier (The "Judge")
* An independent feedforward neural network (784 $\rightarrow$ 128 $\rightarrow$ 64 $\rightarrow$ 10 classes) trained on standard MNIST classification.
* **Role:** It evaluates the Autoencoder's output to check if the generated image is actually recognizable as the target digit.

### 3. PCA Baseline
* Two separate PCA models (one acting as the encoder, one as the decoder) are fitted using 128 components to serve as a linear baseline for comparison.

## 🛠️ Technologies & Libraries

* **PyTorch (`torch`, `torch.nn`, `torch.optim`):** For defining the Autoencoder, the Classifier, and the training loops.
* **scikit-learn (`PCA`, `KFold`):** For the baseline model and validation splitting.
* **torchvision:** For downloading and transforming the MNIST dataset.
* **Matplotlib & Seaborn:** For comprehensive visualizations (training curves, reconstruction grids, per-digit performance).

## 🏆 Evaluation & Key Findings

The models are evaluated on test MSE, classification accuracy, and training efficiency.

1. **Performance Evaluation:**
   * The model successfully learns the mapping from $n \rightarrow n+1$.
   * Using the independent Classifier provides a much more intuitive metric (Accuracy %) than MSE, proving that the model generates structurally sound digits.
   
2. **Latent Space Comparison:**
   * Architectures with **64, 128, and 256 hidden neurons** were compared. 
   * A larger latent space generally lowers the MSE but increases training time and the risk of overfitting. Early stopping (patience = 10) successfully mitigated memorization.

3. **Per-Digit Analysis:**
   * The project features detailed breakdowns showing which digit transitions (e.g., $3 \rightarrow 4$ vs. $9 \rightarrow 0$) are the easiest or hardest for the network to generate.

4. **Autoencoder vs. PCA:**
   * The non-linear Autoencoder significantly outperforms the 128-component PCA baseline, proving that standard linear projections cannot easily map the complex transition between distinct digit classes.
