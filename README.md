# Denoising Autoencoder for CIFAR-100

## **Overview**
This project implements a **convolutional autoencoder** trained on the **CIFAR-100 dataset** to reconstruct noisy images. The model learns to extract meaningful features from corrupted images and generate high-quality reconstructions.

## **Methodology**
### **1. Data Preprocessing**
- Added synthetic noise (Gaussian noise, masked noise, salt-and-pepper) to CIFAR-100 images.
- Normalized pixel values for stable training.

### **2. Model Architecture**
- **Encoder:** Convolutional layers to compress images into a latent representation.
- **Decoder:** Deconvolutional layers to reconstruct the original image.

### **3. Training**
- **Loss Function:** Mean Squared Error (MSE) to minimize reconstruction error.
- **Optimizer:** Adam with learning rate scheduling.

## **Results**
### Noise examples
![Noise examples](https://raw.githubusercontent.com/stevenluchen/denoising-autoencoder/refs/heads/main/results/noise_examples.png)

### Reconstructed images
![Reconstructed images](https://raw.githubusercontent.com/stevenluchen/denoising-autoencoder/refs/heads/main/results/noisy-reconstructions.png)
