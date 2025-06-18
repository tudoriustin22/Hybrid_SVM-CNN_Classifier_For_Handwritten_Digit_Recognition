# Hybrid CNN-SVM Classifier for Handwritten Digit Recognition

**Author**
**Iustin-Andrei Moisa-Tudor**

## Overview
This project implements a hybrid approach combining Convolutional Neural Networks (CNN) for feature extraction with Support Vector Machines (SVM) for classification to achieve superior handwritten digit recognition performance on the MNIST dataset.

* **Main Model**: CNN-SVM Hybrid Architecture  
* **Baseline**: Standard CNN with softmax classifier
* **Dataset**: MNIST (60,000 training + 10,000 test images)

## 🧠 Key Innovation
The hybrid model addresses limitations of traditional approaches:
* **CNNs** excel at feature extraction but struggle with complex pattern separation using softmax
* **SVMs** perform well in high-dimensional spaces but can't handle raw image data directly
* **Solution**: Use CNN for spatial feature extraction + SVM with RBF kernel for robust classification

## Key Steps
* Preprocessed MNIST data: normalized to [0,1] range, 28×28 grayscale images
* CNN training: 20 epochs, batch size 32, Adam optimizer, dropout regularization  
* Feature extraction: Dense layer outputs from trained CNN
* SVM training: RBF kernel with hyperparameter tuning via GridSearchCV
* Enhanced with data augmentation and regularization techniques

## Best Performance
* **Model**: Hybrid CNN-SVM
* **Accuracy**: 99.45%
* **Precision**: 99.45%  
* **Recall**: 99.45%
* **F1-Score**: 99.45%

## 🏗️ Model Architecture
* **CNN Feature Extractor**: 
  - 2 Convolutional layers with ReLU activation
  - Max-pooling layers for spatial dimension reduction
  - Dropout layers for regularization
  - Dense layer for feature vector output
* **SVM Classifier**:
  - RBF kernel for non-linear decision boundaries
  - Regularization strength (C) = 1.0
  - Kernel coefficient (γ) = 0.1

## 📁 Project Structure
* `hybrid_cnn_svm_model.py` – Main hybrid model implementation
* `cnn_feature_extractor.py` – CNN architecture for feature extraction  
* `svm_classifier.py` – SVM training and classification
* `data_preprocessing.py` – MNIST data loading and preprocessing
* `evaluation_metrics.py` – Performance evaluation and visualization
* `Hybrid CNN-SVM Classifier for Handwritten Digit Recognition on the MNIST Dataset.pdf` – Full project report

## ⚙️ Technologies
* Python (NumPy, Pandas, Matplotlib, Seaborn)
* TensorFlow/Keras (CNN implementation and training)
* scikit-learn (SVM implementation and GridSearchCV)
* NVIDIA AI Workbench (development environment)

## 🖥️ Hardware Used
* **GPU**: NVIDIA RTX 6000 Ada (48GB VRAM)
* **RAM**: 92GB system memory
* **CPU**: AMD EPYC 4564P (16-core, 32-thread)

## 📌 Notes
* Final results demonstrate that the **hybrid CNN-SVM approach** significantly outperforms standalone CNN models
* The **99.45% accuracy** surpasses the original research baseline of 72-90%
* **RBF kernel** proved optimal for handling the high-dimensional CNN feature space
* Further improvement ideas: explore different CNN architectures, experiment with other SVM kernels, and test on additional datasets beyond MNIST
