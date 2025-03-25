# VR_Project1_Pranav_Ashirwad_Abhik_IMT2022053_108_117

# Face Mask Detection, Classification, and Segmentation

## 👇 Introduction

*The goal of this project is to develop a comprehensive computer vision solution that can accurately detect, classify, and segment face masks in images. The tasks include:*

- Binary classification using handcrafted features and traditional machine learning models to distinguish between masked and unmasked faces.
- Binary classification using Convolutional Neural Networks (CNNs) to leverage deep learning for improved accuracy.
- Traditional image segmentation methods to isolate mask regions for masked faces.
- Deep learning-based segmentation using the U-Net architecture for precise mask localization.

---

## 📂 Dataset

### 1. Face Mask Classification Dataset

**Source**: [Face Mask Detection Dataset by Chandrika Deb](https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset)
**Description**: 2 folders - one containing images of faces with masks and the other without masks. The image files have no naming convention and contain special characters which are dealt with in the implementation. They are also renamed later for ease of use.

### 2. Masked Face Segmentation Dataset

**Source**: [MFSD Dataset by Sajad Razavi](https://github.com/sadjadrz/MFSD)**Description**:

> *Preprocessing steps used before training*

- Renamed incorrectly labeled image files containing special characters (e.g., ≈, ˙, ◊, ¢) by replacing them with underscores _ using a custom Python script.
- This ensured consistency and avoided potential file read errors during training and evaluation.
- All changes were made within the dataset directory before preprocessing or model training steps.

---

## ⚙️ Methodology

### A. Binary Classification using Handcrafted Features + ML Classifiers

Three types of handcrafted features were extracted from the input images:

- HOG (Histogram of Oriented Gradients): Captures edge and shape information from grayscale images.
- LBP (Local Binary Patterns): Encodes texture by thresholding pixel neighborhoods in grayscale images.
- Color Histograms (in HSV space): Encodes color distribution and relationships in the image.

The extracted features were used to train three machine learning classifiers:

- Support Vector Machine (SVM)
- Random Forest
- XGBoost

Hyperparameters used:

- Random Forest: n_estimators=100, random_state=42
- SVM: kernel='linear', C=1.0, random_state=42
- XGBoost: n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42

These models were trained and evaluated using a train-test split. Accuracy was used as the primary evaluation metric.
---------------------------------------------------------------------------------------------------------------------

---

### B. Binary Classification using CNN

- The CNN was trained on resized (128x128) face mask images using a torchvision.ImageFolder setup with a custom transform pipeline.
- The dataset was split into training (70%), validation (15%), and test (15%) subsets with reproducibility ensured using a fixed random seed.
- The CNN architecture consisted of 3 convolutional layers with batch normalization and max pooling, followed by two fully connected layers.
- Activation functions (relu, tanh) were tested, with relu yielding better performance overall.

Experiments involved looping through multiple hyperparameter combinations:

- Learning Rates: 0.01, 0.001, 0.0001
- Batch Sizes: 16, 32
- Optimizers: Adam, RMSprop
- Activation Functions: relu, tanh

The final classification layer used a sigmoid activation for binary output and BCELoss as the loss function. The best model configuration (based on F1-score) was selected after testing on the holdout test set.
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### C. Region Segmentation using Traditional Techniques\

- Multiple traditional segmentation methods were explored, including **Canny edge detection**, **Otsu' s thresholding**, and  **region growing**.
- Ultimately, **Otsu’s thresholding** was selected as the preferred method due to its fully **algorithmic nature**, requiring no manual parameter tuning.
- Other methods, while sometimes visually effective, required extensive trial-and-error and manual adjustment of hyperparameters like kernel sizes, thresholds, and iterations — making them less robust and scalable.
- Otsu’s method automatically determines the optimal threshold to separate foreground (mask) from background based on image histogram, making it a practical choice for batch segmentation.
- The results are mentioned further.

---

### D. Mask Segmentation using U-Net

- A **U-Net** model was implemented using PyTorch. It consisted of an encoder-decoder structure with skip connections for precise segmentation.
- The model was trained using the MFSD dataset with 3-channel input images and 1-channel binary masks as targets.
- **Binary Cross Entropy (BCE) loss** was used with a sigmoid activation in the output layer.
- Input images were resized to 256 × 256 pixels,  and batched with a size of 16 during training. Model performance was evaluated using IoU (Intersection over Union) and Dice Coefficient metrics.
- The results along with learning rate, batch size and optimizer are mentioned below.

## 🔬 Hyperparameters & Experiments

| Model | Learning Rate        | Batch Size | Optimizer      | Accuracy / IoU / Dice |
| ----- | -------------------- | ---------- | -------------- | --------------------- |
| CNN   | 0.01 / 0.001 /0.0001 | 16 / 32    | Adam / RMSprop |                       |
|       |                      |            |                |                       |
| U-Net | 0.001                | 1          | Adam           | -/0.634627/0.776041   |

---

## 📊 Results Summary

| Task                        | Method        | Accuracy / Score |
| --------------------------- | ------------- | ---------------- |
| Binary Classification (ML)  | SVM           | `92.50%`       |
| Binary Classification (ML)  | Random Forest | `92.28%`       |
| Binary Classification (ML)  | XGBoost       | `92.36%`       |
| Binary Classification (CNN) | CNN           | `97.65%`       |

| Task                       | Method              | Mean IoU/ Mean Dice  |
| -------------------------- | ------------------- | -------------------- |
| Segmentation (Traditional) | Otsu's Thresholding | 0.258274 / 0.360258  |
| Segmentation (U-Net)       | U-Net               | 0.634627 / 0.776041 |

---

## 🧠 Observations & Analysis

*Classification Task:*

- All three classifiers (SVM, Random Forest, and XGBoost) achieved very close accuracy values, with SVM slightly outperforming the others.
- Despite being a simpler model, SVM marginally outperformed the more complex ensemble methods, indicating that the feature space was likely well-suited for linear separation.
- The consistency across models suggests that the handcrafted features captured essential discriminatory information for the mask classification task

### Binary Classification using CNN:

| Learning Rate | Batch Size | Optimizer | Activation | Test Accuracy | Test F1-Score |
| ------------- | ---------- | --------- | ---------- | ------------- | ------------- |
| 0.0001        | 32         | Adam      | relu       | 0.9772        | 0.9765        |
| 0.0001        | 16         | Adam      | relu       | 0.9756        | 0.9749        |
| 0.0001        | 16         | RMSprop   | relu       | 0.9691        | 0.9679        |
| 0.0001        | 16         | RMSprop   | tanh       | 0.9593        | 0.9576        |
| 0.0001        | 16         | Adam      | tanh       | 0.9577        | 0.9555        |

- The best CNN model was achieved using `Adam` optimizer, `relu` activation, learning rate `0.0001`, and batch size `32`, with a Test Accuracy of **97.72%** and F1-Score of **97.65%**.
- Models with `relu` consistently outperformed `tanh`, particularly when paired with `Adam` optimizer.
- RMSprop performed slightly worse than Adam across all trials.
- Increasing batch size to 32 helped the best-performing model generalize better.

*Segmentation Task:*
- In traditional methods, the case is that for each image you need to adjust the parameters by trail and error to get the best direction. Thus in such cases we saw that traditional methods that have an algorithmic approach such as Otsu' thresholding as they are better to use rather than those that require individual adjustments such as canny edge detector etc.
- U-Net produces significantly smoother and more precise mask boundaries compared to Otsu’s thresholding.
---

---
### Challenges Faced
A. Binary Classification using Handcrafted Features + ML Classifiers
Feature quality depends on image conditions
The handcrafted features (like HOG, LBP) did not work well when images had different lighting or face angles.

Time-consuming feature extraction
Extracting features for all images took a lot of time and added extra steps before training.

B. Binary Classification using CNN
Limited access to GPU
Training CNNs locally was slow due to lack of a powerful GPU. We used Kaggle's free GPU environment for faster training.

Slow training and tuning
Testing different learning rates, batch sizes, and optimizers took time because each run had to be trained separately.

Risk of overfitting
Since the dataset was small, the model sometimes performed very well on training data but not as well on test data.

Early bias in training
In early training epochs, the model favored one class more than the other, which affected the balance of predictions.

C. While implementing the UNet Segmentation model, the computation was very slow on CPU. Hence we shifted to Kaggle and used two T4 in parallel using pytorch's parallel computation capabilities and 
---

## ▶️ How to Run the Code

1. Clone this repository
2. Please ensure that you download the dataset mentioned in the above sections and ensure that it is in the root of the directory.
3. Set up environment:
   Ensure that you have python installed.
   ```
   pip install torch torchvision torchinfo opencv-python pillow matplotlib scikit-learn pandas numpy
   ```
5. Run binary classification notebook (handcrafted + CNN):
6. Run segmentation notebook (traditional + U-Net):

