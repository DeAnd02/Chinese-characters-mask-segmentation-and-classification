# Pixel-wise Bounding Box Detection and Character Classification

This project implements a Deep Learning pipeline for the semantic segmentation (pixel-wise bounding box detection) of Chinese characters in complex images, followed by a Transfer Learning activity for binary classification.

The project was developed in Python using the **PyTorch** framework and was executed within the **mlt-gpu2 server** environment.

## 🎯 Project Goals

1.  **Segmentation (Main Task):** Implement two distinct Encoder-Decoder architectures to produce a "soft binary mask," indicating the probability that a pixel belongs to a bounding box containing a Chinese character.
2.  **Imbalance Mitigation:** Address the severe class imbalance (characters vs. background) by primarily using **Dice Loss** as the cost function.
3.  **Transfer Learning (Bonus A):** Reuse the features learned from the segmentation model to train a binary classifier capable of detecting the presence of the most frequent character in an image.

---

## 💻 Repository Structure

The repository contains the following main files and directories:

| File/Directory | Description |
| :--- | :--- |
| **`Report.pdf`** | The academic report detailing the architectures, methodological choices, hyperparameters, and quantitative/qualitative results. |
| **`UNet.ipynb`** | Notebook for defining and training the **CustomUNet** (U-Net-inspired) architecture. This includes the setup for **Bonus A: FeatureClassifier**. |
| **`SegNet.ipynb`** | Notebook for defining and training the **SimpleSegNet** (SegNet-inspired) architecture. |
| **`Visualize_results.ipynb`** | Notebook used to load the trained weights and generate qualitative visualizations of the predicted masks (e.g., overlays). |
| **`created_models.py`** | File containing the network class definitions: `ConvBlock`, `SimpleSegNet`, `CustomUNet`, and `FeatureClassifier`. |
| **`utils.py`** | Contains utility functions. |

---

## 🚀 Execution Instructions

**ATTENTION:** The code is configured to access the input dataset directly from the specified network path on the university ssh.

### Bonus A: Classification Results

* **Task:** Detect the presence of the most frequent character in an image using features from the CustomUNet bottleneck.
* **Evaluation Metric:** F1 Score (used due to label imbalance).
* **Best Validation F1 Score:** $\mathbf{0.4255}$
* **Conclusion:** The features extracted by the U-Net encoder for pixel-wise segmentation proved effective and transferable to the image-level classification task.
