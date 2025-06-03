# MGRAD-UNet: Multi-Gated Reverse Attention Multi-Scale Differential Network for Medical Image Segmentation

*Figure 1: Overview of the MGRAD-UNet architecture.*![MGRAD](https://github.com/user-attachments/assets/b86f175b-9c58-4b19-a587-5ced946482b9)


## 📌 Overview
MGRAD-UNet is an advanced neural network designed for medical image segmentation, addressing limitations in traditional UNet architectures such as pixel-level information loss and redundant feature fusion. By leveraging **multi-scale differential decoders** and a **gated reverse attention mechanism**, MGRAD-UNet enhances segmentation accuracy, particularly for small organs and boundary regions.

---

## 🚀 Key Features
- **Multi-Scale Differential Decoder (MSD)**: Extracts complementary features at pixel and structural levels through differential operations between adjacent layers.
- **Gate-Control Reverse Attention (GRAD)**: Focuses on edge features and key regions using a gated feedback mechanism.
- **Dual-Decoder Structure**: Combines features from two differential decoders to improve robustness and accuracy.
- **Aggregated Loss**: Supervises multi-stage predictions by combining losses from non-empty subsets of feature maps.

---

## 📊 Performance
### Synapse Multi-Organ CT Dataset
| Metric       | MGRAD-UNet | TransUNet | SwinUNet |
|--------------|------------|-----------|----------|
| **Dice (%)** | **83.33**  | 77.61     | 77.58    |
| **HD95**     | **16.67**  | 26.90     | 27.32    |

### ACDC Cardiac MRI Dataset
| Metric       | MGRAD-UNet | TransUNet | SwinUNet |
|--------------|------------|-----------|----------|
| **Dice (%)** | **92.13**  | 89.71     | 88.07    |

---

## 🛠️ Installation
1. **Requirements**:
   - Python 3.8+
   - PyTorch 1.11.0+
   - NVIDIA GPU (Tested on RTX 4090)

2. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/MGRAD-UNet.git
   cd MGRAD-UNet
   pip install -r requirements.txt
