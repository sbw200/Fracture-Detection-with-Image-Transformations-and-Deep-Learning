# **AI Fracture Detection with Enhanced Image Processing and Deep Learning**

This repository contains code for detecting and classifying spinal fractures using deep learning models, including EfficientNet, ResNet, and DenseNet. The project also explores the impact of **image preprocessing techniques**, such as **Contrast Limited Adaptive Histogram Equalization (CLAHE)**, to improve model performance.

## **Project Overview**

This project aims to accurately classify **X-ray images** as either **fractured** or **non-fractured** by leveraging **deep learning models** and **enhanced image preprocessing techniques**. The key steps include:

1. **Dataset Preparation:**  
   - Loading a dataset of spinal X-ray images.  
   - Splitting the dataset into training and testing sets.  

2. **Baseline Model Training:**  
   - Fine-tuning **pretrained deep learning models** (EfficientNet, ResNet, DenseNet) on raw images.  
   - Establishing baseline performance metrics.  

3. **Image Transformations:**  
   - Applying **image enhancement techniques** such as **Histogram Equalization, CLAHE, and Unsharp Masking** to augment the dataset.  

4. **Model Retraining:**  
   - Training models on the **transformed dataset** to evaluate the impact of preprocessing techniques.  

5. **Evaluation and Results:**  
   - Comparing **Accuracy, F1-score, Precision, and Recall** across different models and transformations.  
   - Visualizing performance differences through graphs and tables.  

---

## **Key Features**

✅ **Advanced Image Preprocessing**  
- CLAHE enhances image contrast for improved feature extraction.  
- Additional transformations improve model robustness.  

✅ **Deep Learning Models**  
- **EfficientNet, ResNet, and DenseNet** are fine-tuned for fracture detection.  

✅ **Hugging Face Model Hosting**  
- The trained models are **saved and uploaded to Hugging Face Hub** for easy access.  

✅ **User-Friendly Inference**  
- A **Gradio-based web app** allows users to **upload an X-ray image** and receive a fracture diagnosis in real time.  

---

## **How It Works**

1. **Upload an X-ray Image**  
   - Users can upload an X-ray image through a simple web interface.  

2. **Image Preprocessing with CLAHE**  
   - CLAHE (Contrast Limited Adaptive Histogram Equalization) is applied to enhance the image contrast.  

3. **Fracture Classification**  
   - The **EfficientNet model** classifies the image as **Fractured** or **Not Fractured**.  

4. **Prediction Output**  
   - The model’s prediction is displayed on the interface.  

---

## **Installation & Usage**

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/sbw200/Fracture-Detection-with-Image-Transformations-and-Deep-Learning
cd Fracture-Detection-with-Image-Transformations-and-Deep-Learning
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```
The required dependencies include:
- Python 3.x  
- PyTorch  
- torchvision  
- OpenCV  
- scikit-learn  
- tqdm  
- torchmetrics  
- matplotlib  
- Gradio  

### **3️⃣ Run the Gradio Web App**
```python
python app.py
```
This will start a local server where you can upload images and receive fracture classification results.

### **4️⃣ Access the Hugging Face Model**
The trained **EfficientNet model** is hosted on **Hugging Face Hub** and can be downloaded as follows:
```python
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(repo_id="potguy/efficientnet_clahe_fracture_classification", filename="efficientnet_clahe_hf.pth")
```

---

## **Evaluation Metrics**

| Metric       | Description |
|-------------|------------|
| **Accuracy** | Measures overall correctness of model predictions. |
| **F1-score** | Balances Precision & Recall to handle class imbalance. |
| **Precision** | Measures how many positive predictions were actually correct. |
| **Recall** | Measures how many actual fractures were correctly detected. |

---

## **Hugging Face Integration**

✅ **Model Hosted on Hugging Face**  
The trained model is available on **Hugging Face Hub** for easy sharing and downloading.  

✅ **Using the Model for Inference**  
To use the model for classification:
```python
from torchvision import models
import torch

model = models.efficientnet_b0(weights=None)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load("efficientnet_clahe_hf.pth", map_location="cpu"))
model.eval()
```

---

## **Contributing**

Contributions are welcome! If you’d like to contribute, please:
1. Fork the repository.  
2. Create a new branch.  
3. Submit a pull request with your changes.  

---

## **Acknowledgments**

🔹 **FracAtlas dataset** for providing X-ray images.  
🔹 **PyTorch & torchvision** for model training.  
🔹 **Google Colab** for computational resources.  
🔹 **Hugging Face Hub** for model hosting.  

---

## **License**

📜 This project is licensed under the **MIT License**.  

---

### **🚀 Ready to Detect Fractures? Try It Now!**
Run the web app locally or check out the hosted model on Hugging Face! 🦴🔍


