Brain Tumor Identification and Classification with Deep Learning
Project Summary
This initiative focuses on building and comparing deep learning-based solutions for identifying and categorizing brain tumors using MRI scans. A variety of neural network architectures—including VGG16, ResNet50, ResNet, and EfficientNet—are implemented and evaluated for classification accuracy. Additionally, a YOLOv5 object detection model, trained using a brain tumor dataset available on Roboflow, is incorporated to locate tumors within the MRI images.

To enhance clinical relevance, the project leverages Google's Gemini API to produce automated diagnostic summaries based on model predictions. These summaries are compiled into structured PDF medical reports using the ReportLab library, presenting findings in an accessible and professional format suitable for medical practitioners.

Core Modules
Image Preparation: Includes loading, resizing, normalization, and augmentation of MRI images to prepare data for training.

Deep Learning Models:

VGG16: A pre-trained convolutional neural network adapted for the dataset.

ResNet & ResNet50: Deep residual networks utilized for improved feature extraction.

EfficientNet: Chosen for its balance between accuracy and computational efficiency.

YOLOv5: Object detection framework used to identify tumor regions.

Model Training: Each model is compiled, trained, and validated using standard metrics to track performance.

Performance Assessment: Models are analyzed using accuracy, precision, recall, F1-score, confusion matrices, and classification reports.

Metrics Visualization: Graphs of training and validation statistics help track model behavior and generalization.

Report Automation with Gemini API: Generates AI-powered medical text reports based on predictions.

PDF Report Generation: Uses ReportLab to compile detailed medical summaries, visualizations, and tumor classifications into downloadable PDF files.

Model Performance Snapshot
Custom CNN: Accuracy – 92.5%

VGG16: Accuracy – 92.12%

ResNet50: Accuracy – 90.81%

EfficientNet: Accuracy – 75.24%

ResNet: Accuracy – 74.28%

Technical Stack
Python 3.x

TensorFlow & Keras

NumPy & Pandas

OpenCV

Matplotlib

Google Cloud SDK

ReportLab

Steps to Reproduce
Clone the repository to your local environment.

Install dependencies via pip install -r requirements.txt.

Set up credentials and configuration for Google Cloud access.

Launch the Jupyter Notebooks or run on Google Colab.

Execute the workflow step-by-step to train models and produce reports.

Final Thoughts
This project showcases the potential of AI-driven methods in the field of medical imaging, particularly for assisting in brain tumor diagnosis. The combination of classification, object detection, and automated reporting offers a solid baseline for future enhancements in clinical AI applications.