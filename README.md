# Facial Expression Recognition using Convolutional Neural Networks (CNN)

This project focuses on automatically recognizing facial expressions from grayscale images using a Convolutional Neural Network (CNN). The goal is to classify facial images into one of seven basic emotions: **Angry, Disgust, Fear, Happy, Neutral, Sad,** and **Surprise**. This type of facial expression recognition (FER) is crucial for various applications, including human-computer interaction, sentiment analysis, and affective computing.

## Project Overview

Facial expressions are a primary indicator of human emotions and can be used to enhance the user experience in interactive systems. This project leverages the capabilities of deep learning, specifically CNNs, to automatically learn and classify the subtle patterns of facial muscle movements associated with different emotions.

### Main Objective
The main objective of this project is to develop a robust model that can accurately classify facial images into one of the seven emotion categories. By doing so, the model aims to achieve high generalization performance on unseen data, ensuring that it can be applied effectively in real-world scenarios.

### Problem Definition
The task is to classify facial images into one of the following emotion classes:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

## Dataset

- **FER Dataset**: The model is trained on the Facial Expression Recognition (FER) dataset, which contains grayscale images of human faces, each labeled with one of the seven emotion classes.

### Dataset Preparation
1. **Data Loading**: The FER dataset is loaded in CSV format, where each image is represented by pixel values in a single string, accompanied by its corresponding label.
2. **Data Cleaning**: Ensures dataset balance and removes unnecessary classes (e.g., 'Disgust') if needed to improve model performance.
3. **Image Reshaping**: Converts pixel strings into 48x48 grayscale image matrices.
4. **Data Normalization**: Normalizes pixel values to a range between 0 and 1 to enhance model training.

### Data Splitting
- **Train/Test Split**: The data is split into training and test sets (e.g., 80% for training and 20% for testing).
- **Validation Set**: A portion of the training data is optionally split into a validation set to monitor model performance during training.

## Model Architecture

The CNN model is designed with the following components:

1. **Convolutional Layers**: Multiple convolutional layers with ReLU activation and pooling layers to automatically learn spatial hierarchies and extract features from facial images.
2. **Dense Layers**: After feature extraction, data is flattened and passed through fully connected (dense) layers to perform classification.
3. **Output Layer**: A softmax output layer is used to predict probabilities for each emotion class.

## Model Compilation

- **Loss Function**: Categorical cross-entropy is used for multi-class classification.
- **Optimizer**: Adam optimizer is selected for efficient training.
- **Metrics**: Accuracy is tracked as the primary performance metric.

## Model Training

- **Batch Processing**: The model is trained using mini-batches for computational efficiency.
- **Epochs**: The model is trained for several epochs while monitoring training and validation accuracy.
- **Data Augmentation**: Techniques like rotation, zoom, and horizontal flipping are applied to increase the model's generalization capability and reduce overfitting.

## Model Evaluation

- **Test the Model**: The model is evaluated on the test set to compute metrics such as accuracy, precision, recall, and F1-score.
- **Confusion Matrix**: A confusion matrix is generated to visualize the model's performance across different emotion classes.
- **Classification Report**: A detailed classification report shows precision, recall, and F1-score for each emotion class.

## Prediction and Visualization

- **Sample Predictions**: The model makes predictions on some test images, which are then compared to the actual labels.
- **Visualize Results**: Test images are displayed along with their predicted and actual labels to visually assess the model's performance.

## Technologies Used

- **Python**: The primary programming language used for developing the model and preprocessing data.
- **TensorFlow/Keras**: Deep learning framework used to build, train, and evaluate the CNN model.
- **Pandas**: For data manipulation and preprocessing.
- **NumPy**: For numerical operations and handling image data.
- **Matplotlib**: For visualizing training progress and model evaluation results.
- **Seaborn**: For plotting confusion matrices and other visualizations.

## Getting Started

To get started with the project, follow these steps:

1. Clone the repository: `git clone <repository-url>`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run the notebook or script to train and evaluate the model.

## Future Improvements

- Experiment with different CNN architectures to improve accuracy.
- Implement additional data augmentation techniques to enhance generalization.
- Fine-tune the model on larger, more diverse datasets to improve robustness.

