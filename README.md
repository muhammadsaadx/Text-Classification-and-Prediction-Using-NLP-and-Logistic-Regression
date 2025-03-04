# NLP Text Classification and Prediction using Logistic Regression

This repository contains an implementation of text classification and prediction using Logistic Regression. The project applies Natural Language Processing (NLP) techniques to process text data and train a model for classification tasks.

## Features
- Implements text classification using Logistic Regression.
- Performs text prediction using n-gram models.
- Utilizes `scikit-learn` and `NLTK` for text processing and model training.
- Processes text using tokenization, vectorization, and feature extraction.
- Evaluates model performance using accuracy, precision, and recall metrics.

## Dataset
The project works with a labeled text dataset containing:
- Text samples for training and testing.
- Preprocessing steps include tokenization, stopword removal, and vectorization.

## Model Architectures
### Text Classification Model
- **TF-IDF Vectorization**: Converts text data into numerical feature representations.
- **Logistic Regression Classifier**: A simple yet effective model for binary or multi-class classification.
- **Probability Estimation**: Uses the sigmoid function to predict class probabilities.

### Text Prediction Model
- **N-gram Language Model**: Uses previous words to predict the next word.
- **Feature Extraction**: Computes frequency-based features for prediction.
- **Logistic Regression**: Predicts the most probable next word in a sequence.

## Training Process
1. **Data Preprocessing**:
   - Tokenization and text cleaning.
   - Vectorization using TF-IDF or CountVectorizer.
2. **Model Training**:
   - Logistic Regression trained with cross-entropy loss.
   - Optimization using Stochastic Gradient Descent (SGD) or other solvers.
3. **Validation & Evaluation**:
   - Accuracy, precision, recall, and F1-score for classification tasks.
   - Sample text predictions and evaluation of model accuracy.

## Evaluation
- **Accuracy**: Measures how well the classifier performs.
- **Precision & Recall**: Evaluates classification performance in imbalanced datasets.
- **Confusion Matrix**: Visualizes model performance on test data.

## Usage
- The text classification model predicts the category of input text.
- The text prediction model generates the next word in a given sequence using n-grams and Logistic Regression.
