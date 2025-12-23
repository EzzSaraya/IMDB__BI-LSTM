# IMDb Movie Review Sentiment Classification (LR + BiLSTM) 🎬🧠

A natural language processing project that classifies movie reviews as **positive** or **negative** using both **traditional machine learning (Logistic Regression)** and **deep learning (BiLSTM)** models trained on the IMDB dataset of 50,000 movie reviews.

Dataset: [IMDB Dataset of 50K Movie Reviews (binary sentiment)](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) — contains 50,000 labeled movie reviews for sentiment analysis tasks. :contentReference[oaicite:1]{index=1}

---

## 📌 Project Overview

This repository demonstrates two approaches for sentiment analysis:

1. **Logistic Regression** with text vectorization (e.g., TF-IDF)  
2. **Bidirectional LSTM (BiLSTM)** neural network for deeper semantic understanding

The workflow includes data preprocessing, tokenization, model training, evaluation, and performance comparison for movie review sentiment classification. :contentReference[oaicite:2]{index=2}

---

## 🧠 Key Features

- **Binary Sentiment Classification:** Positive vs Negative review prediction  
- **Text Preprocessing & Tokenization** (cleaning, padding)  
- **Machine Learning Model:** Logistic Regression on TF-IDF vectors  
- **Deep Learning Model:** BiLSTM capturing long-term dependencies  
- **Model Evaluation:** Accuracy, confusion matrix, and performance plots

---

## 🛠️ Tools & Technologies

- Python
- TensorFlow / Keras
- Scikit-Learn
- NumPy, Pandas
- Jupyter / Kaggle Notebook

---

## 🧩 Workflow Summary

1. **Load the IMDB dataset**
2. **Text preprocessing**
   - Remove HTML tags, punctuation, and stopwords
   - Tokenize and pad sequences for deep learning
3. **Vectorization**
   - TF-IDF for Logistic Regression
4. **Model Training**
   - Logistic Regression classifier
   - BiLSTM network with embedding and recurrent layers
5. **Evaluation**
   - Compare accuracy and performance of both models
   - Visualize results with plots


