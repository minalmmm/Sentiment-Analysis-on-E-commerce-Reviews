# Sentiment Analysis of Product Reviews

## Objective

The goal of this project is to analyze customer sentiment from product reviews using machine learning algorithms. The primary objective is to predict the sentiment (Positive, Neutral, Negative) from the reviews.

## Tools Used

- **Programming Language:** Python
- **Libraries:** Scikit-learn, TensorFlow, TextBlob, Keras

## Overview

### Data Visualization

- The distribution of review lengths is heavily skewed towards shorter reviews, with most having fewer than 250 characters.
- The majority of the dataset consists of "Data Reviews," while "Data Test Reviews" form a smaller subset.
- Few reviews exceed 500 characters, and almost none are longer than 1000 characters.
- The plot indicates a need for handling class imbalance in review length during text preprocessing.

### Positive and Negative Words Wordcloud

The project includes word clouds for both positive and negative words to visualize the most frequent terms associated with each sentiment.

## Data Collection & Preprocessing

### Data Collection

- **Source:** Product reviews from an online marketplace.
- **Columns:** Review Text, Review Title, Sentiment Labels (Positive, Neutral, Negative)

### Preprocessing Steps

1. **Text Cleaning:** Removal of stop words, punctuation, and unnecessary characters.
2. **Tokenization and Lemmatization:** Breaking down text into tokens and reducing them to their base forms.
3. **Sentiment Label Encoding and Vectorization.**

### Text Vectorization Techniques

- **TF-IDF Vectorizer:** Converts text data into a matrix of TF-IDF features, weighing words according to their frequency and relevance.
- **Count Vectorizer:** Converts text data into a matrix of token counts.

## Handling Class Imbalance

The class distribution of the sentiment dataset is as follows:

- **Positive:** 3749 instances
- **Neutral:** 158 instances
- **Negative:** 93 instances

This shows a significant class imbalance, where the "Positive" sentiment dominates the dataset.

### Techniques to Balance Data

1. **UnderSampling**
2. **OverSampling**

Logistic Regression models are evaluated on both under-sampled and over-sampled data, with over-sampled data performing better.

## Model Training & Evaluation

### Multinomial Naive Bayes

- The ROC curve shows moderate performance, with Class 0 outperforming others (AUC = 0.74).
- The classifier demonstrates limited ability to distinguish between certain classes, especially Class 1 (AUC = 0.67).

### XGBoost Classifier

- XGBoost performs better in predicting all the classes, with a more balanced performance across each class.
- Strong class separation and higher AUC, especially for underrepresented classes.

### Multiclass SVM Classifier

- The ROC curve for the Multiclass SVM classifier indicates moderate performance, with AUC values ranging from 0.65 to 0.69.
- The model struggles with distinguishing certain classes, reflecting a need for better optimization or data balance.

### Accuracy Comparison

- **XGBoost:** Achieved the highest validation accuracy (95.5%).
- **SVM:** Performed well at 93%.
- **Multinomial Naive Bayes (MNB):** Struggled with imbalanced data but is computationally efficient.

## Deep Learning Models

### Artificial Neural Networks (ANN)

- Multi-layer perceptron for non-linear relationships.
- Key Layers: Dense, Dropout.

### Long Short-Term Memory (LSTM)

- Sequential model for text classification.
- Key Layers: LSTM, Embedding, Spatial Dropout.

## Key Findings

- **XGBoost and Random Forest** showed superior performance (95.5%).
- **SVM** is a strong choice but slightly behind in accuracy (93%).
- **MNB** struggled with imbalanced data but is computationally efficient.

## Conclusion & Future Work

### Conclusion

- **XGBoost** excels in both accuracy and class balance.
- Deep learning models can be further fine-tuned for complex patterns in the data.

### Future Work

- Explore **Transformer models** like **BERT** for improved text understanding.
- Deploy models to production for real-world applications.
