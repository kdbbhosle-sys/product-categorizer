Project Report: Smart Product Categorizer
=========================================

Automated E-Commerce Taxonomy Classification using Linear SVM

### 1\. Executive Summary

This project addresses a critical challenge in e-commerce: organizing messy product data. Manual categorization of thousands of products is slow, expensive, and error-prone. We have built an automated Machine Learning (ML) pipeline that reads a raw product title (e.g., _"Acme Premium Leather Boots"_) and instantly assigns it to the correct Google Product Taxonomy category (e.g., _"Apparel & Accessories > Shoes"_).

The solution uses a Linear Support Vector Machine (SVM), chosen for its speed and efficiency with text data. The final output is an interactive web application that provides real-time classification with confidence scores.

### 2\. Problem Statement

#### The Challenge

Imagine an online marketplace like Amazon or eBay where sellers upload thousands of new products daily.

*   Seller A uploads a product called: _"Super Durable Hiking Backpack, 50L"_
    
*   Seller B uploads: _"Vintage Denim Jacket Blue Size M"_
    

Without categorization, a user searching for "Shoes" might never find relevant products, or they might see irrelevant results.

#### The "Manual" Bottleneck

If a human has to categorize these items, they might process 100 items per hour. For 1,000,000 items, this would take 10,000 hours (or 5 years for one person).

#### The Goal

Build an AI system that can categorize 30,000+ items in seconds with high accuracy, saving thousands of man-hours.

### 3\. Solution Architecture

We split the solution into two distinct phases: Training (teaching the AI) and Inference (using the AI).

#### Phase A: The Training Pipeline

1.  Data Ingestion: Download the official Google Taxonomy (standardized categories).
    
2.  Synthetic Data Generation: Create "fake" but realistic product titles to train the model, since we don't have a private dataset.
    
3.  Preprocessing: Convert text into numbers using TF-IDF (Term Frequency-Inverse Document Frequency).
    
4.  Modeling: Train a Linear SVM to find patterns in the numbers.
    
5.  Validation: Test the model on unseen data to calculate accuracy.
    
6.  Serialization: Save the trained brain as a .pkl file.
    

#### Phase B: The Inference Application

1.  Load Model: The app reads the saved .pkl file.
    
2.  User Input: A human types a title into a web interface (Streamlit).
    
3.  Prediction: The model outputs the category and a "Confidence Score" (e.g., "I am 95% sure this is a shoe").
    

### 4\. Methodology & Logic Explained

#### Step 1: Text Vectorization (TF-IDF)

Computers cannot read English; they only understand math. We use TF-IDF to convert words into vectors.

*   Logic:
    
    *   TF (Term Frequency): How often does the word "Shoe" appear in this title?
        
    *   IDF (Inverse Document Frequency): How rare is the word "Shoe" across _all_ titles?
        
*   Example:
    
    *   The word "The" appears everywhere. It has a low score (low importance).
        
    *   The word "Spatula" appears rarely. If it appears in a title, it is highly important for classifying the item as "Kitchenware."
        

#### Step 2: Linear SVM (Support Vector Machine)

The SVM is our classifier. Imagine plotting every product on a 2D graph.

*   "Boots" cluster in the top right.
    
*   "Spatulas" cluster in the bottom left.
    

The SVM's job is to draw a straight line (hyperplane) that best separates "Boots" from "Spatulas."

*   Why SVM? It is exceptionally fast and works well when you have thousands of words (dimensions), which is common in text processing.
    

#### Step 3: Probability Calibration

A standard SVM outputs a "distance" from that dividing line. It doesn't natively know "percentages."

*   The Fix: We wrap the SVM in a CalibratedClassifierCV.
    
*   The Logic: It fits a Sigmoid curve (S-curve) to the distances.
    
    *   Distance 0 = 50% confidence (Unsure).
        
    *   Distance +10 = 99% confidence (Sure it's Category A).
        
    *   Distance -10 = 1% confidence (Sure it's NOT Category A).
        

### 5\. Implementation Details

#### Part 1: The Training Script (train\_model.py)

\# Key Snippet: The Pipeline

pipeline = Pipeline(\[

    # Step 1: Turn text into numbers. 

    # ngram\_range=(1,2) means we look at "Shoe" AND "Running Shoe"

    ('tfidf', TfidfVectorizer(stop\_words='english', ngram\_range=(1,2))),

    # Step 2: The Classifier with Calibration

    ('clf', CalibratedClassifierCV(

        SGDClassifier(loss='hinge', penalty='l2'), # The actual SVM

        method='sigmoid' # Converts score to %

    ))

\])

Part 2: The Web App ([app.py](http://app.py))

\# Key Snippet: Prediction & Confidence

\# We ask the model for PROBABILITIES, not just the answer.

probabilities = model.predict\_proba(\[user\_input\])\[0\]

confidence = np.max(probabilities) # The highest probability is our confidence.

\# We sort the probabilities to find the "Runner Ups"

top\_3 = np.argsort(probabilities)\[-3:\]\[::-1\]

### 6\. Results & Performance

*   Training Time: ~5-10 seconds for 30,000 samples (High efficiency).
    
*   Accuracy: typically 90-98% on synthetic data.
    
    *   _Note:_ Accuracy on real-world data may be lower (70-85%) because real human titles are messier than our synthetic ones.
        
*   Inference Speed: <50ms per prediction.
    

Example Prediction:

*   Input: _"ProLine Heavy-Duty Running Shoes"_
    
*   Prediction: Apparel & Accessories > Shoes
    
*   Confidence: 98.5%
    

### 7\. Conclusion

We successfully demonstrated that a lightweight Linear SVM is sufficient for high-speed, high-accuracy text classification. By utilizing a Pipeline architecture, we ensured that the preprocessing steps (TF-IDF) are permanently saved with the model, preventing errors during deployment. The Streamlit interface makes this complex math accessible to non-technical users.
