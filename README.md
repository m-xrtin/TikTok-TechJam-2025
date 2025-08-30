# TikTok-TechJam-2025

Topic: Filtering the Noise: ML for Trustworty Location Reviews

## Introduction and Premise 😎

Online reviews often shape public perception of local businesses and locations, but their reliability is often compromised by spam, irrelevant content, and misleading rants. This project tackles the challenge of “Filtering the Noise” by building an ML-based system that evaluates the quality and relevancy of Google location reviews.

Our approach combines rule-based preprocessing (to remove obvious junk) with an LLM-powered classifier that detects policy violations (advertisements, irrelevant content, rants from non-visitors) and assigns a relevancy score. By filtering out low-quality reviews, the system helps users make better decisions, ensures fair representation for businesses, and reduces moderation workload for platforms.

The end result is a scalable pipeline that can process raw review data, output clean and trustworthy reviews, and provide evaluation metrics and a live demo.

## Requirements

Open AI API key (We will provide a temporary one for judging purposes)
Python version 3.13.7
Required python libraries (run pip install {library_name} in python terminal):

- numpy
- pandas
- scikit-learn
- catboost
- joblib
- matplotlib
- seaborn
- nltk
- tqdm
- json

## Pipeline

### 1. Taking in a data set

Our Program takes in a file (Limited to Json, csv, and txt for simplicity sake), calls the GPT API, and standardizes a JSON file with following categories:
- user_id: unique identifier for the reviewer (anonymous ID or hash; helps link multiple reviews from the same person).

- user_name: Display name of the reviewer as shown on the platform.

- business_name: Name of the restaurant or business being reviewed.

- time: Timestamp of when the review was posted, limited by the dataset.

- text: The raw content of the review (what the user actually wrote).

- rating: Numerical star rating provided by the user (1-5).

- sentiment_category: For the first part of our project, this is solely determined by the numerical rating (1-2 : negative, 3 : neutral, 4-5 : positive)

- rating_category: Review type, such as taste, menu, indoor/out-door environment, we use AI to infer when dataset doesn't provide.

- gmap_id: Unique Google Maps identifier for the restaurant/business location.

### 2. Proper Pre-processing

The dataset downloaded will contain many pieces of data that is unanalysable and would cause the model to behave in unpredicatable ways.
Drop NaN values (or replace with -999 to avoid causing noise)

Removing files with excessive null blocks of data

### 3. Sentiment Analysis

Use VADER to perform a simple sentiment analysis on whether the review text seems genuine, or it is a fake/ranting review
Only analyses based on raw text.

Marks every comment with a label as "0", meaning it is likely to be a trustworthy review, or "1", where it is an untrustworthy review.

### 4. Stacking Assembly

Takes in cleaned and labeled csv of all reviews and utilise CatBoost to perform Machine Learning.

Uses K-fold (K = 10 by default) to find the optimal 90% subproportion of training data to train the model.

Use AverageGain score and F1 score to compute the optimal trained model.

Download the model and features locally

Reuses that model and features file to perform Machine Learning on a new set of reviews, within the production phase.

Outputs a CSV file labelling each review as trustworthy or untrustworthy taking into account of all metadata.

### Authors
Kieran Tran ktran09271@gmail.com
Martin Ma hanyangma0195224@gmail.com
Xander Minzenmay xanderminzenmay@gmail.com
Eric Yu yueric3750@gmail.com

