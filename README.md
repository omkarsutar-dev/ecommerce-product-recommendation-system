# Ecommerce Product Recommendation System

## Overview :
This project builds a hybrid product recommendation system using
Amazon Electronics reviews data.

## Dataset :
Source: Stanford SNAP Amazon Reviews (Electronics)

### Day 1 :
- Loaded raw JSON.gz review data
- Converted into user–item interaction table
- Performed sanity checks

### Day 2 : 
- Cleaned interaction data
- Removed cold users/items
- Converted ratings to implicit feedback
- Performed time-based train/test split

### Day 3:
- Implemented popularity-based recommender
- Built user-based collaborative filtering
- Evaluated models using Recall@K

### Day 4:
- Implemented item-based collaborative filtering
- Trained matrix factorization model using SVD
- Compared all models using Recall@K