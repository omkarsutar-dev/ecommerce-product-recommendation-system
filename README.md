# Product Recommendation System (Amazon Reviews)

This project implements a **product recommendation system** using the Amazon Reviews dataset. The goal is to explore different collaborative filtering techniques and evaluate how recommendation quality improves as models become more advanced. The system progresses from a simple popularity-based baseline to **Matrix Factorization**, a widely used approach in industry-scale recommendation systems.

## Project Overview

Recommender systems are widely used by platforms like Amazon, Netflix, and YouTube to personalize user experiences. In this project, we build and evaluate several recommendation techniques:

* Popularity-based recommendation (baseline)
* User-based collaborative filtering
* Item-based collaborative filtering
* Matrix Factorization (latent factor model)

The dataset is converted into a **user–item interaction format**, where each row represents a user interacting with a product.

Example interaction data:

| user_id | item_id | rating |
| ------- | ------- | ------ |
| U1      | P101    | 5      |
| U1      | P205    | 4      |
| U2      | P101    | 3      |

From this data we construct a **User–Item Interaction Matrix**, which becomes the foundation for collaborative filtering models.

## Recommendation Pipeline

The pipeline for this system follows these steps:

1. Load and preprocess dataset
2. Create user–item interaction matrix
3. Train recommendation models
4. Evaluate models using Recall@K

```
Dataset
   ↓
User–Item Matrix
   ↓
Baseline Model
   ↓
User-Based Collaborative Filtering
   ↓
Item-Based Collaborative Filtering
   ↓
Matrix Factorization
   ↓
Evaluation (Recall@K)
```

## Models Implemented

### Baseline Model (Popularity-Based)

This model recommends the **most popular products** to all users. Popularity is determined using the number of interactions or reviews.

Advantages:

* Simple and fast
* Useful benchmark

Limitations:

* No personalization
* Same recommendations for every user

### User-Based Collaborative Filtering

This method recommends products based on **similar users**.

Idea:
Users with similar interaction patterns tend to like similar items.

Steps:

* Compute similarity between users (Cosine similarity)
* Find nearest neighbors
* Recommend items liked by similar users

### Item-Based Collaborative Filtering

Instead of comparing users, this approach compares **items**.

Idea:
Items frequently liked together by many users are similar.

Steps:

* Compute item-to-item similarity
* Recommend items similar to those already interacted with by the user

Advantages:

* More stable than user-based approaches
* Scales better when the number of users is very large

### Matrix Factorization

Matrix Factorization is a **latent factor model** widely used in modern recommendation systems.

The user–item matrix is decomposed into two lower-dimensional matrices:

```
R ≈ P × Qᵀ
```

Where:

* **R** = user–item interaction matrix
* **P** = user latent feature matrix
* **Q** = item latent feature matrix

Each user and item is represented by a **latent embedding vector**.
The recommendation score is computed using the dot product:

```
score(user, item) = dot(user_vector, item_vector)
```

This approach captures hidden relationships between users and products.

## Evaluation Metric

We evaluate the models using **Recall@K**.

Recall@K measures how many relevant items appear in the top K recommendations.

```
Recall@K = Relevant Recommended Items / Total Relevant Items
```

Example:

If a user has interacted with 4 relevant items and the system recommends 2 of them in the top 10:

```
Recall@10 = 2 / 4 = 0.5
```

Higher Recall@K indicates better recommendation performance.

## Example Results

| Model                | Recall@10 |
| -------------------- | --------- |
| Baseline             | ~0.006    |
| User-Based CF        | ~0.09     |
| Item-Based CF        | ~0.02     |
| Matrix Factorization | ~0.12     |

Matrix Factorization provides the best performance among the implemented methods.

## Project Structure

```
recommendation-system/
│
├── data/
│   ├── raw
│   └── processed
│
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── baseline_model.ipynb
│   ├── collaborative_filtering.ipynb
│   └── matrix_factorization.ipynb
│
├── src/
│   ├── data_processing.py
│   ├── metrics.py
│   └── models.py
│
├── artifacts/
│
└── README.md
```

## Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Jupyter Notebook

## Future Improvements

Next phases of this project will extend the system to a **modern recommendation architecture**, including:

* Two-Tower Neural Recommendation Model
* Approximate Nearest Neighbor retrieval using FAISS
* Candidate Ranking Layer
* Streamlit-based recommendation interface

These upgrades will transform the system into a **production-style scalable recommendation pipeline**.

## Author

Omkar Sutar
Software Developer | Machine Learning Enthusiast | Building intelligent and scalable AI systems
