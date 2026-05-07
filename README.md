🎬 Hybrid Recommendation System

A scalable and explainable Hybrid Movie Recommendation System built using Collaborative Filtering (SVD) and Content-Based Filtering (TF-IDF). The project handles the cold-start problem, supports new-user onboarding, and provides an interactive Streamlit web application for personalized movie recommendations.

🚀 Features
Hybrid Recommendation Engine
Collaborative Filtering using Truncated SVD
Content-Based Filtering using TF-IDF
Cold-Start User Handling
Genre-Based Onboarding
Personalized Recommendations
Explainable Recommendations
Sparse Matrix Optimization for Large Datasets
Evaluation using Precision@K and Recall@K
Interactive Streamlit Web App
User Analytics Dashboard

🧠 Project Architecture
+----------------------+
|   Streamlit Frontend |
+----------+-----------+
           |
           v
+----------------------+
|  User Onboarding     |
| Genre / Ratings Flow |
+----------+-----------+
           |
           v
+----------------------+
| Hybrid Recommendation|
|  - Collaborative CF  |
|  - Content-Based CB  |
+----------+-----------+
           |
           v
+----------------------+
| Evaluation Metrics   |
| Precision@K Recall   |
+----------+-----------+
           |
           v
+----------------------+
| Personalized Output  |
+----------------------+

📂 Project Structure
Recommendation_system/
│
├── data/
│   ├── movies.csv
│   ├── ratings.csv
│
├── recommender/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── collaborative.py
│   ├── content_based.py
│   ├── hybrid.py
│   ├── evaluation.py
│
├── app.py
├── main.py
├── requirements.txt
└── README.md

📊 Dataset
🎥 MovieLens Dataset

This project uses the MovieLens Dataset containing movie ratings and metadata.

Dataset Files
File	Description
movies.csv	Movie titles and genres
ratings.csv	User ratings for movies
📥 Dataset Download Steps
Step 1: Download Dataset

Download MovieLens dataset from:

https://grouplens.org/datasets/movielens/

Recommended version:

MovieLens Latest Small
or MovieLens 1M
Step 2: Extract Dataset

After downloading:

Extract ZIP file
Locate:
movies.csv
ratings.csv
Step 3: Create Data Folder

Inside project folder create:

data/
Step 4: Move Dataset Files

Place files here:

Recommendation_system/data/
│
├── movies.csv
├── ratings.csv
⚙️ Installation
Step 1: Clone Repository
git clone https://github.com/your-username/hybrid-recommendation-system.git
cd hybrid-recommendation-system
Step 2: Create Virtual Environment (Optional)
Windows
python -m venv venv
venv\Scripts\activate
Linux / Mac
python3 -m venv venv
source venv/bin/activate
Step 3: Install Dependencies
pip install -r requirements.txt
📦 Requirements

Create requirements.txt

streamlit
pandas
numpy
scikit-learn
scipy
▶️ Run the Project
Run Backend
python main.py

Expected Output:

Hybrid recommendations for user 149:
...
Precision@10: 0.84
Recall@10: 0.02
Run Streamlit App
streamlit run app.py

Browser opens automatically.

🧠 How the System Works
1️⃣ Collaborative Filtering

Uses:

Sparse user-item matrix
Truncated SVD

Learns:

User preferences
Latent movie patterns
2️⃣ Content-Based Filtering

Uses:

Movie genres
TF-IDF vectorization
Cosine similarity

Recommends:

Similar-content movies
3️⃣ Hybrid Recommendation

Combines:

Collaborative Filtering
Content-Based Filtering

Benefits:

Better accuracy
Cold-start handling
Improved personalization
❄️ Cold-Start Handling

For new users:

Genre selection onboarding
“Rate 5 movies” flow
Popularity-based fallback

This improves recommendations even without prior history.

📈 Evaluation Metrics
Metric	Purpose
Precision@K	Measures recommendation relevance
Recall@K	Measures recommendation coverage

Example:

Precision@10: 0.84
Recall@10: 0.02
🎨 Streamlit Features
Elegant Gradient UI
Glassmorphism Cards
Explainable Recommendations
User Analytics Dashboard
New User & Existing User Flow
Personalized Recommendations
💡 Example Recommendation Explanation
Recommended because users similar to you liked this
and it matches your preferred genres.
🛠️ Tech Stack
Programming
Python
Machine Learning
Scikit-learn
TruncatedSVD
TF-IDF
Data Processing
Pandas
NumPy
Frontend
Streamlit
Visualization
Matplotlib
Seaborn
🚀 Future Improvements
Deep Learning Recommenders
Neural Collaborative Filtering
Real-Time Recommendations
Movie Posters & APIs
User Authentication
Cloud Database Integration
🧾 Resume Description

Developed a scalable hybrid recommendation system using collaborative filtering (SVD) and content-based filtering (TF-IDF) with cold-start handling, explainable recommendations, and Streamlit deployment.

👨‍💻 Author

Abhishek Rawal

Hybrid Recommendation System Project
Machine Learning • Data Science • Recommender Systems
