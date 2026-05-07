import streamlit as st
import pandas as pd

from recommender.data_loader import load_movies, load_ratings
from recommender.collaborative import CollaborativeFiltering
from recommender.content_based import ContentBasedRecommender
from recommender.hybrid import HybridRecommender


# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Smart Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# -------------------------------------------------
# CUSTOM CSS (PROFESSIONAL UI)
# -------------------------------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #141E30, #243B55);
    color: white;
}
.title {
    font-size: 44px;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #00F260, #0575E6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #dddddd;
    margin-bottom: 25px;
}
.card {
    background: rgba(255,255,255,0.08);
    padding: 18px;
    border-radius: 14px;
    margin-bottom: 16px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.35);
}
.movie-title {
    font-size: 18px;
    font-weight: 600;
    color: #00F260;
}
.explain {
    font-size: 13px;
    color: #cccccc;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🎥 Smart Recommendation System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Hybrid, Explainable & Cold-Start Aware Recommendation Engine</div>',
    unsafe_allow_html=True
)

# -------------------------------------------------
# LOAD DATA & MODELS (CACHED)
# -------------------------------------------------
@st.cache_data(show_spinner=True)
def load_all():
    movies = load_movies("data/movies.csv")
    ratings = load_ratings("data/ratings.csv")

    # Reduce data for performance
    top_users = ratings["userId"].value_counts().head(5000).index
    top_movies = ratings["movieId"].value_counts().head(3000).index

    ratings = ratings[
        ratings["userId"].isin(top_users) &
        ratings["movieId"].isin(top_movies)
    ]

    movies = movies[movies["movieId"].isin(ratings["movieId"].unique())]

    # Models
    cf = CollaborativeFiltering(ratings)
    cf.build_user_item_matrix()
    cf.train_svd(n_components=20)

    cb = ContentBasedRecommender(movies)
    cb.build_model()

    hybrid = HybridRecommender(cf, cb, ratings)

    return movies, ratings, cf, hybrid


with st.spinner("🚀 Loading models..."):
    movies, ratings, cf, hybrid = load_all()

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.title("👤 User Type")

user_type = st.sidebar.radio(
    "Select user type",
    ["New User", "Existing User"]
)

# -------------------------------------------------
# 📊 ANALYTICS DASHBOARD
# -------------------------------------------------
st.sidebar.markdown("## 📊 Analytics")

if st.sidebar.checkbox("Show Analytics"):
    st.subheader("📈 Dataset Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Rating Distribution")
        st.bar_chart(ratings["rating"].value_counts().sort_index())

    with col2:
        st.write("### Top 10 Popular Movies")
        top_movies = ratings["movieId"].value_counts().head(10)
        st.bar_chart(top_movies)

# =================================================
# 🆕 NEW USER FLOW (ONBOARDING)
# =================================================
if user_type == "New User":

    st.subheader("🎯 New User Onboarding")

    onboarding_mode = st.radio(
        "How would you like to start?",
        ["Select Genres", "Rate 5 Movies"]
    )

    # -------- GENRE SELECTION --------
    if onboarding_mode == "Select Genres":

        all_genres = sorted(set(" ".join(movies["genres"]).split()))

        selected_genres = st.multiselect(
            "Choose your preferred genres",
            all_genres
        )

        if st.button("✨ Get Recommendations") and selected_genres:

            filtered_movies = movies[
                movies["genres"].str.contains("|".join(selected_genres), case=False)
            ]

            popular_ids = (
                ratings["movieId"]
                .value_counts()
                .loc[filtered_movies["movieId"]]
                .head(10)
                .index
            )

            recommendations = movies[movies["movieId"].isin(popular_ids)]

            st.markdown("## 🎬 Recommended for You")

            for _, row in recommendations.iterrows():
                st.markdown(f"""
                <div class="card">
                    <div class="movie-title">{row['title']}</div>
                    <div class="explain">
                        Because you selected genres: {row['genres']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # -------- RATE 5 MOVIES --------
    else:
        st.markdown("⭐ Rate any 5 movies to personalize faster")

        sample_movies = movies.sample(5, random_state=42)
        user_ratings = {}

        for _, row in sample_movies.iterrows():
            user_ratings[row["movieId"]] = st.slider(
                row["title"],
                1, 5, 3,
                key=f"rate_{row['movieId']}"
            )

        if st.button("🚀 Get Personalized Recommendations"):
            temp_user_id = ratings["userId"].max() + 1

            temp_ratings = pd.DataFrame({
                "userId": [temp_user_id]*5,
                "movieId": list(user_ratings.keys()),
                "rating": list(user_ratings.values())
            })

            new_ratings = pd.concat([ratings, temp_ratings])

            cf_temp = CollaborativeFiltering(new_ratings)
            cf_temp.build_user_item_matrix()
            cf_temp.train_svd(n_components=20)

            recs = cf_temp.recommend_movies(temp_user_id, movies)

            st.markdown("## 🎬 Personalized for You")

            for _, row in recs.iterrows():
                st.markdown(f"""
                <div class="card">
                    <div class="movie-title">{row['title']}</div>
                    <div class="explain">
                        Based on your ratings and similar users
                    </div>
                </div>
                """, unsafe_allow_html=True)

# =================================================
# 👥 EXISTING USER FLOW
# =================================================
else:
    user_id = st.sidebar.selectbox(
        "Select User ID",
        sorted(ratings["userId"].unique())
    )

    if st.sidebar.button("🎯 Recommend Movies"):
        recs = hybrid.recommend(user_id, movies, top_n=10)

        st.markdown("## 🎬 Personalized Recommendations")

        for _, row in recs.iterrows():
            st.markdown(f"""
            <div class="card">
                <div class="movie-title">{row['title']}</div>
                <div class="explain">
                    Recommended because users similar to you liked this
                    and it matches genres: {row['genres']}
                </div>
            </div>
            """, unsafe_allow_html=True)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("""
<hr>
<center style="color:#bbb;">
Hybrid Recommendation System • Explainable AI • Streamlit Deployment
</center>
""", unsafe_allow_html=True)
