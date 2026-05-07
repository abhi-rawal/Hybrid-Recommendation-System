~from recommender.data_loader import load_movies, load_ratings
from recommender.collaborative import CollaborativeFiltering
from recommender.content_based import ContentBasedRecommender
from recommender.hybrid import HybridRecommender
from recommender.evaluation import precision_recall_at_k


def main():
    print("Loading data...")
    movies = load_movies("data/movies.csv")
    ratings = load_ratings("data/ratings.csv")

    # =====================================================
    # 🔥 DATA REDUCTION (CRITICAL FOR MEMORY SAFETY)
    # =====================================================
    print("Filtering active users and popular movies...")

    top_users = ratings["userId"].value_counts().head(5000).index
    top_movies = ratings["movieId"].value_counts().head(3000).index

    ratings = ratings[
        ratings["userId"].isin(top_users)
        & ratings["movieId"].isin(top_movies)
    ]

    # Keep only movies that appear in filtered ratings
    movies = movies[movies["movieId"].isin(ratings["movieId"].unique())]

    print(f"Remaining ratings: {len(ratings)}")
    print(f"Remaining movies: {len(movies)}")

    # =====================================================
    # 🔹 COLLABORATIVE FILTERING (SPARSE SVD)
    # =====================================================
    print("\nTraining Collaborative Filtering...")
    cf = CollaborativeFiltering(ratings)
    cf.build_user_item_matrix()
    cf.train_svd(n_components=20)

    # =====================================================
    # 🔹 CONTENT-BASED FILTERING (ON-THE-FLY SIMILARITY)
    # =====================================================
    print("\nTraining Content-Based Model...")
    cb = ContentBasedRecommender(movies)
    cb.build_model()

    # =====================================================
    # 🔹 HYBRID RECOMMENDER
    # =====================================================
    print("\nInitializing Hybrid Recommender...")
    hybrid = HybridRecommender(cf, cb, ratings)

    # Pick a valid user from filtered data
    user_id = ratings["userId"].iloc[0]

    print(f"\nHybrid recommendations for user {user_id}:\n")
    recommendations = hybrid.recommend(user_id, movies, top_n=10)
    print(recommendations)

    # =====================================================
    # 🔹 EVALUATION (OPTIONAL BUT IMPORTANT)
    # =====================================================
    print("\nEvaluating model...")
    metrics = precision_recall_at_k(cf, ratings, k=10)
    print(f"Precision@10: {metrics['Precision@K']:.4f}")
    print(f"Recall@10: {metrics['Recall@K']:.4f}")


if __name__ == "__main__":
    main()
