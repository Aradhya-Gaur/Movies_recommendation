import pandas as pd
import ast
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


print("Loading datasets...")
movies  = pd.read_csv(os.path.join(DATA_DIR, "tmdb_5000_movies.csv"))
credits = pd.read_csv(os.path.join(DATA_DIR, "tmdb_5000_credits.csv"))

movies = movies.merge(credits, on="title")


movies = movies[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]
movies.dropna(inplace=True)
movies.drop_duplicates(subset="title", inplace=True)
movies.reset_index(drop=True, inplace=True)


def extract_names(obj):
    """Extract 'name' fields from a JSON-encoded list of dicts."""
    try:
        return [i["name"].replace(" ", "") for i in ast.literal_eval(obj)]
    except Exception:
        return []

def extract_top3_cast(obj):
    """Extract top-3 actor names."""
    try:
        return [i["name"].replace(" ", "") for i in ast.literal_eval(obj)[:3]]
    except Exception:
        return []

def extract_director(obj):
    """Extract director name from crew list."""
    try:
        for i in ast.literal_eval(obj):
            if i["job"] == "Director":
                return [i["name"].replace(" ", "")]
        return []
    except Exception:
        return []


print("Parsing columns...")
movies["genres"]   = movies["genres"].apply(extract_names)
movies["keywords"] = movies["keywords"].apply(extract_names)
movies["cast"]     = movies["cast"].apply(extract_top3_cast)
movies["crew"]     = movies["crew"].apply(extract_director)
movies["overview"] = movies["overview"].apply(lambda x: x.split())

movies["tags"] = (
    movies["overview"] +
    movies["genres"]   +
    movies["keywords"] +
    movies["cast"]     +
    movies["crew"]
)
movies["tags"] = movies["tags"].apply(lambda x: " ".join(x).lower())

# Keep only what we need downstream
movies_clean = movies[["movie_id", "title", "tags"]].copy()

# ── Vectorise ────────────────────────────────────────────────────────────────
print("Vectorising tags...")
cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(movies_clean["tags"]).toarray()

# ── Cosine similarity ─────────────────────────────────────────────────────────
print("Computing cosine similarity (this may take a moment)...")
similarity = cosine_similarity(vectors)


pickle.dump(movies_clean, open(os.path.join(MODELS_DIR, "movies.pkl"),     "wb"))
pickle.dump(similarity,   open(os.path.join(MODELS_DIR, "similarity.pkl"), "wb"))

print(f"Done! Saved {len(movies_clean)} movies.")
print(f"  models/movies.pkl      — {movies_clean.shape}")
print(f"  models/similarity.pkl  — {similarity.shape}")
