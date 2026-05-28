import pickle
import os
import requests
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# ── Load saved artefacts ─────────────────────────────────────────────────────
def load_models():
    movies_path     = os.path.join(MODELS_DIR, "movies.pkl")
    similarity_path = os.path.join(MODELS_DIR, "similarity.pkl")

    if not os.path.exists(movies_path) or not os.path.exists(similarity_path):
        raise FileNotFoundError(
            "Model files not found. Please run preprocess.py first:\n"
            "  python preprocess.py"
        )

    movies     = pickle.load(open(movies_path,     "rb"))
    similarity = pickle.load(open(similarity_path, "rb"))
    return movies, similarity


movies, similarity = load_models()

# ── TMDB poster fetcher ───
TMDB_API_KEY   = os.getenv("TMDB_API_KEY", "")
print("API KEY:", TMDB_API_KEY)
TMDB_BASE_URL  = "https://api.themoviedb.org/3/movie"
POSTER_BASE    = "https://image.tmdb.org/t/p/w500"
FALLBACK_IMG   = "https://via.placeholder.com/500x750?text=No+Poster"

def fetch_poster(movie_id: int) -> str:
    if not TMDB_API_KEY:
        return FALLBACK_IMG

    try:
        url = f"{TMDB_BASE_URL}/{movie_id}?api_key={TMDB_API_KEY}"
        print("Request URL:", url) 

        response = requests.get(url, timeout=5)
        print("Status Code:", response.status_code) 

        data = response.json()
        print("Poster Path:", data.get("poster_path"))

        path = data.get("poster_path")
        return (POSTER_BASE + path) if path else FALLBACK_IMG

    except Exception as e:
        print("Error:", e)
        return FALLBACK_IMG


# ── Core recommendation function ─────
def recommend(movie_title: str, n: int = 5) -> list[dict]:
    """
    Return a list of n recommendation dicts for the given movie title.

    Each dict has keys: title, movie_id, poster_url
    """
    matches = movies[movies["title"] == movie_title]
    if matches.empty:
        return []

    idx       = matches.index[0]
    distances = list(enumerate(similarity[idx]))
    distances = sorted(distances, key=lambda x: x[1], reverse=True)

    results = []
    for i, score in distances[1 : n + 1]:
        row = movies.iloc[i]
        results.append({
            "title":      row["title"],
            "movie_id":   int(row["movie_id"]),
            "poster_url": fetch_poster(int(row["movie_id"])),
            "score":      round(float(score), 3),
        })
        print("Movie ID:", row["movie_id"])
    return results


def get_all_titles() -> list[str]:
    """Return sorted list of all movie titles (for dropdown)."""
    return sorted(movies["title"].tolist())
