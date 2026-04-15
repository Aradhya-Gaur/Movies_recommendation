import streamlit as st
from model import recommend, get_all_titles

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .movie-title {
        font-size: 13px;
        font-weight: 600;
        text-align: center;
        margin-top: 6px;
        line-height: 1.3;
    }
    .score-badge {
        font-size: 11px;
        color: #888;
        text-align: center;
        margin-top: 2px;
    }
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 4px;
    }
    .subtitle {
        text-align: center;
        color: #888;
        margin-bottom: 32px;
        font-size: 15px;
    }
    img {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🎬 Movie Recommender</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by cosine similarity on TMDB data</div>', unsafe_allow_html=True)

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    n_results = st.slider("Number of recommendations", min_value=3, max_value=10, value=5)
    show_score = st.checkbox("Show similarity score", value=False)
    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown(
        "Each movie is converted into a tags vector combining "
        "its overview, genres, keywords, cast, and director. "
        "Cosine similarity measures how close two vectors are — "
        "closer = more similar."
    )

# ── Movie selector ────────────────────────────────────────────────────────────
all_titles = get_all_titles()

selected_movie = st.selectbox(
    "Search or select a movie",
    options=all_titles,
    index=all_titles.index("The Dark Knight") if "The Dark Knight" in all_titles else 0,
    placeholder="Type to search...",
)

recommend_btn = st.button("🔍 Find Similar Movies", use_container_width=True, type="primary")

# ── Results ───────────────────────────────────────────────────────────────────
if recommend_btn and selected_movie:
    with st.spinner("Finding recommendations..."):
        results = recommend(selected_movie, n=n_results)

    if not results:
        st.error("Movie not found. Please try another title.")
    else:
        st.markdown(f"### Because you liked **{selected_movie}**")
        cols = st.columns(len(results))

        for col, movie in zip(cols, results):
            with col:
                st.image(movie["poster_url"], use_container_width=True)
                st.markdown(
                    f'<div class="movie-title">{movie["title"]}</div>',
                    unsafe_allow_html=True,
                )
                if show_score:
                    st.markdown(
                        f'<div class="score-badge">Score: {movie["score"]}</div>',
                        unsafe_allow_html=True,
                    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#aaa;font-size:12px;'>"
    "Data: TMDB 5000 Movies Dataset &nbsp;|&nbsp; "
    "Built with Streamlit + scikit-learn"
    "</div>",
    unsafe_allow_html=True,
)