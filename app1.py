# SmartCine V3 - Full Feature App.py
# Paste this entire file as app.py (replace existing)

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import requests
from io import BytesIO

# ---------------------------
# PAGE CONFIG & STYLES
# ---------------------------
st.set_page_config(page_title="🎬 SmartCine V3", page_icon="🎥", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background-color: #0b0f1a; color: #e6eef8; }
    .dark .stButton>button { background-color: #1f2937; color: white; }
    .movie-card { padding:10px; border-radius:8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# small helper for dark/light style toggle — we will keep default streamlit theme but add card background choices
THEME_DARK = True

# ---------------------------
# UTILS
# ---------------------------
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w300"  # w300 for smaller cards; use w500 for bigger

@st.cache_data
def load_artifacts(path="smartcine_artifacts_v2.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def get_image_from_tmdb(poster_path):
    if not poster_path or pd.isna(poster_path) or poster_path == '':
        return None
    url = TMDB_IMG_BASE + poster_path
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        return Image.open(BytesIO(r.content))
    except Exception:
        return None

def short_summary(text, max_sentences=2):
    # Not external AI — just return first N sentences for a concise summary
    if not isinstance(text, str) or text.strip()=="":
        return "No overview available."
    # naive split by period, ensures not extremely long
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    summary = '. '.join(sentences[:max_sentences])
    if not summary.endswith('.'):
        summary = summary + '.'
    return summary

# Hybrid recommender: prefer same cluster, then supplement by cosine similarity on SVD vectors
def hybrid_recommend_by_title(title_query, artifacts, top_n=10, cluster_weight=0.6):
    df = artifacts['df']
    # find matching movie (case-insensitive substring)
    matches = df[df['original_title'].str.lower().str.contains(title_query.lower())]
    if matches.empty:
        return None
    idx = matches.index[0]

    # get cluster
    cluster_col = 'kmeans_cluster_v2' if 'kmeans_cluster_v2' in df.columns else 'kmeans_cluster'
    movie_cluster = df.loc[idx, cluster_col]

    # base candidates: same cluster
    same_cluster = df[df[cluster_col] == movie_cluster].copy()
    # prepare cosine similarity on X_reduced (SVD-space)
    X_reduced = artifacts['X_reduced']
    # ensure index alignment: df index -> row in X_reduced must match
    # we assume artifacts['df'] and X_reduced share the same order (they should)
    movie_vec = X_reduced[idx].reshape(1, -1)
    cosines = cosine_similarity(movie_vec, X_reduced).flatten()

    df_cos = df.copy()
    df_cos['cosine_sim'] = cosines
    # score = cluster_weight*(same_cluster membership) + (1-cluster_weight)*cosine_sim normalized
    cluster_mask = (df[col_name := cluster_col].values == movie_cluster).astype(float)
    # normalize cosine to 0-1
    cos_norm = (cosines - cosines.min()) / (cosines.max() - cosines.min() + 1e-9)
    score = cluster_weight * cluster_mask + (1 - cluster_weight) * cos_norm
    df_cos['hybrid_score'] = score

    # remove the movie itself
    df_cos = df_cos.drop(index=idx)

    recs = df_cos.sort_values('hybrid_score', ascending=False).head(top_n)
    return df.loc[idx], recs[['original_title', 'vote_average', 'popularity', 'hybrid_score']]

# recommend by genre: top popular by cluster/genre
def recommend_by_genre(genre, artifacts, top_n=12):
    df = artifacts['df']
    # genres_parsed likely a list; check membership
    filtered = df[df['genres_parsed'].apply(lambda g: genre in g)]
    return filtered.sort_values('popularity', ascending=False).head(top_n)

# recommend by actor
def recommend_by_actor(actor_name, artifacts, top_n=12):
    df = artifacts['df']
    filtered = df[df['cast_parsed'].apply(lambda cl: any(actor_name.lower() in c.lower() for c in cl))]
    return filtered.sort_values('popularity', ascending=False).head(top_n)

# recommend by director
def recommend_by_director(director_name, artifacts, top_n=12):
    df = artifacts['df']
    filtered = df[df['director'].str.lower().str.contains(director_name.lower())]
    return filtered.sort_values('popularity', ascending=False).head(top_n)

# trending: sort by popularity * log(vote_count + 1)
def trending_movies(artifacts, top_n=12):
    df = artifacts['df'].copy()
    df['trend_score'] = df['popularity'] * np.log1p(df['vote_count'])
    return df.sort_values('trend_score', ascending=False).head(top_n)

# multi-genre filter
def filter_by_genres(genres_list, artifacts, top_n=24, year_range=None):
    df = artifacts['df']
    # require all selected genres
    def has_genres(glist):
        return all([g in glist for g in genres_list])
    filtered = df[df['genres_parsed'].apply(has_genres)]
    if year_range is not None:
        start, end = year_range
        # safe parse release_date year
        def year_ok(x):
            try:
                return start <= int(str(x)[:4]) <= end
            except:
                return False
        filtered = filtered[filtered['release_date'].apply(year_ok)]
    return filtered.sort_values('popularity', ascending=False).head(top_n)

# simple poster grid renderer
def poster_grid(df_rows, cols=4, show_overview=False):
    # df_rows: DataFrame of movies
    rows = list(df_rows.itertuples())
    n = len(rows)
    per_row = cols
    for i in range(0, n, per_row):
        cols_ui = st.columns(per_row)
        for j, col_ui in enumerate(cols_ui):
            idx = i + j
            if idx >= n:
                col_ui.write("")  # empty
                continue
            row = rows[idx]
            # assuming row has poster_path, original_title, vote_average, popularity, overview
            poster_path = getattr(row, 'poster_path', None) if 'poster_path' in df_rows.columns else None
            title = getattr(row, 'original_title', '')
            rating = getattr(row, 'vote_average', '')
            pop = getattr(row, 'popularity', '')
            overview = getattr(row, 'overview', '') if 'overview' in df_rows.columns else ''
            with col_ui:
                # card
                try:
                    img = get_image_from_tmdb(poster_path)
                except Exception:
                    img = None
                if img:
                    st.image(img, use_column_width=True)
                else:
                    st.markdown("🖼️ No poster")
                st.markdown(f"**{title}**")
                st.markdown(f"⭐ {rating}  •  🔥 {pop}")
                if show_overview:
                    st.caption(short_summary(overview, max_sentences=2))
                st.markdown("<hr>", unsafe_allow_html=True)


# ---------------------------
# LOAD ARTIFACTS
# ---------------------------
artifacts = load_artifacts()
df = artifacts['df']
vec = artifacts.get('vec', None)
svd = artifacts.get('svd', None)
kmeans = artifacts.get('kmeans', None)
X_reduced = artifacts.get('X_reduced', None)

# Ensure cluster column exists
if 'kmeans_cluster_v2' not in df.columns and 'kmeans_cluster' in df.columns:
    df['kmeans_cluster_v2'] = df['kmeans_cluster']

# ---------------------------
# APP LAYOUT - TABS
# ---------------------------
st.title("🎬 SmartCine V3 — Next Level Movie Recommender")

tabs = st.tabs(["🏠 Home", "🔎 Search & Recommend", "🎭 Browse by Category", "👥 Actor / Director", "🔥 Trending & Top Rated", "⚙️ Filters & Collections"])

# ---------------------------
# TAB: HOME (Highlights + Trending)
# ---------------------------
with tabs[0]:
    st.header("Welcome to SmartCine V3")
    st.write("Explore curated collections, trending movies, and quick recommendations. Select a tab above to start.")

    st.subheader("🔥 Trending Now")
    top_trending = trending_movies(artifacts, top_n=8)
    poster_grid(top_trending, cols=4, show_overview=False)

    st.subheader("🎯 Curated Collections")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🚀 Best Sci-Fi**")
        sci = df[df['genres_parsed'].apply(lambda g: 'ScienceFiction' in g)].sort_values('vote_average', ascending=False).head(8)
        poster_grid(sci, cols=4)
    with col2:
        st.markdown("**❤️ Best Romance**")
        rom = df[df['genres_parsed'].apply(lambda g: 'Romance' in g)].sort_values('vote_average', ascending=False).head(8)
        poster_grid(rom, cols=4)

# ---------------------------
# TAB: SEARCH & RECOMMEND
# ---------------------------
with tabs[1]:
    st.header("🔎 Search & Hybrid Recommendation")
    st.write("Type a movie name (autocomplete from titles), or choose from suggestions. The system returns hybrid recommendations (cluster + cosine similarity).")

    # autocomplete-like suggestions: provide a selectbox with titles (fast) and also a text input
    all_titles = df['original_title'].dropna().unique().tolist()
    all_titles_sorted = sorted(all_titles)
    movie_choice = st.selectbox("Select a movie (or type one):", ["-- Type / Select --"] + all_titles_sorted)
    custom_input = st.text_input("Or type a movie name (partial allowed):", "")

    query = ""
    if movie_choice and movie_choice != "-- Type / Select --":
        query = movie_choice
    elif custom_input.strip() != "":
        query = custom_input.strip()

    if st.button("Get Recommendations"):
        if not query:
            st.warning("Please select or type a movie first.")
        else:
            item, recs = hybrid_recommend_by_title(query, artifacts, top_n=12, cluster_weight=0.6)
            if item is None:
                st.error("Movie not found in dataset.")
            else:
                st.subheader(f"Recommendations based on: {item['original_title']}")
                # show movie card
                st.markdown("#### Seed Movie")
                poster_grid(pd.DataFrame([item]), cols=1, show_overview=True)
                st.markdown("#### Recommended Movies")
                # recs DataFrame contains columns original_title, vote_average, popularity, hybrid_score
                # we want to show poster grid; find rows in df by title
                rec_titles = recs['original_title'].tolist()
                rec_rows = df[df['original_title'].isin(rec_titles)]
                poster_grid(rec_rows, cols=4, show_overview=False)

# ---------------------------
# TAB: BROWSE BY CATEGORY
# ---------------------------
with tabs[2]:
    st.header("🎭 Browse by Category")
    genre_list = sorted(list({g for sub in df['genres_parsed'].tolist() for g in sub}))
    # map some names to friendly names (ScienceFiction -> Sci-Fi)
    friendly = { "ScienceFiction":"Sci-Fi", "TVMovie":"TV Movie" }
    genre_friendly_list = [friendly.get(g,g) for g in genre_list]

    genre_selected = st.selectbox("Select genre:", genre_friendly_list)
    # convert friendly name back to dataset name
    genre_actual = {v:k for k,v in friendly.items()}.get(genre_selected, genre_selected)

    # show top movies
    filtered = df[df['genres_parsed'].apply(lambda gl: genre_actual in gl)]
    st.write(f"Top {genre_selected} movies (by popularity)")
    top12 = filtered.sort_values('popularity', ascending=False).head(12)
    poster_grid(top12, cols=4, show_overview=True)

# ---------------------------
# TAB: ACTOR / DIRECTOR
# ---------------------------
with tabs[3]:
    st.header("👥 Actor & Director Explorer")
    st.write("Choose an actor or director to view their movies and recommendations.")

    colA, colB = st.columns(2)
    with colA:
        actors_flat = sorted({a for sub in df['cast_parsed'].tolist() for a in sub})
        actor_sel = st.selectbox("Choose Actor:", ["-- None --"] + actors_flat)
    with colB:
        directors_unique = sorted(df['director'].dropna().unique().tolist())
        director_sel = st.selectbox("Choose Director:", ["-- None --"] + directors_unique)

    if actor_sel and actor_sel != "-- None --":
        st.subheader(f"Movies with {actor_sel}")
        actor_movies = recommend_by_actor(actor_sel, artifacts, top_n=12)
        poster_grid(actor_movies, cols=4, show_overview=True)

    if director_sel and director_sel != "-- None --":
        st.subheader(f"Movies by {director_sel}")
        dir_movies = recommend_by_director(director_sel, artifacts, top_n=12)
        poster_grid(dir_movies, cols=4, show_overview=True)

# ---------------------------
# TAB: TRENDING & TOP RATED
# ---------------------------
with tabs[4]:
    st.header("🔥 Trending & Top Rated")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Trending Now")
        trending = trending_movies(artifacts, top_n=8)
        poster_grid(trending, cols=4)
    with col2:
        st.subheader("Top Rated (by vote_average)")
        top_rated = df.sort_values('vote_average', ascending=False).head(12)
        poster_grid(top_rated, cols=4)

    st.subheader("Popularity Distribution")
    fig_pop = px.histogram(df, x="popularity", nbins=50, title="Popularity Distribution")
    st.plotly_chart(fig_pop, use_container_width=True)

# ---------------------------
# TAB: FILTERS & COLLECTIONS
# ---------------------------
with tabs[5]:
    st.header("⚙️ Filters & Curated Collections")

    st.subheader("Multi-Genre + Year Filter")
    # multi-select genres from dataset
    genres_all = sorted(list({g for sub in df['genres_parsed'].tolist() for g in sub}))
    genres_map = { "ScienceFiction":"Sci-Fi", "TVMovie":"TV Movie" }
    genres_friendly = [genres_map.get(g,g) for g in genres_all]
    sel_genres = st.multiselect("Choose genres (movies must include ALL chosen):", genres_friendly)

    # year slider
    min_year = 1900
    max_year = 2025
    year_range = st.slider("Filter by release year range:", min_year, max_year, (1990, 2024))

    # convert back friendly names
    sel_genres_actual = [ {v:k for k,v in genres_map.items()}.get(g,g) for g in sel_genres ]

    if st.button("Apply Filters"):
        if len(sel_genres_actual) == 0:
            st.warning("Pick at least one genre.")
        else:
            res = filter_by_genres(sel_genres_actual, artifacts, top_n=24, year_range=year_range)
            if res.empty:
                st.info("No movies match these filters.")
            else:
                poster_grid(res, cols=4, show_overview=True)

    st.markdown("---")
    st.subheader("Curated Playlists")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Mind-bending Movies**")
        mb = df[df['keywords_parsed'].apply(lambda kw: 'mindbending' in kw or 'dream' in kw or 'twist' in kw)].sort_values('vote_average', ascending=False).head(8)
        poster_grid(mb, cols=4)
    with c2:
        st.markdown("**Space Adventures**")
        space = df[df['genres_parsed'].apply(lambda g: 'ScienceFiction' in g or 'Adventure' in g)].sort_values('popularity', ascending=False).head(8)
        poster_grid(space, cols=4)
    with c3:
        st.markdown("**Award Winners (high rating & votes)**")
        aw = df[(df['vote_average']>7.5) & (df['vote_count']>5000)].sort_values('vote_average', ascending=False).head(8)
        poster_grid(aw, cols=4)

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("---")
st.markdown("<center>Built with ❤ by <b>Jani</b> — SmartCine V3</center>", unsafe_allow_html=True)
