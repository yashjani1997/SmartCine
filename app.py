import streamlit as st
import pickle
import pandas as pd
import numpy as np
import requests
import os
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="SmartCine", page_icon="🎬", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp { background-color: #0b0f1a; color: white; }
.block-container { padding-top: 0rem; }

header {visibility: hidden;}
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}

/* Navbar */
.navbar {
    background: linear-gradient(to right, #000000, #111827);
    padding: 15px 30px;
    font-size: 24px;
    font-weight: 700;
}

/* Hero */
.hero-title { font-size:56px; font-weight:900; margin-top:20px; }
.hero-overview { font-size:18px; color:#d1d5db; max-width:900px; }

/* Blur effect */
.blur {
    filter: blur(8px);
    transition: filter 0.4s ease;
}

/* Buttons */
button[kind="secondary"] {
    background-color: rgba(255,255,255,0.1) !important;
    color: white !important;
    border-radius: 6px !important;
    border: 1px solid rgba(255,255,255,0.3) !important;
}
button[kind="secondary"]:hover {
    background-color: rgba(255,255,255,0.25) !important;
}

hr { border: 1px solid #222; }
</style>
""", unsafe_allow_html=True)

# ---------------- NAVBAR ----------------
st.markdown("<div class='navbar'>🎬 SmartCine Cinematic</div>", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = None

# ---------------- TMDB ----------------
api_key = st.secrets["TMDB_API_KEY"]
TMDB_BASE = "https://api.themoviedb.org/3/movie/"
IMG_BASE = "https://image.tmdb.org/t/p/w500"
BACKDROP_BASE = "https://image.tmdb.org/t/p/original"

# ---------------- LOAD ARTIFACTS ----------------
@st.cache_data
def load_artifacts():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "artifacts.pkl")
    with open(file_path, "rb") as f:
        return pickle.load(f)

artifacts = load_artifacts()
df = artifacts["df"]
X_reduced = artifacts["X_reduced"]

# ---------------- TMDB FETCH ----------------
@st.cache_data
def fetch_movie(movie_id):
    try:
        url = f"{TMDB_BASE}{int(movie_id)}?api_key={api_key}&language=en-US"
        res = requests.get(url, timeout=5)
        data = res.json()

        # fetch trailer
        trailer_url = None
        videos = requests.get(f"{TMDB_BASE}{int(movie_id)}/videos?api_key={api_key}").json()
        for v in videos.get("results", []):
            if v["type"] == "Trailer" and v["site"] == "YouTube":
                trailer_url = f"https://www.youtube.com/watch?v={v['key']}"
                break

        return {
            "poster": IMG_BASE + data["poster_path"] if data.get("poster_path") else None,
            "backdrop": BACKDROP_BASE + data["backdrop_path"] if data.get("backdrop_path") else None,
            "overview": data.get("overview", ""),
            "rating": data.get("vote_average", 0),
            "title": data.get("title", ""),
            "trailer": trailer_url
        }
    except:
        return None

# ---------------- RECOMMENDER ----------------
def recommend(title_query, top_n=10):
    matches = df[df["original_title"].str.lower().str.contains(title_query.lower())]
    if matches.empty:
        return None, None

    idx = matches.index[0]
    movie_vec = X_reduced[idx].reshape(1, -1)
    cosines = cosine_similarity(movie_vec, X_reduced).flatten()

    df_temp = df.copy()
    df_temp["score"] = cosines
    df_temp = df_temp.drop(index=idx)

    recs = df_temp.sort_values("score", ascending=False).head(top_n)
    return df.loc[idx], recs

# ---------------- HERO SECTION ----------------
def show_hero(movie_id):
    movie = fetch_movie(movie_id)
    if not movie:
        return

    if movie["backdrop"]:
        st.image(movie["backdrop"], use_column_width=True)

    st.markdown(f"<div class='hero-title'>{movie['title']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='hero-overview'>{movie['overview']}</div>", unsafe_allow_html=True)
    st.markdown(f"⭐ {movie['rating']}")

    if movie["trailer"]:
        st.video(movie["trailer"])

    st.markdown("<hr>", unsafe_allow_html=True)

# ---------------- MOVIE GRID ----------------
def movie_grid(df_rows, cols=6):
    rows = list(df_rows.itertuples())
    for i in range(0, len(rows), cols):
        columns = st.columns(cols)
        for j, col in enumerate(columns):
            if i + j >= len(rows):
                continue

            row = rows[i + j]
            movie_id = getattr(row, "id", None)
            movie = fetch_movie(movie_id)

            with col:
                if movie and movie["poster"]:
                    st.image(movie["poster"], use_column_width=True)

                if st.button("▶", key=f"btn_{movie_id}_{i}_{j}", use_container_width=True):
                    st.session_state.selected_movie = movie_id

# ---------------- APP UI ----------------
# Always show selected movie on top
if st.session_state.selected_movie:
    show_hero(st.session_state.selected_movie)

tabs = st.tabs(["🏠 Home", "🔎 Search", "🔥 Trending"])

# HOME
with tabs[0]:

    st.subheader("🔥 Popular")
    popular = df.sort_values("popularity", ascending=False).head(18)
    movie_grid(popular)

# SEARCH
with tabs[1]:
    movie_input = st.text_input("Search Movie")
    if st.button("Search"):
        seed, recs = recommend(movie_input, top_n=18)
        if seed:
            st.session_state.selected_movie = seed["id"]
            show_hero(seed["id"])
            movie_grid(recs)
        else:
            st.error("Movie not found")

# TRENDING
with tabs[2]:
    trending = df.sort_values("vote_average", ascending=False).head(18)
    movie_grid(trending)

st.markdown("<center>Built with ❤ — SmartCine Cinematic</center>", unsafe_allow_html=True)
