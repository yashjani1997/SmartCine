import streamlit as st
import pickle
import pandas as pd
import numpy as np
import requests
import os
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="SmartCine", page_icon="🎬", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #0e1117; color: white; }
.movie-title { font-size:18px; font-weight:600; }
.hero-title { font-size:48px; font-weight:800; margin-bottom:10px; }
.hero-overview { font-size:18px; color:#ccc; }
hr { border: 1px solid #333; }
</style>
""", unsafe_allow_html=True)

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

        return {
            "poster": IMG_BASE + data["poster_path"] if data.get("poster_path") else None,
            "backdrop": BACKDROP_BASE + data["backdrop_path"] if data.get("backdrop_path") else None,
            "overview": data.get("overview", ""),
            "rating": data.get("vote_average", 0),
            "title": data.get("title", "")
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

# ---------------- HERO BANNER ----------------
def show_hero(movie_id):
    movie = fetch_movie(movie_id)
    if not movie:
        return

    if movie["backdrop"]:
        st.image(movie["backdrop"], use_column_width=True)

    st.markdown(f"<div class='hero-title'>{movie['title']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='hero-overview'>{movie['overview']}</div>", unsafe_allow_html=True)
    st.markdown(f"⭐ {movie['rating']}")
    st.markdown("<hr>", unsafe_allow_html=True)

# ---------------- MOVIE GRID ----------------
def movie_grid(df_rows, cols=5):
    rows = list(df_rows.itertuples())
    for i in range(0, len(rows), cols):
        columns = st.columns(cols)
        for j, col in enumerate(columns):
            if i + j >= len(rows):
                continue

            row = rows[i + j]
            movie_id = getattr(row, "id", None)

            with col:
                movie = fetch_movie(movie_id)

                if movie and movie["poster"]:
                    st.image(movie["poster"], use_column_width=True)

                st.markdown(f"<div class='movie-title'>{row.original_title}</div>", unsafe_allow_html=True)

                if st.button("▶", key=f"btn_{movie_id}"):
                    st.session_state.selected_movie = movie_id

# ---------------- APP UI ----------------
st.title("🎬 SmartCine — Netflix Style")

tabs = st.tabs(["🏠 Home", "🔎 Search", "🔥 Trending"])

# ---------- HOME ----------
with tabs[0]:

    if st.session_state.selected_movie:
        show_hero(st.session_state.selected_movie)

    st.subheader("🔥 Popular Movies")
    popular = df.sort_values("popularity", ascending=False).head(15)
    movie_grid(popular)

# ---------- SEARCH ----------
with tabs[1]:
    movie_input = st.text_input("Search Movie")

    if st.button("Search"):
        seed, recs = recommend(movie_input, top_n=15)

        if seed is None:
            st.error("Movie not found")
        else:
            st.session_state.selected_movie = seed["id"]
            show_hero(seed["id"])
            movie_grid(recs)

# ---------- TRENDING ----------
with tabs[2]:
    trending = df.sort_values("vote_average", ascending=False).head(15)
    movie_grid(trending)

st.markdown("---")
st.markdown("<center>Built with ❤ — SmartCine</center>", unsafe_allow_html=True)
