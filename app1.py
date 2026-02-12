import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
import requests

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="🎬 SmartCine V3", page_icon="🎥", layout="wide")

# ---------------------------
# TMDB API (from Streamlit Secrets)
# ---------------------------
api_key = st.secrets["TMDB_API_KEY"]
TMDB_BASE_URL = "https://api.themoviedb.org/3/movie/"
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w500"

# ---------------------------
# LOAD ARTIFACTS
# ---------------------------
@st.cache_data
def load_artifacts(path="artifacts.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

artifacts = load_artifacts()
df = artifacts["df"]
X_reduced = artifacts["X_reduced"]

# ---------------------------
# TMDB LIVE FETCH
# ---------------------------
@st.cache_data
def fetch_movie_from_tmdb(movie_id):
    try:
        url = f"{TMDB_BASE_URL}{movie_id}?api_key={api_key}&language=en-US"
        res = requests.get(url, timeout=5)
        res.raise_for_status()
        data = res.json()

        poster = data.get("poster_path")
        poster_url = TMDB_IMG_BASE + poster if poster else None

        return {
            "poster": poster_url,
            "overview": data.get("overview", ""),
            "rating": data.get("vote_average", 0)
        }
    except:
        return None

# ---------------------------
# HYBRID RECOMMENDATION
# ---------------------------
def hybrid_recommend(title_query, top_n=10):
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

# ---------------------------
# POSTER GRID
# ---------------------------
def poster_grid(df_rows, cols=4):
    rows = list(df_rows.itertuples())
    for i in range(0, len(rows), cols):
        columns = st.columns(cols)
        for j, col in enumerate(columns):
            if i + j >= len(rows):
                continue

            row = rows[i + j]
            movie_id = getattr(row, "movie_id", None)

            with col:
                movie_data = fetch_movie_from_tmdb(movie_id)

                if movie_data and movie_data["poster"]:
                    st.image(movie_data["poster"], use_column_width=True)
                else:
                    st.write("No Poster")

                st.markdown(f"**{row.original_title}**")
                rating = movie_data["rating"] if movie_data else row.vote_average
                st.caption(f"⭐ {rating}")

# ---------------------------
# APP UI
# ---------------------------
st.title("🎬 SmartCine V3 — Live TMDB Powered")

tabs = st.tabs(["🏠 Home", "🔎 Recommend", "🔥 Trending"])

# ---------------- HOME
with tabs[0]:
    st.header("🔥 Top Popular Movies")
    top_pop = df.sort_values("popularity", ascending=False).head(8)
    poster_grid(top_pop)

# ---------------- RECOMMEND
with tabs[1]:
    st.header("🔎 Search Movie")
    movie_input = st.text_input("Type movie name:")

    if st.button("Recommend"):
        if movie_input.strip() == "":
            st.warning("Enter movie name")
        else:
            seed, recs = hybrid_recommend(movie_input, top_n=8)

            if seed is None:
                st.error("Movie not found")
            else:
                st.subheader("🎯 Based on:")
                poster_grid(pd.DataFrame([seed]), cols=1)

                st.subheader("🎬 Recommendations:")
                poster_grid(recs)

# ---------------- TRENDING
with tabs[2]:
    st.header("🔥 Trending (Popularity Based)")
    trending = df.sort_values("popularity", ascending=False).head(12)
    poster_grid(trending)

st.markdown("---")
st.markdown("<center>Built with ❤ — SmartCine V3</center>", unsafe_allow_html=True)
