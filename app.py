# # 🎬 SmartCine: Cinematic Hybrid Movie Recommendation System
# # ------------------------------------------------------------
# # Developed by Jani | Data Analyst & ML Enthusiast
# # ------------------------------------------------------------

# import streamlit as st
# import pickle, requests
# import pandas as pd
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.preprocessing import normalize

# # ✅ This must come before any other st. commands
# st.set_page_config(
#     page_title="🎬 SmartCine",
#     layout="wide",
#     page_icon="🎥",
# )

# # ============================
# # 1️⃣ Load Saved Artifacts
# # ============================
# @st.cache_resource
# def load_artifacts():
#     with open("smartcine_artifacts.pkl", "rb") as f:
#         data = pickle.load(f)
#     return data

# art = load_artifacts()
# df = art["df"]
# vec = art["vec"]
# kmeans = art["kmeans"]
# tags_matrix = art["tags_matrix"]

# tags_matrix_norm = normalize(tags_matrix, axis=1)

# # ============================
# # 2️⃣ Recommend Function
# # ============================
# def recommend(movie_name):
#     movie_name = movie_name.strip()
#     matches = df[df["original_title"].str.lower().str.contains(movie_name.lower())]

#     if matches.empty:
#         return [], "Movie not found in database."

#     idx = matches.index[0]
#     cluster = df.loc[idx, "kmeans_cluster"]

#     candidates = df[df["kmeans_cluster"] == cluster].index.tolist()
#     candidates = [i for i in candidates if i != idx]

#     sims = cosine_similarity(tags_matrix_norm[idx], tags_matrix_norm[candidates]).flatten()
#     top_idx = np.argsort(-sims)[:12]
#     recs = df.loc[[candidates[i] for i in top_idx], ["original_title", "overview", "id"]].reset_index(drop=True)
#     return recs, df.loc[idx, "original_title"]

# # # ============================
# # # 3️⃣ Streamlit Page Setup
# # # ============================
# # st.set_page_config(
# #     page_title="🎬 SmartCine",
# #     layout="wide",
# #     page_icon="🎥",
# # )

# # ============================
# # 4️⃣ Cinematic UI Styling
# # ============================
# st.markdown("""
# <style>
# body {
#     background-color: #0a0a0a;
#     color: #f2f2f2;
#     font-family: 'Poppins', sans-serif;
# }
# h1, h2, h3 {
#     color: #ff0066;
#     text-align: center;
#     text-shadow: 0 0 15px #ff3399;
# }
# .stButton button {
#     background: linear-gradient(90deg, #ff0055, #ff3300);
#     color: white;
#     border-radius: 12px;
#     padding: 0.7rem 1.3rem;
#     font-weight: 600;
#     transition: 0.3s;
# }
# .stButton button:hover {
#     background: linear-gradient(90deg, #ff3300, #ff0055);
#     transform: scale(1.05);
#     box-shadow: 0 0 15px #ff3399;
# }
# .movie-card {
#     background: rgba(255,255,255,0.07);
#     border-radius: 15px;
#     padding: 12px;
#     margin: 10px;
#     box-shadow: 0 0 15px rgba(255, 26, 117, 0.25);
#     transition: transform 0.3s ease, box-shadow 0.3s ease;
# }
# .movie-card:hover {
#     transform: scale(1.04);
#     box-shadow: 0 0 25px rgba(255, 51, 153, 0.5);
# }
# .movie-title {
#     font-size: 1rem;
#     color: #ff66a3;
#     font-weight: bold;
#     text-align: center;
# }
# .poster {
#     border-radius: 10px;
#     width: 100%;
#     height: 320px;
#     object-fit: cover;
# }
# </style>
# """, unsafe_allow_html=True)

# # ============================
# # 5️⃣ TMDB Poster Fetch
# # ============================
# TMDB_KEY = "1f6b8a8dbdeeadb9e18a3ef5b31e7d47"  # Demo key, replace with your own
# BASE_URL = "https://api.themoviedb.org/3/search/movie"

# def get_poster(title):
#     try:
#         response = requests.get(BASE_URL, params={"api_key": TMDB_KEY, "query": title})
#         data = response.json()
#         poster_path = data["results"][0]["poster_path"]
#         return f"https://image.tmdb.org/t/p/w500{poster_path}"
#     except:
#         return "https://via.placeholder.com/500x750?text=No+Poster"

# # ============================
# # 6️⃣ Streamlit Layout
# # ============================
# st.markdown("<h1>🎥 SmartCine: Your AI Movie Recommender</h1>", unsafe_allow_html=True)
# st.markdown("<p style='text-align:center;'>Discover films that match your cinematic soul 🎞️</p>", unsafe_allow_html=True)

# movie_input = st.text_input("🎬 Enter a movie title:", placeholder="e.g. Avatar, Inception, Interstellar")

# if st.button("🚀 Recommend"):
#     if not movie_input.strip():
#         st.warning("Please type a movie title.")
#     else:
#         recs, found = recommend(movie_input)
#         if recs.empty:
#             st.error(found)
#         else:
#             st.markdown(f"<h2>🎞️ Recommended Movies Like <span style='color:#ff3399'>{found}</span></h2>", unsafe_allow_html=True)
#             cols = st.columns(4)
#             for i, (_, row) in enumerate(recs.iterrows()):
#                 with cols[i % 4]:
#                     poster = get_poster(row["original_title"])
#                     st.markdown(f"<div class='movie-card'>", unsafe_allow_html=True)
#                     st.image(poster, use_container_width=True)
#                     st.markdown(f"<p class='movie-title'>{row['original_title']}</p>", unsafe_allow_html=True)
#                     st.markdown(f"<p style='font-size:0.8rem; color:#ccc;'>{row['overview'][:120]}...</p>", unsafe_allow_html=True)
#                     st.markdown("</div>", unsafe_allow_html=True)

# st.markdown("---")
# st.markdown("<p style='text-align:center;'>🚀 Built by <b>Jani</b> with ❤️ | Powered by Streamlit & scikit-learn</p>", unsafe_allow_html=True)

################ ----------------------  SmartCineV2 ---------------------- ################


import streamlit as st
import pickle
import pandas as pd
import numpy as np

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="🎬 SmartCine V2",
    page_icon="🎥",
    layout="wide"
)

st.markdown("<h1 style='text-align:center;'>🎬 SmartCine V2 – Movie Recommendation System</h1>", unsafe_allow_html=True)

# -------------------------------------------------------
# LOAD MODEL ARTIFACTS
# -------------------------------------------------------
@st.cache_data
def load_artifacts():
    with open("smartcine_artifacts_v2.pkl", "rb") as f:
        return pickle.load(f)

artifacts = load_artifacts()
df = artifacts["df"]
vec = artifacts["vec"]
svd = artifacts["svd"]
kmeans = artifacts["kmeans"]
X_reduced = artifacts["X_reduced"]

# -------------------------------------------------------
# FUNCTIONS
# -------------------------------------------------------

def recommend(movie_name, n=10):
    movie_name = movie_name.lower()
    matches = df[df['original_title'].str.lower().str.contains(movie_name)]
    if matches.empty:
        return None, None

    idx = matches.index[0]
    movie_cluster = df.loc[idx, "kmeans_cluster_v2"]

    recs = df[df["kmeans_cluster_v2"] == movie_cluster] \
            .sort_values("popularity", ascending=False) \
            .head(n)[["original_title", "vote_average", "popularity"]]

    return movie_cluster, recs

# -------------------------------------------------------
# 2 MAIN TABS ONLY
# -------------------------------------------------------
tab1, tab2 = st.tabs(
    ["🎥 Movie Name Recommendation",
     "🎭 Browse by Category"]
)

# -------------------------------------------------------
# 1️⃣ TAB — MOVIE NAME RECOMMENDATION
# -------------------------------------------------------
with tab1:
    st.subheader("🎞️ Find Similar Movies")

    movie_input = st.text_input("Enter a movie name:", "")

    if st.button("Recommend Movies"):
        if movie_input.strip() == "":
            st.warning("Please enter a movie name!")
        else:
            cluster_id, recs = recommend(movie_input)

            if recs is None:
                st.error("❌ Movie not found in database!")
            else:
                st.success(f"Movie belongs to Cluster **{cluster_id}**")

                st.write("### 🎬 Recommended Movies:")
                st.dataframe(recs)

# -------------------------------------------------------
# 2️⃣ TAB — CATEGORY-BASED RECOMMENDATION
# -------------------------------------------------------
with tab2:
    st.subheader("🎭 Browse Movies by Category")

    genre_list = sorted([
        "Action","Adventure","Animation","Comedy","Crime","Documentary","Drama",
        "Family","Fantasy","History","Horror","Music","Mystery",
        "Romance","ScienceFiction","TVMovie","Thriller","War","Western"
    ])

    genre_selected = st.selectbox("Select a Genre Category:", genre_list)

    # Filter movies based on category
    filtered_df = df[df['genres_parsed'].apply(lambda g: genre_selected in g)]

    # Top movies in that category
    top_movies = filtered_df.sort_values("popularity", ascending=False).head(12)

    st.write(f"### 🎬 Top {genre_selected} Movies")

    for idx, row in top_movies.iterrows():
        st.markdown(f"""
        **{row['original_title']}**  
        ⭐ Rating: {row['vote_average']}  
        🔥 Popularity: {row['popularity']}
        <hr>
        """, unsafe_allow_html=True)

# -------------------------------------------------------
# FOOTER
# -------------------------------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Built by <b>Jani</b> 🌟 SmartCine V2</p>", unsafe_allow_html=True)
