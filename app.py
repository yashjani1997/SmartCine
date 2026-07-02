import streamlit as st
import pickle
import pandas as pd
import numpy as np
import requests
import os
import json
from google import genai
from google.genai import types
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

.navbar {
    background: linear-gradient(to right, #000000, #111827);
    padding: 15px 30px;
    font-size: 24px;
    font-weight: 700;
}

.hero-title { font-size:56px; font-weight:900; margin-top:20px; }
.hero-overview { font-size:18px; color:#d1d5db; max-width:900px; }

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

/* ---- Chat UI Styles ---- */
.chat-bubble-user {
    background: #1e3a5f;
    border-radius: 16px 16px 4px 16px;
    padding: 12px 16px;
    margin: 8px 0;
    max-width: 75%;
    margin-left: auto;
    color: white;
    font-size: 15px;
}
.chat-bubble-ai {
    background: #1a1f2e;
    border: 1px solid #2d3748;
    border-radius: 16px 16px 16px 4px;
    padding: 12px 16px;
    margin: 8px 0;
    max-width: 75%;
    color: #e2e8f0;
    font-size: 15px;
}
.chat-intent-badge {
    display: inline-block;
    background: #2d3748;
    color: #90cdf4;
    border-radius: 8px;
    padding: 2px 10px;
    font-size: 12px;
    margin-bottom: 6px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='navbar'>🎬 SmartCine Cinematic</div>", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # list of {"role": "user"/"ai", "content": str, "intent": dict}

# ---------------- API KEYS ----------------
api_key = st.secrets["TMDB_API_KEY"]
gemini_client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

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

# ---------------- LIVE TRENDING ----------------
@st.cache_data(ttl=3600)
def fetch_trending_movies():
    try:
        url = f"https://api.themoviedb.org/3/trending/movie/week?api_key={api_key}"
        res = requests.get(url, timeout=5)
        data = res.json()
        movies = []
        for m in data.get("results", []):
            movies.append({"id": m["id"], "original_title": m["title"]})
        return pd.DataFrame(movies)
    except:
        return pd.DataFrame()

# ---------------- FETCH GENRES FOR NEW MOVIE ----------------
@st.cache_data(ttl=3600)
def fetch_movie_genres(movie_name):
    try:
        search_url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={movie_name}"
        search_res = requests.get(search_url, timeout=5).json()
        if not search_res.get("results"):
            return None, None
        movie = search_res["results"][0]
        movie_id = movie["id"]
        details = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}"
        ).json()
        genres = [g["name"] for g in details.get("genres", [])]
        return movie_id, genres
    except:
        return None, None

# ---------------- GENRE BASED FALLBACK ----------------
def genre_based_recommendation(genres, top_n=18):
    if not genres:
        return pd.DataFrame()
    filtered = df[df["genres_parsed"].apply(
        lambda g_list: any(g in g_list for g in genres)
    )]
    return filtered.sort_values("popularity", ascending=False).head(top_n)

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

# ================================================================
# ---- V4: GEMINI INTENT PARSER ----
# ================================================================

SYSTEM_PROMPT = """You are an intelligent movie query parser for SmartCine, a movie recommendation system.

Your ONLY job is to extract structured intent from the user's natural language movie request and return a JSON object. Nothing else.

Rules:
- Always respond with ONLY valid JSON. No explanation, no markdown, no extra text.
- If a field is not mentioned or cannot be inferred, set it to null.
- genres must be from standard TMDB genres: Action, Adventure, Animation, Comedy, Crime, Documentary, Drama, Family, Fantasy, History, Horror, Music, Mystery, Romance, Science Fiction, Thriller, War, Western.
- mood must be one of: light, dark, intense, emotional, fun, inspiring, scary, romantic, suspenseful.
- query_type must be one of: movie_based, mood_based, actor_director_based, trending.
- If user mentions a specific movie name → query_type = "movie_based", put it in similar_to.
- If user mentions actor/director → query_type = "actor_director_based".
- If user asks for trending/popular/new releases → query_type = "trending".
- Otherwise → query_type = "mood_based".
- language: detect if user wants hindi/english/regional. Default null.
- avoid: list of things user explicitly does NOT want (genres, moods, etc.).

Output Format (strict):
{
  "genres": [] or null,
  "mood": "" or null,
  "language": "" or null,
  "year_range": {"min": int, "max": int} or null,
  "actors": [] or null,
  "director": "" or null,
  "avoid": [] or null,
  "similar_to": "" or null,
  "occasion": "" or null,
  "query_type": "mood_based"
}

Examples:

User: "kuch light comedy chahiye family ke saath dekhne ke liye"
Output: {"genres": ["Comedy", "Family"], "mood": "light", "language": null, "year_range": null, "actors": null, "director": null, "avoid": null, "similar_to": null, "occasion": "family", "query_type": "mood_based"}

User: "Zindagi Na Milegi Dobara jaisi koi movie do"
Output: {"genres": null, "mood": null, "language": "hindi", "year_range": null, "actors": null, "director": null, "avoid": null, "similar_to": "Zindagi Na Milegi Dobara", "occasion": null, "query_type": "movie_based"}

User: "Shah Rukh Khan ki best romantic movies"
Output: {"genres": ["Romance"], "mood": "romantic", "language": "hindi", "year_range": null, "actors": ["Shah Rukh Khan"], "director": null, "avoid": null, "similar_to": null, "occasion": null, "query_type": "actor_director_based"}

User: "kuch dark thriller do, horror nahi chahiye, 2010 ke baad ki"
Output: {"genres": ["Thriller", "Crime"], "mood": "dark", "language": null, "year_range": {"min": 2010, "max": 2024}, "actors": null, "director": null, "avoid": ["Horror"], "similar_to": null, "occasion": null, "query_type": "mood_based"}

User: "what's trending this week"
Output: {"genres": null, "mood": null, "language": null, "year_range": null, "actors": null, "director": null, "avoid": null, "similar_to": null, "occasion": null, "query_type": "trending"}
"""


def parse_intent(user_query: str, chat_history: list) -> dict:
    """
    Sends user query + recent chat context to Gemini.
    Returns structured intent dict.
    """
    # Build recent context (last 3 turns) so follow-ups work
    context = ""
    if chat_history:
        recent = chat_history[-6:]  # last 3 user+ai pairs
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            context += f"{role}: {msg['content']}\n"
        context = f"Previous context:\n{context}\n"

    full_prompt = f"{context}New user query: {user_query}"

    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=full_prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.1
        )
    )
    raw = response.text.strip()

    # Strip markdown fences if Gemini wraps in ```json ... ```
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    return json.loads(raw)


def route_and_recommend(intent: dict, top_n: int = 12):
    """
    Routes parsed intent to correct recommendation path.
    Returns (seed_movie_or_None, recs_df, route_label)
    """
    query_type = intent.get("query_type", "mood_based")
    genres = intent.get("genres") or []
    actors = intent.get("actors") or []
    director = intent.get("director")
    similar_to = intent.get("similar_to")
    year_range = intent.get("year_range")
    avoid = intent.get("avoid") or []

    # --- movie_based: cosine similarity ---
    if query_type == "movie_based" and similar_to:
        seed, recs = recommend(similar_to, top_n=top_n)
        if recs is not None and not recs.empty:
            return seed, recs, "🎯 Similar to " + similar_to
        # Fallback: TMDB genre search for that movie
        movie_id, fetched_genres = fetch_movie_genres(similar_to)
        if movie_id and fetched_genres:
            recs = genre_based_recommendation(fetched_genres, top_n=top_n)
            return None, recs, "🎯 Genre match for " + similar_to

    # --- trending ---
    elif query_type == "trending":
        trending = fetch_trending_movies()
        return None, trending, "🔥 Trending This Week"

    # --- actor_director_based ---
    elif query_type == "actor_director_based":
        filtered = df.copy()

        if actors:
            # Check if cast column exists (could be cast_parsed or similar)
            cast_col = None
            for col in ["cast_parsed", "cast", "actors"]:
                if col in filtered.columns:
                    cast_col = col
                    break

            if cast_col:
                filtered = filtered[filtered[cast_col].apply(
                    lambda c: any(
                        actor.lower() in str(c).lower() for actor in actors
                    )
                )]

        if director:
            dir_col = None
            for col in ["crew_parsed", "director", "crew"]:
                if col in filtered.columns:
                    dir_col = col
                    break
            if dir_col:
                filtered = filtered[
                    filtered[dir_col].str.contains(director, case=False, na=False)
                ]

        # Apply year filter if given
        if year_range and "release_date" in filtered.columns:
            filtered["year"] = pd.to_datetime(
                filtered["release_date"], errors="coerce"
            ).dt.year
            filtered = filtered[
                (filtered["year"] >= year_range["min"]) &
                (filtered["year"] <= year_range["max"])
            ]

        if filtered.empty:
            # Fallback to genre if actor search fails
            if genres:
                filtered = genre_based_recommendation(genres, top_n=top_n)
            else:
                filtered = df.sample(top_n)

        label = "🎬 " + (", ".join(actors) if actors else director or "Filter")
        return None, filtered.sort_values("popularity", ascending=False).head(top_n), label

    # --- mood_based (default) ---
    filtered = df.copy()

    if genres:
        filtered = filtered[filtered["genres_parsed"].apply(
            lambda g_list: any(g in g_list for g in genres)
        )]

    # Avoid filter
    if avoid and not filtered.empty:
        filtered = filtered[filtered["genres_parsed"].apply(
            lambda g_list: not any(a in g_list for a in avoid)
        )]

    # Year filter
    if year_range and "release_date" in filtered.columns and not filtered.empty:
        filtered["year"] = pd.to_datetime(
            filtered["release_date"], errors="coerce"
        ).dt.year
        filtered = filtered[
            (filtered["year"] >= year_range["min"]) &
            (filtered["year"] <= year_range["max"])
        ]

    if filtered.empty:
        filtered = df.sample(top_n)

    label = "✨ " + (", ".join(genres) if genres else "Curated for you")
    return None, filtered.sort_values("popularity", ascending=False).head(top_n), label


def build_ai_reply(intent: dict, route_label: str) -> str:
    """Generates a friendly chat reply based on what was understood."""
    query_type = intent.get("query_type", "mood_based")
    mood = intent.get("mood")
    genres = intent.get("genres") or []
    similar_to = intent.get("similar_to")
    actors = intent.get("actors") or []
    avoid = intent.get("avoid") or []

    if query_type == "movie_based" and similar_to:
        return f"Got it! Finding movies similar to **{similar_to}** using our ML engine 🎯"
    elif query_type == "trending":
        return "Here's what's hot this week on the big screen 🔥"
    elif query_type == "actor_director_based":
        names = ", ".join(actors) if actors else intent.get("director", "")
        return f"Pulling up movies featuring **{names}** ✨"
    else:
        parts = []
        if mood:
            parts.append(f"**{mood}** mood")
        if genres:
            parts.append(", ".join(f"**{g}**" for g in genres))
        if avoid:
            parts.append(f"avoiding {', '.join(avoid)}")
        desc = " · ".join(parts) if parts else "something great"
        return f"Searching for {desc} recommendations 🍿"


# ================================================================
# ---- SHARED UI COMPONENTS ----
# ================================================================

def show_hero(movie_id):
    movie = fetch_movie(movie_id)
    if not movie:
        return
    if movie["backdrop"]:
        st.image(movie["backdrop"], width='stretch')
    st.markdown(f"<div class='hero-title'>{movie['title']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='hero-overview'>{movie['overview']}</div>", unsafe_allow_html=True)
    st.markdown(f"⭐ {movie['rating']}")
    if movie["trailer"]:
        st.video(movie["trailer"])
    st.markdown("<hr>", unsafe_allow_html=True)


def movie_grid(df_rows, section_name="home", cols=5):
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
                    st.image(movie["poster"], width='stretch')
                st.markdown(
                    f"<div class='movie-title'>{row.original_title}</div>",
                    unsafe_allow_html=True
                )
                unique_key = f"{section_name}_btn_{movie_id}_{row.Index}"
                if st.button("▶", key=unique_key, use_container_width=True):
                    st.session_state.selected_movie = movie_id


# ================================================================
# ---- APP SHELL ----
# ================================================================

if st.session_state.selected_movie:
    show_hero(st.session_state.selected_movie)

tabs = st.tabs(["🏠 Home", "🔎 Search", "🔥 Trending", "🤖 Ask AI"])

# ---- HOME ----
with tabs[0]:
    st.subheader("🔥 Popular")
    popular = df.sort_values("popularity", ascending=False).head(18)
    movie_grid(popular, section_name="home")

# ---- SEARCH ----
with tabs[1]:
    movie_input = st.text_input("Search Movie")
    if st.button("Search"):
        seed, recs = recommend(movie_input, top_n=18)
        if seed is not None:
            st.session_state.selected_movie = seed["id"]
            show_hero(seed["id"])
            movie_grid(recs, section_name="search_ml")
        else:
            movie_id, genres = fetch_movie_genres(movie_input)
            if movie_id:
                st.session_state.selected_movie = movie_id
                show_hero(movie_id)
                st.subheader("🎯 Genre-Based Recommendations")
                genre_recs = genre_based_recommendation(genres)
                if not genre_recs.empty:
                    movie_grid(genre_recs, section_name="search_genre")
                else:
                    st.info("No similar movies found in dataset.")
            else:
                st.error("Movie not found anywhere.")

# ---- TRENDING ----
with tabs[2]:
    st.subheader("🔥 Live Trending This Week")
    trending_live = fetch_trending_movies()
    if not trending_live.empty:
        movie_grid(trending_live, section_name="trending_live")
    else:
        st.error("Unable to fetch trending movies.")

# ================================================================
# ---- V4: ASK AI TAB ----
# ================================================================
with tabs[3]:
    st.subheader("🤖 Ask AI — Tell me what you're in the mood for")
    st.caption("Try: *'kuch light comedy chahiye'* · *'Interstellar jaisi movie do'* · *'dark thriller, horror nahi'*")

    # --- Render chat history ---
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f"<div class='chat-bubble-user'>🧑 {msg['content']}</div>",
                unsafe_allow_html=True
            )
        else:
            intent = msg.get("intent", {})
            qt = intent.get("query_type", "")
            st.markdown(
                f"<div class='chat-bubble-ai'>"
                f"<span class='chat-intent-badge'>🧠 {qt}</span><br>"
                f"🤖 {msg['content']}"
                f"</div>",
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Input row ---
    col_input, col_btn, col_clear = st.columns([6, 1, 1])

    with col_input:
        user_query = st.text_input(
            "Your message",
            label_visibility="collapsed",
            placeholder="Kuch suggest karo...",
            key="chat_input"
        )

    with col_btn:
        send = st.button("Send", use_container_width=True)

    with col_clear:
        if st.button("Clear", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # --- Handle send ---
    if send and user_query.strip():
        # 1. Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_query.strip()
        })

        try:
            # 2. Parse intent via Gemini
            with st.spinner("🧠 Thinking..."):
                intent = parse_intent(user_query.strip(), st.session_state.chat_history)

            # 3. Route to ML engine
            seed, recs, route_label = route_and_recommend(intent, top_n=12)

            # 4. Build friendly reply
            ai_reply = build_ai_reply(intent, route_label)

            # 5. Save AI message to history
            st.session_state.chat_history.append({
                "role": "ai",
                "content": ai_reply,
                "intent": intent
            })

            # 6. Show results
            st.markdown(f"### {route_label}")

            if seed is not None:
                st.session_state.selected_movie = seed["id"]
                show_hero(seed["id"])

            if recs is not None and not recs.empty:
                movie_grid(recs, section_name="chat_results")
            else:
                st.info("No results found. Try rephrasing your request.")

        except json.JSONDecodeError:
            st.error("Gemini returned unexpected format. Please try again.")
        except Exception as e:
            st.error(f"Something went wrong: {str(e)}")

        st.rerun()

st.markdown("<center>Built with ❤ — SmartCine Cinematic V4</center>", unsafe_allow_html=True)
