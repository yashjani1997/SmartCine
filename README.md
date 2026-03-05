# 🎬 SmartCine – Hybrid Movie Recommendation System

SmartCine is a **version-wise evolved movie recommendation system** built using Machine Learning.  
The project demonstrates how a core recommendation engine can be gradually upgraded into a **full-featured, scalable movie discovery platform**.

---

## 🚀 Project Overview

SmartCine recommends movies using a **hybrid approach** that combines:
- Content-based filtering (Cosine Similarity)
- Unsupervised learning (KMeans Clustering)
- Dimensionality Reduction (SVD)
- Live data via **TMDB API**

The system is built on rich metadata from the **TMDB 5000 Movies Dataset**.

---

## 📂 Dataset

- TMDB 5000 Movies
- TMDB 5000 Credits

**Features used:** genres, keywords, cast, director, overview, ratings, popularity.

---

## 🔹 Version Breakdown

### ✅ Version 1 (V1) – Core Recommendation Engine

**Focus:** ML logic & accuracy

- Feature engineering using movie metadata
- Text vectorization and **Cosine Similarity** on TMDB 5000 dataset
- Content-based recommendations from trained dataset

**Why this approach?**  
Cosine similarity on rich metadata provides accurate content-based recommendations as a strong baseline.

---

### 🔁 Version 2 (V2) – Enhanced Logic & Flexibility

**Focus:** Smarter and optimized recommendations

- **SVD (Singular Value Decomposition)** for dimensionality reduction and faster computation
- **KMeans Clustering** to reduce search space and improve scalability
- Weighted hybrid scoring (cluster + similarity)
- Multiple recommendation paths:
  - Movie-based
  - Genre-based
  - Actor & Director-based

**Why upgrade?**  
SVD + KMeans significantly improve performance and flexibility beyond a single recommendation flow.

---

### 🚀 Version 3 (V3) – Full Movie Discovery Platform *(Live)*

**Focus:** Product-level & scalable system

- Advanced Streamlit UI with 3 tabs: **Home, Search, Trending**
- **TMDB API Integration** (fully implemented):
  - `/movie/{id}` — Movie details, ratings, overview
  - `/movie/{id}/videos` — Trailer fetch (YouTube)
  - `/trending/movie/week` — Live trending movies
  - `/search/movie` — Real-time movie search for new/unseen movies
- Hybrid Search Logic:
  - Dataset movies → ML-based Cosine Similarity recommendations
  - New/unseen movies → TMDB API search + Genre-based fallback recommendations
- TMDB movie poster & backdrop integration
- Interactive visualizations (Plotly)
- Trending, popular, and curated collections

**Why V3?**  
Combines trained ML model with live TMDB API data — handles both dataset movies and brand new releases seamlessly.

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| ML & Data | Pandas, NumPy, Scikit-learn |
| ML Techniques | Cosine Similarity, KMeans, SVD |
| UI | Streamlit |
| Visualization | Plotly |
| API | TMDB API |
| Dataset | TMDB 5000 Movies & Credits |

---

## 🔌 TMDB API Endpoints Used

| Endpoint | Purpose |
|---|---|
| `/movie/{id}` | Fetch movie details, poster, backdrop |
| `/movie/{id}/videos` | Fetch YouTube trailer |
| `/trending/movie/week` | Live trending movies |
| `/search/movie?query=` | Real-time search for new movies |

---

## ⚙️ How to Run Locally

```bash
# Clone the repository
git clone https://github.com/yashjani1997/SmartCine.git
cd SmartCine

# Install dependencies
pip install -r requirements.txt

# Add your TMDB API Key in .streamlit/secrets.toml
# TMDB_API_KEY = "your_api_key_here"

# Run the app
streamlit run app.py
```

---

## 🧠 Key Learnings

- Feature engineering for recommender systems
- Hybrid recommendation design (ML + API)
- Performance optimization using KMeans & SVD
- ML + UI integration with Streamlit
- Designing ML systems with **API & scalability mindset**
- Handling cold-start problem using TMDB API fallback

---

## 🎯 Project Highlights

- ✅ Trained on TMDB 5000 dataset (V1 baseline)
- ✅ SVD + KMeans optimization (V2)
- ✅ Full TMDB API integration — live posters, trailers, trending & search (V3)
- ✅ Hybrid search: ML model + API fallback for new movies
- ✅ Live deployed on Streamlit Cloud

---

## 🔗 Live Demo

👉 [SmartCine Live App](https://smartcine-yflwioje8c2ggv8lqzemqu.streamlit.app/)

---

## 👤 Author

**Yash Jani**  
Data Analyst & Machine Learning Enthusiast  
[GitHub: yashjani1997](https://github.com/yashjani1997)
