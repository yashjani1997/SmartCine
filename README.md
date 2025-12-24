# 🎬 SmartCine – Hybrid Movie Recommendation System

SmartCine is a **version-wise evolved movie recommendation system** built using Machine Learning.  
The project demonstrates how a core recommendation engine can be gradually upgraded into a **full-featured, scalable movie discovery platform**.

---

## 🚀 Project Overview

SmartCine recommends movies using a **hybrid approach** that combines:
- Content-based filtering
- Unsupervised learning (KMeans clustering)
- Cosine similarity for text-based similarity

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
- Text vectorization and cosine similarity  
- KMeans clustering to reduce search space  
- Hybrid recommendation (cluster + similarity)  

**Why this approach?**  
Clustering improves scalability, while cosine similarity provides accurate content-based recommendations.

---

### 🔁 Version 2 (V2) – Enhanced Logic & Flexibility
**Focus:** Smarter and optimized recommendations  

- Weighted hybrid scoring (cluster + similarity)  
- Dimensionality reduction (SVD) for faster computation  
- Multiple recommendation paths:
  - Movie-based
  - Genre-based
  - Actor & Director-based  

**Why upgrade?**  
To improve performance, flexibility, and user exploration beyond a single recommendation flow.

---

### 🚀 Version 3 (V3) – Full Movie Discovery Platform
**Focus:** Product-level & scalable system  

- Advanced Streamlit UI with tabs  
- TMDB movie poster integration  
- Trending, top-rated, filters & curated collections  
- Interactive visualizations (Plotly)  

#### 🔌 API Integration (Planned / Ongoing)
- Integration with **TMDB API** for:
  - Real-time movie posters
  - Latest movie metadata
  - Dynamic updates instead of static datasets
- Exposing recommendation logic as an **API service** for:
  - Web / mobile app usage
  - Future microservice-based deployment

**Why V3?**  
To transform the ML model into a real-world application with **live data, scalability, and production readiness**.

---

## 🛠 Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn (KMeans, Cosine Similarity, SVD)
- Streamlit
- Plotly
- TMDB Dataset & TMDB API (planned)

---

## 🧠 Key Learnings
- Feature engineering for recommender systems  
- Hybrid recommendation design  
- Performance optimization using clustering & SVD  
- ML + UI integration  
- Designing ML systems with **API & scalability mindset**  

---

## 👤 Author
**Jani**  
Data Analyst & Machine Learning Enthusiast  

---
