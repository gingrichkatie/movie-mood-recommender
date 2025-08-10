# CIS 9660 - Project Q2
# Movie Mood Recommender using TMDB + OpenAI
# ==============================================================

import os
import streamlit as st
import requests
from openai import OpenAI
import pandas as pd

# --- Load API keys ---
# Note: Keys must be set in Streamlit Secrets or as environment variables
TMDB_API_KEY = st.secrets.get("TMDB_API_KEY", os.getenv("TMDB_API_KEY", ""))
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))

# --- Key validation ---
if not TMDB_API_KEY or not OPENAI_API_KEY:
    st.error("Missing API keys. Please set TMDB_API_KEY and OPENAI_API_KEY in Streamlit secrets.")
    st.stop()

# --- Initialize OpenAI client ---
client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://api.openai.com/v1")

# --- TMDB fetch helper ---
def fetch_movies_by_genre(genre_id=35, pages=1):
    """Fetch movies from TMDB for a given genre."""
    movies = []
    for page in range(1, pages + 1):
        url = "https://api.themoviedb.org/3/discover/movie"
        params = {
            "api_key": TMDB_API_KEY,
            "language": "en-US",
            "sort_by": "popularity.desc",
            "include_adult": False,
            "with_genres": genre_id,
            "page": page
        }
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        movies.extend(r.json().get("results", []))
    return movies

# --- AI ranking helper ---
def ai_rank_movies(mood, movie_list):
    """Use OpenAI to pick the top 5 matches for a given mood."""
    titles = [m["title"] for m in movie_list[:15]]  # Keep list short for faster prompts
    prompt = f"""
    You are a movie recommendation assistant.
    The user is in the mood for: {mood}.
    Here is a list of movies: {titles}.
    Pick the 5 best matches for that mood and give a one-sentence reason for each.
    Return as JSON with keys: title, reason.
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return resp.choices[0].message.content

# --- Streamlit page setup ---
st.set_page_config(page_title="Movie Mood Recommender", layout="wide")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["Recommendation Tool", "Movie Data Explorer", "About"])


# Tab 1: Recommendation Tool

with tab1:
    st.header("Mood-Based Movie Recommendations")
    st.markdown("Enter your mood and pick a genre to receive AI-curated suggestions.")

    mood = st.text_input("Describe your mood:", "cozy and heartwarming")
    genre_map = {
        "Comedy": 35, "Drama": 18, "Action": 28, "Romance": 10749, "Horror": 27
    }
    genre = st.selectbox("Choose a base genre", list(genre_map.keys()))

    if st.button("Get Recommendations"):
        try:
            with st.spinner("Fetching movies from TMDB..."):
                movies = fetch_movies_by_genre(genre_map[genre], pages=1)
            with st.spinner("Generating AI recommendations..."):
                ai_output = ai_rank_movies(mood, movies)

            st.subheader("AI-Selected Top Picks")
            st.write(ai_output)

        except Exception as e:
            st.error(f"Error: {e}")


# Tab 2: Movie Data Explorer

with tab2:
    st.header("Explore TMDB Movie Data")
    st.markdown("View the raw movie data returned by TMDB for your selected genre.")

    genre_exp = st.selectbox("Select a genre to explore", list(genre_map.keys()), key="explorer_genre")
    if st.button("Load Data"):
        try:
            with st.spinner("Loading movie data..."):
                movies = fetch_movies_by_genre(genre_map[genre_exp], pages=1)
            df = pd.DataFrame(movies)
            if not df.empty:
                st.dataframe(df[["title", "release_date", "vote_average", "vote_count"]])
            else:
                st.warning("No data found for that genre.")
        except Exception as e:
            st.error(f"Error: {e}")


# Tab 3: About

with tab3:
    st.header("About This App")
    st.markdown("""
    **Purpose**  
    This app demonstrates integrating two APIs — The Movie Database (TMDB) for data retrieval  
    and OpenAI for natural language processing — to build an interactive recommendation tool.

    **How It Works**  
    1. The user specifies a mood and a base genre.  
    2. The app calls the TMDB API to get a list of popular movies in that genre.  
    3. The list is passed to OpenAI, which ranks and explains the top 5 matches for the mood.  
    4. Users can also explore the raw TMDB data in the Data Explorer tab.

    **Limitations**  
    - Recommendations are based on popularity, not personal viewing history.  
    - OpenAI responses depend on the prompt and may vary between runs.  
    - TMDB data changes over time as movie popularity updates.

    **APIs Used**  
    - TMDB API: [https://developer.themoviedb.org/](https://developer.themoviedb.org/)  
    - OpenAI API: [https://platform.openai.com/](https://platform.openai.com/)
    """)
