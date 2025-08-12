# ==============================================================
# 🎬 CIS 9660 - Project Q2
# Movie Mood Recommender (TMDB + OpenAI) 
# ==============================================================

import os
import json
import re
import requests
import pandas as pd
import streamlit as st
from datetime import datetime
from openai import OpenAI

# -----------------------------
# Page + Global Styling
# -----------------------------
st.set_page_config(
    page_title="🎬 Movie Mood Recommender",
    page_icon="🍿",
    layout="wide"
)

# --- Tiny CSS makeover (subtle, readable, fun) ---
st.markdown(
    """
    <style>
      .app-hero {
        background: radial-gradient(1200px 400px at 10% -20%, #ffe7a3 0%, rgba(255,231,163,0) 60%),
                    radial-gradient(1200px 400px at 90% -20%, #a3d8ff 0%, rgba(163,216,255,0) 60%);
        border-radius: 20px;
        padding: 24px 28px;
        border: 1px solid rgba(0,0,0,0.06);
        margin-bottom: 16px;
      }
      .app-hero h1 { margin: 0 0 8px 0; font-size: 2.1rem; }
      .app-hero p { margin: 0; font-size: 1.05rem; opacity: 0.85; }

      .movie-card {
        border-radius: 18px;
        padding: 14px;
        border: 1px solid rgba(0,0,0,0.08);
        background: #ffffffaa;
        backdrop-filter: blur(6px);
      }
      .movie-title {
        font-weight: 700;
        font-size: 1.05rem;
        margin: 2px 0 0 0;
      }
      .movie-meta {
        font-size: 0.92rem;
        opacity: 0.8;
      }
      .reason {
        font-size: 0.95rem;
        margin-top: 6px;
      }
      .chip {
        display:inline-block;
        font-size: 0.80rem;
        padding: 2px 8px;
        margin-right: 6px;
        border-radius: 999px;
        border: 1px solid rgba(0,0,0,0.08);
        background: #f7f7f7;
      }
      /* Make button rows breathe */
      .stButton>button { width: 100%; border-radius: 12px; }
      /* Dataframe tweaks */
      div[data-testid="stDataFrame"] { border-radius: 12px; overflow:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Secrets / API Keys
# -----------------------------
TMDB_API_KEY = st.secrets.get("TMDB_API_KEY", os.getenv("TMDB_API_KEY", ""))
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))

if not TMDB_API_KEY or not OPENAI_API_KEY:
    st.error("🔐 Missing API keys. Please set TMDB_API_KEY and OPENAI_API_KEY in Streamlit Secrets or env vars.")
    st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# Helpers: TMDB
# -----------------------------
TMDB_IMG_BASE = "https://image.tmdb.org/t/p"
def img_url(path, size="w500"):
    if not path:
        return None
    return f"{TMDB_IMG_BASE}/{size}{path}"

@st.cache_data(show_spinner=False, ttl=60*10)
def fetch_movies_by_genre(genre_id=35, pages=1, language="en-US"):
    """Fetch popular movies from TMDB for a given genre, multiple pages."""
    movies = []
    for page in range(1, pages + 1):
        url = "https://api.themoviedb.org/3/discover/movie"
        params = {
            "api_key": TMDB_API_KEY,
            "language": language,
            "sort_by": "popularity.desc",
            "include_adult": False,
            "with_genres": genre_id,
            "page": page
        }
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        payload = r.json().get("results", [])
        movies.extend(payload)
    return movies

@st.cache_data(show_spinner=False, ttl=60*10)
def fetch_trailer_key(movie_id):
    """Get the first YouTube trailer key for a movie (if available)."""
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos"
        params = {"api_key": TMDB_API_KEY, "language": "en-US"}
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        results = r.json().get("results", [])
        # Prefer official trailers
        for v in results:
            if v.get("site") == "YouTube" and "Trailer" in v.get("type", ""):
                return v.get("key")
        # Fallback: any YouTube video
        for v in results:
            if v.get("site") == "YouTube":
                return v.get("key")
    except Exception:
        pass
    return None

# -----------------------------
# Helpers: OpenAI
# -----------------------------
def _extract_json_block(text):
    """Extract JSON array/dict from a model response that might include extra text or code fences."""
    if not text:
        return None
    # Try code-fence first
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence:
        return fence.group(1).strip()
    # Then try to find first [ ... ] or { ... }
    bracket = re.search(r"(\{.*\}|\[.*\])", text, re.S)
    if bracket:
        return bracket.group(1).strip()
    return text.strip()

def ai_rank_movies(mood, movie_list):
    """
    Use OpenAI to pick the top 5 matches for a given mood.
    Returns a list of dicts: [{title, reason}]
    """
    titles = [m.get("title") for m in movie_list[:18] if m.get("title")]  # keep it short-ish
    system = "You are a concise movie recommendation assistant."
    user = f"""
Pick the 5 best movies for the mood: "{mood}" from this list of titles:
{titles}

Return ONLY JSON as a list of objects with keys: "title" and "reason".
Example:
[
  {{ "title": "Movie A", "reason": "Why it fits the mood in one sentence." }},
  ...
]
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.6,
    )
    raw = resp.choices[0].message.content
    block = _extract_json_block(raw)
    try:
        data = json.loads(block)
        if isinstance(data, dict):  # rare case
            data = [data]
        cleaned = []
        for d in data:
            t = (d.get("title") or "").strip()
            r = (d.get("reason") or "").strip()
            if t:
                cleaned.append({"title": t, "reason": r})
        return cleaned[:5]
    except Exception:
        # Fallback: return the raw text so the user still sees something
        return [{"title": "Model Output", "reason": raw}]

# -----------------------------
# UI: Sidebar
# -----------------------------
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1485846234645-a62644f84728?q=80&w=1200&auto=format&fit=crop", use_column_width=True)
    st.markdown("### 🍿 Quick Tips")
    st.caption("• Try moods like **“cozy and heartwarming”**, **“need a cathartic cry”**, **“adrenaline!”**, or **“weird and artsy”**.\n"
               "• Switch genres to tilt the vibe.\n"
               "• Click **Trailer** for a quick peek.")
    st.markdown("---")
    st.markdown("Made with ❤️ using TMDB + OpenAI")

# -----------------------------
# Hero / Banner
# -----------------------------
st.markdown(
    """
    <div class="app-hero">
      <h1>🍿 Movie Mood Recommender</h1>
      <p>Describe your vibe and we'll serve five perfect picks — with posters, trailers, and a dash of movie magic. 🎥✨</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["🎯 Recommendation Tool", "🔎 Movie Data Explorer", "ℹ️ About"])

# Genre mapping + emoji chips
GENRE_MAP = {
    "Comedy 😂": 35,
    "Drama 🎭": 18,
    "Action 💥": 28,
    "Romance 💘": 10749,
    "Horror 👻": 27,
    "Sci-Fi 🚀": 878,
    "Animation 🐭": 16,
    "Thriller 🕵️": 53,
}

def as_stars(vote):
    if vote is None:
        return "—"
    # TMDB vote_average is 0–10; convert to 5 stars
    stars = round(float(vote) / 2, 1)
    full = int(stars)
    half = 1 if stars - full >= 0.5 else 0
    return "★" * full + ("½" if half else "") + "☆" * (5 - full - half)

def render_movie_card(movie, reason=None):
    poster = img_url(movie.get("poster_path"), "w342")
    title = movie.get("title") or "Untitled"
    year = (movie.get("release_date") or "")[:4]
    rating = movie.get("vote_average")
    votes = movie.get("vote_count")
    tmdb_id = movie.get("id")

    with st.container(border=False):
        col1, col2 = st.columns([1, 2], vertical_alignment="center")
        with col1:
            if poster:
                st.image(poster, use_column_width=True)
            else:
                st.markdown("🎬 *(No poster)*")
        with col2:
            st.markdown('<div class="movie-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="movie-title">{title} {f"({year})" if year else ""}</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="movie-meta">{" ".join([f"<span class=\\"chip\\">⭐ {rating:.1f}</span>" if rating else "", f"<span class=\\"chip\\">🗳️ {votes or 0} votes</span>"])} 
                f'{f"<span class=\\"chip\\">{as_stars(rating)}</span>" if rating else ""}</div>',
                unsafe_allow_html=True
            )
            if reason:
                st.markdown(f'<div class="reason">💡 {reason}</div>', unsafe_allow_html=True)

            # Buttons row
            c1, c2 = st.columns(2)
            tmdb_link = f"https://www.themoviedb.org/movie/{tmdb_id}" if tmdb_id else None
            trailer_key = fetch_trailer_key(tmdb_id) if tmdb_id else None

            with c1:
                if tmdb_link:
                    st.link_button("TMDB Page 🔗", tmdb_link, use_container_width=True)
                else:
                    st.button("TMDB Page 🔗", disabled=True, use_container_width=True)
            with c2:
                if trailer_key:
                    st.link_button("Trailer ▶️", f"https://www.youtube.com/watch?v={trailer_key}", use_container_width=True)
                else:
                    st.button("Trailer ▶️", disabled=True, use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# Tab 1: Recommendation Tool
# =========================================================
with tab1:
    st.subheader("🎯 Mood-Based Movie Recommendations")
    st.caption("Describe your mood and pick a genre. We’ll blend your vibe with what’s trending to find five gems.")

    colA, colB = st.columns([2, 1])
    with colA:
        mood = st.text_input("How are you feeling? (e.g., “cozy and heartwarming”, “need an adrenaline rush”)", "cozy and heartwarming")
    with colB:
        genre_label = st.selectbox("Choose a base genre", list(GENRE_MAP.keys()))
        genre_id = GENRE_MAP[genre_label]

    go = st.button("🎬 Get Recommendations", type="primary")
    if go:
        try:
            st.toast("🍿 Grabbing popular titles from TMDB…", icon="🍿")
            movies = fetch_movies_by_genre(genre_id=genre_id, pages=2)

            if not movies:
                st.warning("No movies found. Try a different genre.")
            else:
                st.toast("🧠 Asking the AI for mood-fit picks…", icon="🧠")
                picks = ai_rank_movies(mood, movies)

                # Match AI titles back to TMDB metadata (best-effort fuzzy title match)
                st.subheader("✨ AI-Selected Top Picks")
                matched = []
                titles_lower = {m.get("title","").lower(): m for m in movies}
                for p in picks:
                    t = (p.get("title") or "").lower().strip()
                    m = titles_lower.get(t)
                    if not m:
                        # loose match: remove punctuation & compare
                        def norm(s): return re.sub(r"[^a-z0-9 ]","", s.lower())
                        norm_map = {norm(mv.get("title","")): mv for mv in movies}
                        m = norm_map.get(norm(t))
                    if m:
                        matched.append((m, p.get("reason")))
                    else:
                        # Fallback “unknown” card
                        matched.append(({
                            "title": p.get("title") or "Model Pick",
                            "release_date": "",
                            "poster_path": None,
                            "vote_average": None,
                            "vote_count": None,
                            "id": None
                        }, p.get("reason")))

                # Display cards in a responsive grid (2 columns)
                cols = st.columns(2)
                for i, (movie, reason) in enumerate(matched):
                    with cols[i % 2]:
                        render_movie_card(movie, reason)

        except Exception as e:
            st.error(f"Oops — something went wrong: {e}")

# =========================================================
# Tab 2: Movie Data Explorer
# =========================================================
with tab2:
    st.subheader("🔎 Explore TMDB Movie Data")
    st.caption("Peek at the raw TMDB fields for the selected genre. Great for sanity-checks and curiosity.")

    col1, col2, col3 = st.columns([1.4, 1, 1])
    with col1:
        genre_exp_label = st.selectbox("Genre", list(GENRE_MAP.keys()), key="explorer_genre")
        genre_exp = GENRE_MAP[genre_exp_label]
    with col2:
        pages = st.slider("Pages", 1, 5, 1, help="Each page ~20 results (sorted by popularity).")
    with col3:
        show_posters = st.toggle("Show Posters", value=True)

    if st.button("📥 Load Data"):
        try:
            movies = fetch_movies_by_genre(genre_exp, pages=pages)
            if not movies:
                st.warning("No data found for that genre.")
            else:
                df = pd.DataFrame(movies)
                # Curate a friendly subset
                view_cols = ["title", "release_date", "vote_average", "vote_count", "popularity"]
                view_cols = [c for c in view_cols if c in df.columns]
                if show_posters:
                    df["poster"] = df["poster_path"].apply(lambda p: img_url(p, "w92"))
                    view_cols = ["poster"] + view_cols
                # Clean dates
                if "release_date" in df:
                    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce").dt.date
                st.dataframe(df[view_cols], use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Error loading data: {e}")

# =========================================================
# Tab 3: About
# =========================================================
with tab3:
    st.subheader("ℹ️ About This App")
    st.markdown(
        """
**What it does**  
Describe your mood → pick a genre → get 5 AI-curated picks with reasons, posters, and quick trailer links.

**How it works**  
1. TMDB API returns popular movies for the selected genre.  
2. OpenAI ranks them against your mood and explains the picks.  
3. We match those titles back to TMDB to show posters, ratings, and links.

**Notes & Limits**  
- Popularity ≠ personal history — it won’t know what you’ve seen (yet!).  
- AI outputs can vary a bit between runs.  
- TMDB data updates over time.

**APIs Used**  
- TMDB: https://developer.themoviedb.org/  
- OpenAI: https://platform.openai.com/
"""
    )

# End of file ✂️
