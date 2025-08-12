# ==============================================================
# 🎬 CIS 9660 - Project Q2
# Movie Mood Recommender 
# ==============================================================

import os, re, json, requests, pandas as pd, streamlit as st
from openai import OpenAI

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="🎬 Movie Mood Recommender", page_icon="🍿", layout="wide")

# -----------------------------
# Styles: keep it light, crisp, readable
# -----------------------------
st.markdown("""
<style>
:root{
  --text:#111827;         /* main text */
  --muted:#374151;        /* secondary text */
  --panel:#ffffff;        /* card/panel bg */
  --line:rgba(0,0,0,.08); /* borders */
}

/* Clean, light app background with soft vignette */
.stApp{
  background:
    radial-gradient(1200px 420px at 10% -18%, #fff3bf 0%, rgba(255,243,191,0) 60%),
    radial-gradient(1200px 420px at 90% -18%, #cfe8ff 0%, rgba(207,232,255,0) 60%),
    #f7f8fb;
}

/* Global type colors (dark on light) */
h1,h2,h3,h4,strong { color: var(--text); }
p, label, .stMarkdown, .stCaption, span, small { color: var(--muted); }

/* Hero with subtle spotlight BEHIND the banner */
.app-hero{
  position:relative;
  border-radius:20px;
  padding:24px 28px;
  background: linear-gradient(180deg, #ffffff 0%, #f9fbff 100%);
  border:1px solid var(--line);
  box-shadow: 0 10px 24px rgba(16,24,40,.08);
  margin: 8px 0 16px 0;
}
.app-hero::before{
  content:"";
  position:absolute; inset:-22px; z-index:-1; border-radius:26px;
  background:
    radial-gradient(900px 260px at 12% -22%, rgba(255,233,165,.9) 0%, rgba(255,233,165,0) 65%),
    radial-gradient(900px 260px at 88% -22%, rgba(163,216,255,.85) 0%, rgba(163,216,255,0) 65%);
  filter: blur(2px);
}
.app-hero h1{
  display:inline-block;
  padding:10px 20px;
  border-radius:12px;
  font-weight:800;
  background: linear-gradient(180deg, #fff, #f5f7fb);
  box-shadow: inset 0 0 8px rgba(255,255,255,.75), 0 8px 20px rgba(0,0,0,.05);
}

/* Section headers + spacing */
.section-title{ margin-top:6px; }

/* Movie cards */
.movie-card{ border-radius:16px; padding:14px; border:1px solid var(--line); background: var(--panel); }
.movie-title{ font-weight:700; font-size:1.06rem; margin:2px 0 6px; color: var(--text); }
.movie-meta{ font-size:.92rem; color: var(--muted); }
.reason{ font-size:.95rem; margin-top:8px; color: var(--text); }
.chip{ display:inline-block; font-size:.80rem; padding:2px 8px; margin-right:6px; border-radius:999px; border:1px solid var(--line); background:#f4f7fb; color: var(--text); }

/* Buttons */
.stButton>button{ width:100%; border-radius:12px; font-weight:600; }

/* Dataframe */
div[data-testid="stDataFrame"]{ border-radius:12px; overflow:hidden; }

/* Images: use container width (no deprecation warnings) */
.stImage img, [data-testid="stSidebar"] img{ width:100% !important; height:auto !important; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Keys
# -----------------------------
TMDB_API_KEY = st.secrets.get("TMDB_API_KEY", os.getenv("TMDB_API_KEY", ""))
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
if not TMDB_API_KEY or not OPENAI_API_KEY:
    st.error("🔐 Missing API keys. Please set TMDB_API_KEY and OPENAI_API_KEY in Streamlit secrets or env vars.")
    st.stop()
client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# TMDB helpers
# -----------------------------
TMDB_IMG_BASE = "https://image.tmdb.org/t/p"
def img_url(path, size="w500"):
    return f"{TMDB_IMG_BASE}/{size}{path}" if path else None

@st.cache_data(ttl=600, show_spinner=False)
def fetch_movies_by_genre(genre_id=35, pages=1, language="en-US"):
    out=[]
    for page in range(1, pages+1):
        r = requests.get(
            "https://api.themoviedb.org/3/discover/movie",
            params=dict(api_key=TMDB_API_KEY, language=language, sort_by="popularity.desc",
                        include_adult=False, with_genres=genre_id, page=page),
            timeout=20
        )
        r.raise_for_status()
        out.extend(r.json().get("results", []))
    return out

@st.cache_data(ttl=600, show_spinner=False)
def fetch_trailer_key(movie_id):
    try:
        r = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}/videos",
                         params={"api_key": TMDB_API_KEY, "language":"en-US"}, timeout=15)
        r.raise_for_status()
        vids = r.json().get("results", [])
        for v in vids:
            if v.get("site")=="YouTube" and "Trailer" in (v.get("type") or ""): return v.get("key")
        for v in vids:
            if v.get("site")=="YouTube": return v.get("key")
    except Exception:
        pass
    return None

# -----------------------------
# OpenAI helper
# -----------------------------
def _extract_json_block(text:str):
    if not text: return None
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence: return fence.group(1).strip()
    bracket = re.search(r"(\{.*\}|\[.*\])", text, re.S)
    return bracket.group(1).strip() if bracket else text.strip()

def ai_rank_movies(mood, movie_list):
    titles = [m.get("title") for m in movie_list[:18] if m.get("title")]
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":"You are a concise movie recommendation assistant."},
            {"role":"user","content": f"""
Pick the 5 best movies for the mood: "{mood}" from this list:
{titles}

Return ONLY JSON list of objects with keys "title" and "reason".
"""}],
        temperature=0.6
    )
    raw = resp.choices[0].message.content
    block = _extract_json_block(raw)
    try:
        data = json.loads(block)
        if isinstance(data, dict): data=[data]
        cleaned=[{"title":(d.get("title") or "").strip(), "reason":(d.get("reason") or "").strip()}
                 for d in data if d.get("title")]
        return cleaned[:5]
    except Exception:
        return [{"title":"Model Output","reason":raw}]

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1485846234645-a62644f84728?q=80&w=1200&auto=format&fit=crop",
             use_container_width=True)
    st.markdown("### 🍿 Quick Tips")
    st.caption("• Try moods like **“cozy and heartwarming”**, **“need a cathartic cry”**, **“adrenaline!”**, or **“weird and artsy”**.\n"
               "• Switch genres to tilt the vibe.\n"
               "• Click **Trailer** for a quick peek.")
    st.markdown("---")
    st.markdown("Made with ❤️ using TMDB + OpenAI")

# -----------------------------
# Hero
# -----------------------------
st.markdown("""
<div class="app-hero">
  <h1>🍿 Movie Mood Recommender</h1>
  <p>Describe your vibe and we'll serve five perfect picks — with posters, trailers, and a dash of movie magic. 🎥✨</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["🎯 Recommendation Tool", "🔎 Movie Data Explorer", "ℹ️ About"])

GENRE_MAP = {
  "Comedy 😂":35, "Drama 🎭":18, "Action 💥":28, "Romance 💘":10749,
  "Horror 👻":27, "Sci-Fi 🚀":878, "Animation 🐭":16, "Thriller 🕵️":53
}

def as_stars(vote):
    if vote is None: return "—"
    stars = round(float(vote)/2, 1)
    full=int(stars); half=1 if stars-full>=0.5 else 0
    return "★"*full + ("½" if half else "") + "☆"*(5-full-half)

def render_movie_card(movie, reason=None):
    poster = img_url(movie.get("poster_path"), "w342")
    title = movie.get("title") or "Untitled"
    year  = (movie.get("release_date") or "")[:4]
    rating = movie.get("vote_average"); votes = movie.get("vote_count")
    tmdb_id = movie.get("id")

    with st.container(border=False):
        c1, c2 = st.columns([1,2], vertical_alignment="center")
        with c1:
            if poster: st.image(poster, use_container_width=True)
            else:      st.markdown("🎬 *(No poster)*")
        with c2:
            st.markdown('<div class="movie-card">', unsafe_allow_html=True)
            st.markdown(f"<div class='movie-title'>{title} {f'({year})' if year else ''}</div>", unsafe_allow_html=True)
            chips=[]
            if rating is not None: chips.append(f"<span class='chip'>⭐ {rating:.1f}</span>")
            chips.append(f"<span class='chip'>🗳️ {votes or 0} votes</span>")
            if rating is not None: chips.append(f"<span class='chip'>{as_stars(rating)}</span>")
            st.markdown(f"<div class='movie-meta'>{' '.join(chips)}</div>", unsafe_allow_html=True)
            if reason: st.markdown(f"<div class='reason'>💡 {reason}</div>", unsafe_allow_html=True)

            b1,b2 = st.columns(2)
            tmdb_link = f"https://www.themoviedb.org/movie/{tmdb_id}" if tmdb_id else None
            trailer_key = fetch_trailer_key(tmdb_id) if tmdb_id else None
            with b1:
                if tmdb_link:
                    try: st.link_button("TMDB Page 🔗", tmdb_link, use_container_width=True)
                    except: st.markdown(f"[TMDB Page 🔗]({tmdb_link})", unsafe_allow_html=True)
                else:
                    st.button("TMDB Page 🔗", disabled=True, use_container_width=True)
            with b2:
                if trailer_key:
                    try: st.link_button("Trailer ▶️", f"https://www.youtube.com/watch?v={trailer_key}", use_container_width=True)
                    except: st.markdown(f"[Trailer ▶️](https://www.youtube.com/watch?v={trailer_key})", unsafe_allow_html=True)
                else:
                    st.button("Trailer ▶️", disabled=True, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------
# Tab 1 — Recommendation Tool
# --------------------------------
with tab1:
    st.markdown("### 🎯 Mood-Based Movie Recommendations", unsafe_allow_html=True)
    st.caption("Describe your mood and pick a genre. We’ll blend your vibe with what’s trending to find five gems.")

    colA, colB = st.columns([2,1])
    with colA:
        mood = st.text_input("How are you feeling? (e.g., “cozy and heartwarming”, “need an adrenaline rush”)",
                             "cozy and heartwarming")
    with colB:
        genre_label = st.selectbox("Choose a base genre", list(GENRE_MAP.keys()))
        genre_id = GENRE_MAP[genre_label]

    if st.button("🎬 Get Recommendations", type="primary"):
        try:
            st.toast("🍿 Grabbing popular titles from TMDB…", icon="🍿")
            movies = fetch_movies_by_genre(genre_id=genre_id, pages=2)
            if not movies:
                st.warning("No movies found. Try a different genre.")
            else:
                st.toast("🧠 Asking the AI for mood-fit picks…", icon="🧠")
                picks = ai_rank_movies(mood, movies)

                matched=[]
                by_lower = {m.get("title","").lower(): m for m in movies}
                norm = lambda s: re.sub(r"[^a-z0-9 ]","", (s or "").lower())
                norm_map = {norm(m.get("title","")): m for m in movies}

                for p in picks:
                    t = (p.get("title") or "").strip()
                    m = by_lower.get(t.lower()) or norm_map.get(norm(t))
                    if not m:
                        m = {"title": t or "Model Pick", "release_date":"", "poster_path":None,
                             "vote_average":None, "vote_count":None, "id":None}
                    matched.append((m, p.get("reason")))

                cols = st.columns(2)
                for i,(mv,rsn) in enumerate(matched):
                    with cols[i%2]: render_movie_card(mv, rsn)
        except Exception as e:
            st.error(f"Oops — something went wrong: {e}")

# --------------------------------
# Tab 2 — Movie Data Explorer
# --------------------------------
with tab2:
    st.markdown("### 🔎 Explore TMDB Movie Data", unsafe_allow_html=True)
    st.caption("Peek at the raw TMDB fields for the selected genre. Great for sanity-checks and curiosity.")

    c1,c2,c3 = st.columns([1.4,1,1])
    with c1:
        genre_exp_label = st.selectbox("Genre", list(GENRE_MAP.keys()), key="explorer_genre")
        genre_exp = GENRE_MAP[genre_exp_label]
    with c2:
        pages = st.slider("Pages", 1, 5, 1, help="Each page ~20 results (sorted by popularity).")
    with c3:
        show_posters = st.toggle("Show Posters", value=True)

    if st.button("📥 Load Data"):
        try:
            movies = fetch_movies_by_genre(genre_exp, pages=pages)
            if not movies:
                st.warning("No data found for that genre.")
            else:
                df = pd.DataFrame(movies)
                view_cols = ["title","release_date","vote_average","vote_count","popularity"]
                view_cols = [c for c in view_cols if c in df.columns]
                if show_posters:
                    df["poster"] = df["poster_path"].apply(lambda p: img_url(p,"w92"))
                    view_cols = ["poster"] + view_cols
                if "release_date" in df:
                    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce").dt.date

                cfg={}
                if show_posters:
                    cfg["poster"] = st.column_config.ImageColumn("poster", width="small", help="TMDB poster")
                st.dataframe(df[view_cols], use_container_width=True, hide_index=True, column_config=cfg)
        except Exception as e:
            st.error(f"Error loading data: {e}")

# --------------------------------
# Tab 3 — About
# --------------------------------
with tab3:
    st.markdown("### ℹ️ About This App", unsafe_allow_html=True)
    st.markdown("""
**What it does**  
Describe your mood → pick a genre → get 5 AI-curated picks with reasons, posters, and quick trailer links.

**How it works**  
1) TMDB returns popular movies for the selected genre.  
2) OpenAI ranks them by your mood and explains why.  
3) We match titles back to TMDB to show posters, ratings, and links.
    """)






