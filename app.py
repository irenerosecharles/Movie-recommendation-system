import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import json
import os
from datetime import datetime
from rapidfuzz import process, fuzz

warnings.filterwarnings("ignore")

# Page config
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# --- CONFIG ---
DATASET_PATH = "trimmed_dataset_movies.xlsx"
USER_PROFILE_PATH = "user_profile_streamlit.json"

# --- CACHE DATA LOADING (Only loads once!) ---
@st.cache_data
def load_data():
    """Load and preprocess dataset - cached for performance"""
    df = pd.read_excel(DATASET_PATH)
    
    # Ensure required text columns exist
    text_cols = ['genres','keywords','overview','production_companies','spoken_languages']
    for c in text_cols:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("").astype(str)
    
    # Build metadata field
    df['metadata'] = (
        df['genres'].str.replace('[^A-Za-z0-9, ]', ' ', regex=True).str.lower() + " " +
        df['keywords'].str.replace('[^A-Za-z0-9, ]', ' ', regex=True).str.lower() + " " +
        df['overview'].str.replace('[^A-Za-z0-9 ]', ' ', regex=True).str.lower() + " " +
        df['production_companies'].str.replace('[^A-Za-z0-9, ]', ' ', regex=True).str.lower() + " " +
        df['spoken_languages'].str.replace('[^A-Za-z0-9, ]', ' ', regex=True).str.lower()
    )
    
    df['metadata'] = df['metadata'].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    return df

@st.cache_data
def compute_similarity(_df):
    """Compute TF-IDF and cosine similarity - cached for performance"""
    vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(_df['metadata'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

# Load data
df = load_data()
cosine_sim = compute_similarity(df)

# Title lookup
titles_list = df['title'].astype(str).tolist()
title_to_index = pd.Series(df.index, index=df['title'].astype(str)).to_dict()

# --- USER PROFILE MANAGEMENT ---
class UserProfile:
    """Manages user's watched movies and ratings"""
    
    def __init__(self, filepath=USER_PROFILE_PATH):
        self.filepath = filepath
        self.profile = self.load_profile()
    
    def load_profile(self):
        if os.path.exists(self.filepath):
            with open(self.filepath, 'r') as f:
                return json.load(f)
        return {"watched": {}, "created": datetime.now().isoformat()}
    
    def save_profile(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.profile, f, indent=2)
    
    def add_rating(self, title, rating):
        self.profile["watched"][title] = {
            "rating": rating,
            "timestamp": datetime.now().isoformat()
        }
        self.save_profile()
    
    def get_rating(self, title):
        return self.profile["watched"].get(title, {}).get("rating")
    
    def has_watched(self, title):
        return title in self.profile["watched"]
    
    def get_liked_movies(self, min_rating=4.0):
        return [title for title, data in self.profile["watched"].items() 
                if data["rating"] >= min_rating]
    
    def get_watch_history(self):
        return self.profile["watched"]
    
    def clear_profile(self):
        self.profile = {"watched": {}, "created": datetime.now().isoformat()}
        self.save_profile()

# Initialize session state
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = UserProfile()

if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None

if 'matched_title' not in st.session_state:
    st.session_state.matched_title = None

# --- HELPER FUNCTIONS ---
def closest_title(query, titles=titles_list, min_score=70):
    """Uses RapidFuzz to find a reliable best match"""
    query = query.strip().lower()
    
    # exact match
    for t in titles:
        if t.lower() == query:
            return t
    
    # fuzzy match
    match = process.extractOne(query, titles, scorer=fuzz.token_sort_ratio)
    if not match:
        return None
    
    candidate, score, _ = match
    
    if len(query) <= 4:
        return candidate if score >= 85 else None
    
    return candidate if score >= min_score else None

def get_user_preference_boost(user_liked_movies, query_idx, boost_factor=0.3):
    """Boost scores for movies similar to user's liked movies"""
    if not user_liked_movies:
        return np.zeros(len(df))
    
    boost_scores = np.zeros(len(df))
    
    for liked_title in user_liked_movies:
        if liked_title not in title_to_index:
            continue
        
        liked_idx = title_to_index[liked_title]
        similarities = cosine_sim[liked_idx]
        boost_scores += similarities * boost_factor
    
    boost_scores /= len(user_liked_movies)
    return boost_scores

def get_recommendations(query_title, top_k=10):
    """Get movie recommendations"""
    matched = closest_title(query_title)
    if not matched:
        return None, None
    
    idx = title_to_index.get(matched)
    if idx is None:
        return None, None
    
    # Base similarity scores
    sim_scores = cosine_sim[idx].copy()
    
    # Apply user preference boost
    liked_movies = st.session_state.user_profile.get_liked_movies(min_rating=4.0)
    if liked_movies:
        boost = get_user_preference_boost(liked_movies, idx)
        sim_scores = sim_scores + boost
    
    # Create list of (index, score) tuples
    sim_scores_list = list(enumerate(sim_scores))
    sim_scores_list = sorted(sim_scores_list, key=lambda x: x[1], reverse=True)
    
    # Get top recommendations
    results = []
    count = 0
    
    for i, score in sim_scores_list:
        if count >= top_k:
            break
        if i == idx:
            continue
        
        row = df.iloc[i]
        
        # Extract year
        year = None
        try:
            if pd.notnull(row['release_date']):
                year = int(pd.to_datetime(row['release_date']).year)
        except:
            pass
        
        movie_title = row['title']
        watched = st.session_state.user_profile.has_watched(movie_title)
        user_rating = st.session_state.user_profile.get_rating(movie_title)
        
        results.append({
            "title": movie_title,
            "year": year,
            "score": float(score),
            "vote_average": float(row['vote_average']) if pd.notnull(row.get('vote_average')) else None,
            "vote_count": int(row['vote_count']) if pd.notnull(row.get('vote_count')) else None,
            "watched": watched,
            "user_rating": user_rating
        })
        
        count += 1
    
    return matched, results

# --- STREAMLIT UI ---
st.title("🎬 Personalized Movie Recommender")
st.markdown("Find movies similar to your favorites!")

# Sidebar - Watch History
with st.sidebar:
    st.header("📊 Your Watch History")
    
    history = st.session_state.user_profile.get_watch_history()
    
    if history:
        st.success(f"You've rated {len(history)} movies")
        
        # Show top rated
        sorted_movies = sorted(history.items(), key=lambda x: x[1]['rating'], reverse=True)
        
        st.markdown("### Top Rated:")
        for title, data in sorted_movies[:5]:
            rating = data['rating']
            stars = "⭐" * int(rating)
            st.markdown(f"{stars} **{rating:.1f}** - {title}")
        
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.user_profile.clear_profile()
            st.rerun()
    else:
        st.info("No ratings yet. Start by searching for a movie!")
    
    st.markdown("---")
    st.markdown(f"**Dataset:** {len(df)} movies")

# Main content
col1, col2 = st.columns([3, 1])

with col1:
    movie_query = st.text_input(
        "🔍 Enter a movie title:",
        placeholder="e.g., Inception, Batman, Zodiac...",
        key="movie_input"
    )

with col2:
    search_button = st.button("🎯 Find Similar", use_container_width=True, type="primary")

# Search logic
if search_button and movie_query:
    with st.spinner("Searching..."):
        matched, recommendations = get_recommendations(movie_query, top_k=10)
        
        if matched is None:
            st.error(f"❌ No confident match found for '{movie_query}'")
        else:
            st.session_state.matched_title = matched
            st.session_state.recommendations = recommendations
            
            # Check for personalization
            liked_movies = st.session_state.user_profile.get_liked_movies(min_rating=4.0)
            if liked_movies:
                st.info(f"✨ Personalized based on {len(liked_movies)} movies you liked!")

# Display recommendations
if st.session_state.recommendations:
    st.success(f"🎯 Found: **{st.session_state.matched_title}**")
    st.markdown("---")
    st.subheader("📽️ Recommended Movies:")
    
    # Display in a nice format
    for i, movie in enumerate(st.session_state.recommendations, 1):
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            year_str = f"({movie['year']})" if movie['year'] else ""
            title_display = f"**{i}. {movie['title']}** {year_str}"
            
            if movie['watched']:
                st.markdown(f"{title_display} ✅ *Rated: {movie['user_rating']:.1f}⭐*")
            else:
                st.markdown(title_display)
        
        with col2:
            if movie['vote_average']:
                st.caption(f"⭐ {movie['vote_average']:.1f}/10")
        
        with col3:
            st.caption(f"Score: {movie['score']:.2f}")
    
    # Rating section
    st.markdown("---")
    st.subheader("⭐ Rate Movies You've Watched")
    
    # Multi-select for movies
    movie_options = [f"{i+1}. {m['title']}" for i, m in enumerate(st.session_state.recommendations)]
    selected = st.multiselect(
        "Select movies you've watched:",
        options=movie_options,
        help="Select one or more movies to rate"
    )
    
    if selected:
        st.markdown("### Rate Your Selections:")
        
        for selection in selected:
            # Extract movie index
            idx = int(selection.split('.')[0]) - 1
            movie = st.session_state.recommendations[idx]
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**{movie['title']}**")
            
            with col2:
                rating = st.slider(
                    "Rating",
                    min_value=1.0,
                    max_value=5.0,
                    value=movie['user_rating'] if movie['user_rating'] else 3.0,
                    step=0.5,
                    key=f"rating_{movie['title']}",
                    label_visibility="collapsed"
                )
            
            # Auto-save rating
            if rating != movie['user_rating']:
                st.session_state.user_profile.add_rating(movie['title'], rating)
                movie['user_rating'] = rating
                movie['watched'] = True
        
        if st.button("💾 Save All Ratings", type="primary", use_container_width=True):
            st.success("✅ All ratings saved! Search again for personalized recommendations.")
            st.balloons()

# Footer
st.markdown("---")
st.caption("💡 Tip: Rate more movies to get better personalized recommendations!")
