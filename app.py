# AI-Based Post Recommender System

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# UI Setup
st.set_page_config(page_title="Post Recommender", layout="wide")
st.title("📬 AI-Based Post Recommender System")

st.markdown("""
<style>
.big-font {font-size:20px !important; font-weight:600;}
.small-font {font-size:14px !important; color:gray;}
.recommendation-box {
    border: 1px solid #444;
    border-radius: 10px;
    padding: 10px;
    background-color: #1e1e1e;
    margin-bottom: 10px;
    color: #f1f1f1;
}
input, .stSelectbox > div > div {
    background-color: #262730 !important;
    color: #f1f1f1 !important;
}
</style>
""", unsafe_allow_html=True)

# Load and preprocess data
@st.cache_data
def load_and_prepare_data():
    posts = pd.read_csv("posts.csv")
    users = pd.read_csv("users.csv")
    views = pd.read_csv("views.csv")

    posts['category'] = posts['category'].fillna('random')

    # Expand multi-category rows
    post_expanded = []
    for _, row in posts.iterrows():
        for cat in row['category'].split('|'):
            post_expanded.append({
                'post_id': row['_id'],
                'title': row['title'],
                'category': cat.strip(),
                'post_type': row[' post_type']
            })
    post1 = pd.DataFrame(post_expanded)

    main = pd.merge(views, post1, on='post_id')
    return posts, post1, main

posts_df, post1, main = load_and_prepare_data()
users = main['user_id'].unique().tolist()
categories = main['category'].unique().tolist()
post_types = post1['post_type'].unique().tolist()
posts = main['post_id'].unique().tolist()

# User-category matrix for collaborative filtering
@st.cache_data
def get_user_matrix():
    user_idx_map = {uid: idx for idx, uid in enumerate(users)}
    cat_idx_map = {cat: idx for idx, cat in enumerate(categories)}

    user_cat_mat = np.zeros((len(users), len(categories)))
    for _, row in main.iterrows():
        u_idx = user_idx_map[row['user_id']]
        c_idx = cat_idx_map[row['category']]
        user_cat_mat[u_idx][c_idx] += 1

    return csr_matrix(user_cat_mat), user_idx_map, cat_idx_map

user_mat, user_idx_map, cat_idx_map = get_user_matrix()
model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10)
model.fit(user_mat)

# Build content profiles
@st.cache_data
def build_profiles():
    post_idx_map = {pid: idx for idx, pid in enumerate(posts)}

    item_profiles = {
        cat: np.array([
            1.0 if cat in main[main['post_id'] == pid]['category'].values else 0.0
            for pid in posts
        ]) for cat in categories
    }

    user_profiles = {}
    for uid in users:
        user_data = main[main['user_id'] == uid]
        cat_weights = user_data['category'].value_counts(normalize=True).to_dict()
        profile = np.zeros(len(posts))
        for cat, weight in cat_weights.items():
            profile += weight * item_profiles[cat]
        user_profiles[uid] = profile
    return item_profiles, user_profiles

item_profiles, user_profiles = build_profiles()

# Collaborative Filtering
@st.cache_data
def recommender_collab(user_id):
    idx = user_idx_map[user_id]
    user_data = main[main['user_id'] == user_id]
    _, indices = model.kneighbors(user_mat[idx])
    recs = set()
    for i in indices[0]:
        peer_data = main[main['user_id'] == users[i]]
        for cat in peer_data['category'].unique():
            if cat not in user_data['category'].unique():
                recs.add(cat)
    return list(recs)[:10]

# Content-Based Filtering
@st.cache_data
def recommender_content(user_id):
    user_profile = user_profiles[user_id]
    similarity = {
        cat: cosine_similarity(user_profile.reshape(1, -1), item_profiles[cat].reshape(1, -1))[0][0]
        for cat in item_profiles
    }
    sorted_sim = sorted(similarity.items(), key=lambda x: x[1], reverse=True)
    viewed_posts = set(main[main['user_id'] == user_id]['post_id'])
    recs = []
    for cat, _ in sorted_sim:
        for pid in main[main['category'] == cat]['post_id'].unique():
            if pid not in viewed_posts:
                recs.append((cat, pid))
            if len(recs) == 10:
                return recs
    return recs

# Streamlit UI
st.subheader("🎯 Select Preferences")
user_id = st.selectbox("👤 Select User ID", users)

with st.expander("🔎 Filter Posts"):
    post_type_filter = st.multiselect("Post Types", post_types, default=post_types)
    category_filter = st.multiselect("Categories", categories, default=categories)

if st.button("🚀 Get Recommendations"):
    with st.spinner("Finding best recommendations..."):
        colab_recs = recommender_collab(user_id)
        content_recs = recommender_content(user_id)

        st.subheader("🤝 Collaborative Filtering Recommendations")
        for cat in colab_recs:
            if cat in category_filter:
                st.markdown(f"<div class='recommendation-box'><span class='big-font'>{cat}</span><br><span class='small-font'>Suggested Category</span></div>", unsafe_allow_html=True)

        st.subheader("📚 Content-Based Filtering Recommendations")
        for cat, pid in content_recs:
            post_row = posts_df[posts_df['_id'] == pid]
            if not post_row.empty:
                title = post_row['title'].values[0]
                ptype = post_row[' post_type'].values[0]
                if cat in category_filter and ptype in post_type_filter:
                    st.markdown(f"""
                    <div class='recommendation-box'>
                        <span class='big-font'>{title}</span><br>
                        <span class='small-font'>Post ID: {pid} | Category: {cat} | Type: {ptype}</span>
                    </div>
                    """, unsafe_allow_html=True)
