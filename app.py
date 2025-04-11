# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# UI Setup
st.set_page_config(page_title="Post Recommender", layout="wide")
st.markdown(open("assets/style.css").read(), unsafe_allow_html=True)
st.title("📬 AI-Based Post Recommender System")

# Load data
@st.cache_data
def load_data():
    posts = pd.read_csv("posts.csv")
    users = pd.read_csv("users.csv")
    views = pd.read_csv("views.csv")
    return posts, users, views

post, user, view = load_data()

# Preprocessing
post['category'] = post['category'].fillna('random')
post1 = post.dropna(subset=['category']).copy()
post1 = post1.assign(category=post1['category'].str.split('|')).explode('category')
post1['category'] = post1['category'].str.strip()
main = pd.merge(view, post1, left_on='post_id', right_on='_id')

users = main['user_id'].unique().tolist()
categories = main['category'].unique().tolist()
posts = main['post_id'].unique().tolist()
post_lookup = post.set_index('_id')[['title', ' post_type', 'category']].to_dict('index')

# Create user-category matrix
@st.cache_data
def get_user_matrix():
    user_index = {uid: i for i, uid in enumerate(users)}
    cat_index = {cat: i for i, cat in enumerate(categories)}
    row_ind = main['user_id'].map(user_index)
    col_ind = main['category'].map(cat_index)
    data = np.ones(len(row_ind))
    matrix = csr_matrix((data, (row_ind, col_ind)), shape=(len(users), len(categories)))
    return matrix, user_index

user_mat, user_index = get_user_matrix()

# Nearest Neighbors model
@st.cache_resource
def get_knn_model():
    model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10, n_jobs=-1)
    model.fit(user_mat)
    return model

model = get_knn_model()

# Build content profiles
@st.cache_data
def build_profiles():
    item_profiles = {}
    post_cat_map = main.groupby('post_id')['category'].apply(set).to_dict()
    for cat in categories:
        item_profiles[cat] = np.array([1.0 if cat in post_cat_map.get(pid, set()) else 0.0 for pid in posts], dtype=float)

    user_profiles = {}
    for uid in users:
        user_data = main[main['user_id'] == uid]
        viewed_cats = user_data['category'].value_counts(normalize=True).to_dict()
        result_vector = np.sum([weight * item_profiles[cat] for cat, weight in viewed_cats.items() if cat in item_profiles], axis=0)
        user_profiles[uid] = result_vector
    return item_profiles, user_profiles

item_profiles, user_profiles = build_profiles()

# Collaborative recommender
@st.cache_data
def recommender_collab(user_id):
    index = users.index(user_id)
    current_user_cats = set(main[main['user_id'] == user_id]['category'])
    _, indices = model.kneighbors(user_mat[index], 10)
    recommendations = set()
    for i in indices[0]:
        peer_cats = set(main[main['user_id'] == users[i]]['category'])
        recommendations.update(peer_cats - current_user_cats)
    return list(recommendations)[:10]

# Content recommender
@st.cache_data
def recommender_content(user_id):
    user_vector = user_profiles[user_id].reshape(1, -1)
    scores = {cat: cosine_similarity(user_vector, vec.reshape(1, -1))[0][0] for cat, vec in item_profiles.items()}
    sorted_cats = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    user_seen = set(main[main['user_id'] == user_id]['post_id'])
    recs = []
    for cat, _ in sorted_cats:
        for pid in main[main['category'] == cat]['post_id'].unique():
            if pid not in user_seen:
                recs.append((cat, pid))
            if len(recs) >= 10:
                break
        if len(recs) >= 10:
            break
    return recs

# UI Filters
st.subheader("🎯 Select Preferences")
user_id = st.selectbox("👤 Select User ID", users)

with st.expander("🔎 Filter Posts"):
    post_type_filter = st.multiselect("Post Types", post1[' post_type'].unique(), default=post1[' post_type'].unique())
    category_filter = st.multiselect("Categories", categories, default=categories)

# Show recommendations
if st.button("🚀 Get Recommendations"):
    with st.spinner("Finding best recommendations for you..."):
        colab_recs = recommender_collab(user_id)
        content_recs = recommender_content(user_id)

        st.subheader("🤝 Collaborative Filtering Recommendations")
        for cat in colab_recs:
            if cat in category_filter:
                st.markdown(f"<div class='recommendation-box'><span class='big-font'>{cat}</span><br><span class='small-font'>Suggested Category</span></div>", unsafe_allow_html=True)

        st.subheader("📚 Content-Based Filtering Recommendations")
        for cat, pid in content_recs:
            meta = post_lookup.get(pid)
            if meta:
                title = meta['title']
                ptype = meta[' post_type']
                if cat in category_filter and ptype in post_type_filter:
                    st.markdown(f"""
                    <div class='recommendation-box'>
                        <span class='big-font'>{title}</span><br>
                        <span class='small-font'>Post ID: {pid} | Category: {cat} | Type: {ptype}</span>
                    </div>
                    """, unsafe_allow_html=True)
