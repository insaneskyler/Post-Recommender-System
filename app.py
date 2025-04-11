# AI-Based Post Recommender System

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# UI SETUP
st.set_page_config(page_title="Post Recommender", layout="wide")
st.title("📬 AI-Based Post Recommender System")

st.markdown("""
<style>
.big-font {font-size:20px !important; font-weight:600;}
.small-font {font-size:14px !important; color:#aaa;}
.recommendation-box {
    border: 1px solid #444;
    border-radius: 10px;
    padding: 10px;
    background-color: #1e1e1e;
    margin-bottom: 10px;
    color: #f1f1f1;
}
input, .stSelectbox > div > div, select, textarea {
    background-color: #262730 !important;
    color: #f1f1f1 !important;
}
</style>
""", unsafe_allow_html=True)

# LOAD DATA with caching for performance
@st.cache_data
def load_data():
    posts = pd.read_csv("posts.csv")
    users = pd.read_csv("users.csv")
    views = pd.read_csv("views.csv")
    return posts, users, views

post, user, view = load_data()

# DATA CLEANING
post['category'] = post['category'].fillna('random')

# Expand category combinations
post1 = post.dropna(subset=['category']).copy()
post1 = post1.assign(category=post1['category'].str.split('|'))
post1 = post1.explode('category')
post1['category'] = post1['category'].str.strip()

# MERGE AND PREPARE FINAL DATASET
main = pd.merge(view, post1, left_on='post_id', right_on='_id')
users = main['user_id'].unique().tolist()
categories = main['category'].unique().tolist()
posts = main['post_id'].unique().tolist()

# USER-CATEGORY MATRIX
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

# Nearest Neighbors for Collaborative Filtering
@st.cache_resource
def get_knn_model():
    model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10, n_jobs=-1)
    model.fit(user_mat)
    return model

model = get_knn_model()

# CONTENT-BASED PROFILE
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

# COLLABORATIVE FILTERING RECOMMENDER
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

# CONTENT-BASED RECOMMENDER
@st.cache_data
def recommender_content(user_id):
    user_vector = user_profiles[user_id].reshape(1, -1)
    scores = {cat: cosine_similarity(user_vector, vec.reshape(1, -1))[0][0] for cat, vec in item_profiles.items()}
    sorted_cats = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    user_seen_posts = set(main[main['user_id'] == user_id]['post_id'])
    recs = [(cat, pid) for cat, _ in sorted_cats for pid in main[main['category'] == cat]['post_id'].unique() if pid not in user_seen_posts]
    return recs[:10]

# UI
st.subheader("🎯 Select Preferences")
user_id = st.selectbox("👤 Select User ID", users)

with st.expander("🔎 Filter Posts"):
    post_type_filter = st.multiselect("Post Types", post1[' post_type'].unique(), default=post1[' post_type'].unique())
    category_filter = st.multiselect("Categories", categories, default=categories)

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
            title = post[post['_id'] == pid]['title'].values[0]
            ptype = post[post['_id'] == pid][' post_type'].values[0]
            if cat in category_filter and ptype in post_type_filter:
                st.markdown(f"""
                <div class='recommendation-box'>
                    <span class='big-font'>{title}</span><br>
                    <span class='small-font'>Post ID: {pid} | Category: {cat} | Type: {ptype}</span>
                </div>
                """, unsafe_allow_html=True)
