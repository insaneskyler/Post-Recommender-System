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
updated_data = []
for idx, row in post.iterrows():
    for cat in row['category'].split('|'):
        updated_data.append({
            'post_id': row['_id'],
            'title': row['title'],
            'category': cat.strip(),
            'post_type': row[' post_type']
        })

post1 = pd.DataFrame(updated_data)

# MERGE AND PREPARE FINAL DATASET
main = pd.merge(view, post1, on='post_id')
users = list(main['user_id'].unique())
categories = list(main['category'].unique())
posts = list(main['post_id'].unique())

# USER-CATEGORY MATRIX
@st.cache_data
def get_user_matrix():
    user_cat_matrix = np.zeros((len(users), len(categories)))
    user_index = {uid: i for i, uid in enumerate(users)}
    cat_index = {cat: i for i, cat in enumerate(categories)}
    for _, row in main.iterrows():
        user_idx = user_index[row['user_id']]
        cat_idx = cat_index[row['category']]
        user_cat_matrix[user_idx][cat_idx] += 1
    return csr_matrix(user_cat_matrix), user_index

user_mat, user_index = get_user_matrix()

# Nearest Neighbors for Collaborative Filtering
model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10)
model.fit(user_mat)

# CONTENT-BASED PROFILE
@st.cache_data
def build_profiles():
    item_profiles = {}
    for cat in categories:
        item_profiles[cat] = np.array([
            cat in main[main['post_id'] == pid]['category'].values
            for pid in posts
        ], dtype=float)

    user_profiles = {}
    for uid in users:
        user_data = main[main['user_id'] == uid]
        viewed_cats = user_data['category'].value_counts(normalize=True).to_dict()
        profile = np.zeros(len(posts))
        for cat, weight in viewed_cats.items():
            profile += weight * item_profiles[cat]
        user_profiles[uid] = profile
    return item_profiles, user_profiles

item_profiles, user_profiles = build_profiles()

# COLLABORATIVE FILTERING RECOMMENDER
@st.cache_data
def recommender_collab(user_id):
    index = users.index(user_id)
    current_user = main[main['user_id'] == user_id]
    _, indices = model.kneighbors(user_mat[index], 10)
    recommendations = set()
    for i in indices[0]:
        peer = main[main['user_id'] == users[i]]
        for cat in peer['category'].unique():
            if cat not in current_user['category'].unique():
                recommendations.add(cat)
    return list(recommendations)[:10]

# CONTENT-BASED RECOMMENDER
@st.cache_data
def recommender_content(user_id):
    similarity = {
        cat: cosine_similarity(
            user_profiles[user_id].reshape(1, -1),
            item_profiles[cat].reshape(1, -1)
        )[0][0] for cat in item_profiles
    }
    sorted_sim = sorted(similarity.items(), key=lambda x: x[1], reverse=True)
    user_posts = set(main[main['user_id'] == user_id]['post_id'])
    recs = []
    for cat, _ in sorted_sim:
        for pid in main[main['category'] == cat]['post_id'].unique():
            if pid not in user_posts:
                recs.append((cat, pid))
            if len(recs) == 10:
                return recs
    return recs

# UI
st.subheader("🎯 Select Preferences")
user_id = st.selectbox("👤 Select User ID", users)

with st.expander("🔎 Filter Posts"):
    post_type_filter = st.multiselect("Post Types", post1['post_type'].unique().tolist(), default=post1['post_type'].unique().tolist())
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
