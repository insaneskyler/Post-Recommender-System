# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# Title
st.title("AI-based Post Recommender System")

# Load data with caching
@st.cache_data
def load_data():
    post = pd.read_csv("posts.csv")
    user = pd.read_csv("users.csv")
    view = pd.read_csv("views.csv")
    return post, user, view

post, user, view = load_data()

# Clean data
post["category"] = post["category"].fillna("random")

# Split categories
@st.cache_data
def process_post_data(post):
    cat = {}
    for i in post["category"]:
        cat[i] = i.split("|")

    updated_data = []
    for i, subcats in cat.items():
        dummy = post[post["category"] == i]
        if dummy.empty:
            continue
        id = dummy["_id"].values[0]
        title = dummy["title"].values[0]
        post_type = dummy[" post_type"].values[0]
        for subcat in subcats:
            updated_data.append({
                "_id": id,
                "title": title,
                "category": subcat,
                " post_type": post_type
            })

    post1 = pd.DataFrame(updated_data)
    post1.rename(columns={"_id": "post_id"}, inplace=True)
    return post1

post1 = process_post_data(post)
main = pd.merge(view, post1, on="post_id")

# Prepare user-category matrix
@st.cache_data
def build_user_matrix(main):
    users = main["user_id"].unique().tolist()
    categories = main["category"].unique().tolist()

    user_index = {user: idx for idx, user in enumerate(users)}
    category_index = {cat: idx for idx, cat in enumerate(categories)}
    
    data = []
    row = []
    col = []

    for _, row_data in main.iterrows():
        u = user_index[row_data["user_id"]]
        c = category_index[row_data["category"]]
        data.append(1)
        row.append(u)
        col.append(c)

    matrix = csr_matrix((data, (row, col)), shape=(len(users), len(categories)))
    return matrix, users, categories

user_mat, users, categories = build_user_matrix(main)

# Fit collaborative model once
model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=15)
model.fit(user_mat)

# Collaborative filtering
def recommender(user_id):
    index = users.index(user_id)
    current_user = main[main['user_id'] == user_id]
    distances, indices = model.kneighbors(user_mat[index], 15)
    recommendation = []
    seen_categories = set(current_user['category'].unique())
    for i in indices[0]:
        neighbor = main[main['user_id'] == users[i]]
        for cat in neighbor['category'].unique():
            if cat not in seen_categories:
                recommendation.append(cat)
    return recommendation[:10]

# Content-based filtering setup
@st.cache_data
def build_profiles(main, users, categories):
    posts = main["post_id"].unique().tolist()
    post_index = {post_id: idx for idx, post_id in enumerate(posts)}

    item_profiles = {
        cat: np.array([
            1 if cat in main[main['post_id'] == pid]['category'].values else 0
            for pid in posts
        ]) for cat in categories
    }

    user_profiles = {}
    for user in users:
        current_user = main[main['user_id'] == user]
        categories_viewed = current_user['category'].tolist()
        posts_viewed = current_user['post_id'].tolist()

        weights = {cat: categories_viewed.count(cat)/len(posts_viewed) for cat in set(categories_viewed)}
        result_vector = np.zeros(len(posts), dtype=float)

        for cat, weight in weights.items():
            result_vector += weight * item_profiles[cat].astype(float)

        user_profiles[user] = result_vector / len(posts_viewed)

    return item_profiles, user_profiles, posts

item_profiles, user_profiles, all_posts = build_profiles(main, users, categories)

# Content-based filtering
def recommender1(user_id):
    similarities = {
        cat: cosine_similarity(user_profiles[user_id].reshape(1, -1), item_profiles[cat].reshape(1, -1))[0][0]
        for cat in categories
    }
    sorted_cats = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    user_posts = set(main[main['user_id'] == user_id]['post_id'])
    recommendations = []

    for cat, _ in sorted_cats:
        posts_in_cat = main[main['category'] == cat]['post_id'].unique()
        for post_id in posts_in_cat:
            if post_id not in user_posts:
                recommendations.append((cat, post_id))
        if len(recommendations) >= 20:
            break

    return recommendations[:10]

# UI
user_id = st.selectbox("Select a User ID", users)

if st.button("Get Recommendations"):
    st.subheader("Collaborative Filtering")
    colab_recs = recommender(user_id)
    st.write(colab_recs)

    st.subheader("Content-Based Filtering")
    content_recs = recommender1(user_id)
    for category, post_id in content_recs:
        title = post[post["_id"] == post_id]["title"].values
        if len(title) > 0:
            st.markdown(f"- **{title[0]}** _(Category: {category})_")
