# -*- coding: utf-8 -*-
"""post-recommender-system-beginner.ipynb"""

# IMPORTING LIBRARIES
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# IMPORTING DATA
post = pd.read_csv("posts.csv")
user = pd.read_csv("users.csv")
view = pd.read_csv("views.csv")

# FILL MISSING CATEGORIES
post["category"] = post["category"].fillna("random")

# SPLIT MULTI-CATEGORY POSTS INTO MULTIPLE ROWS
cat = {}
for i in post["category"]:
    cat.update({i: []})
    for j in i.split("|"):
        cat[i].append(j)

updated_data = []
for i in cat:
    dummy = post[post['category'] == i]
    id = dummy['_id'].values[0]
    title = dummy['title'].values[0]
    post_type = dummy[' post_type'].values[0]
    for j in cat[i]:
        updated_data.append({
            '_id': id,
            'title': title,
            'category': j,
            ' post_type': post_type
        })

post1 = pd.DataFrame(updated_data)
post1.rename(columns={"_id": 'post_id'}, inplace=True)

# FIX POTENTIAL ERROR: user.rename
user.rename(columns={"user_id": 'user_id'}, inplace=True)

# MERGE DATA
main = pd.merge(view, post1, on="post_id")

# CREATE USER-CATEGORY MATRIX FOR COLLABORATIVE FILTERING
users = list(main["user_id"].unique())
categories = list(main["category"].unique())

user_mat = [[] for _ in range(len(users))]
for i in range(len(users)):
    for j in range(len(categories)):
        value = len(main[(main["user_id"] == users[i]) & (main["category"] == categories[j])])
        user_mat[i].append(value)

user_mat = csr_matrix(user_mat)
model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=15)
model.fit(user_mat)

def recommender(user_id, data=user_mat, model=model):
    model.fit(data)
    index = users.index(user_id)
    current_user = main[main['user_id'] == user_id]
    distances, indices = model.kneighbors(data[index], 15)
    recommendations = []
    for i in indices[0]:
        user = main[main['user_id'] == users[i]]
        for cat in user['category'].unique():
            if cat not in current_user['category'].unique():
                recommendations.append(cat)
    return recommendations[:10]

# CONTENT-BASED FILTERING

posts = list(main["post_id"].unique())

item_profiles = {}
for cat in categories:
    item_profiles.update({cat: []})
    for post_id in posts:
        item_profiles[cat].append(1 if cat in list(main[main['post_id'] == post_id]['category'].unique()) else 0)

# Convert to numpy arrays
for cat in item_profiles:
    item_profiles[cat] = np.array(item_profiles[cat])

user_profiles = {}
for u in users:
    current_user = main[main['user_id'] == u]
    current_user_categories = list(current_user['category'].unique())
    current_user_post = list(current_user['post_id'].unique())

    category_weight = {}
    result_vector = np.zeros(len(posts), dtype=float)  # FIXED casting error

    for cat in current_user_categories:
        category_weight[cat] = sum(1 for k in list(current_user['category']) if k == cat)
        category_weight[cat] /= len(current_user_post)
        result_vector += category_weight[cat] * item_profiles[cat]

    user_profiles[u] = result_vector / len(current_user_post)

def recommender1(user_id, user_profiles=user_profiles, item_profiles=item_profiles):
    similarity = {}
    for cat in item_profiles:
        similarity[cat] = cosine_similarity(user_profiles[user_id].reshape(1, -1), item_profiles[cat].reshape(1, -1))[0][0]

    sorted_similarity = sorted(similarity.items(), key=lambda x: x[1], reverse=True)

    user_posts = list(main[main['user_id'] == user_id]['post_id'].unique())
    recommendations = []
    for cat, _ in sorted_similarity:
        cat_posts = list(main[main['category'] == cat]['post_id'].unique())
        for post_id in cat_posts:
            if post_id not in user_posts:
                recommendations.append((cat, post_id))
        if len(recommendations) >= 20:
            break
    return recommendations

# STREAMLIT APP

st.title("AI-based Post Recommender")

user_id = st.selectbox("Select a User ID", users)

if st.button("Get Recommendations"):
    st.subheader("Collaborative Filtering")
    colab_recs = recommender(user_id)
    st.write(colab_recs)

    st.subheader("Content-Based Filtering")
    content_recs = recommender1(user_id)
    for category, post_id in content_recs:
        title_row = post[post["_id"] == post_id]
        if not title_row.empty:
            title = title_row["title"].values[0]
            st.markdown(f"- **{title}** _(Category: {category})_")
