import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

st.write("📁 Files available in app:", os.listdir())

st.set_page_config(layout="wide")
# --- 1. Define all CSS styles ONCE at the top of the app ---
st.markdown("""
<style>
    
    
    .stApp {
        background-color: #f0f2fgoing;
    }
    .book-card {
        /* --- ✅ FIX: Make card fill the column flexibly --- */
        width: 95%;
        box-sizing: border-box; /* This is the key */

        height: 100%;
        text-align: center;
        border-radius: 12px;
        background-color: #282626;
        padding: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: transform 0.2s ease-in-out;
        margin-bottom: 25px;
    }
    .book-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 8px 16px rgb(52, 152, 219);
    }
    .book-img {
        width: 100%;
        height: 280px;
        object-fit: cover;
        border-radius: 10px;
    }
    .book-title {
        margin-top: 12px;
        font-size: 16px;
        font-weight: 600;
        color: #8B8B8B;
        height: 50px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# --- Load data ONCE at the start ---
if os.path.exists("embeddings_small.npz"):
    data = np.load("embeddings_small.npz")
    embeddings = data["emb"].astype(np.float32)
else:
    st.warning("Upload embeddings_small.npz to run the recommender.")
    uploaded_file = st.file_uploader("Upload embeddings_small.npz", type=["npz"])

    if uploaded_file:
        data = np.load(uploaded_file)
        embeddings = data["emb"].astype(np.float32)
    else:
        st.stop()
new_df = pd.read_csv('new_df.csv')
book_list = new_df['title'].values

# --- Initialize session state ---
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'num_to_show' not in st.session_state:
    st.session_state.num_to_show = 5  # Start with 5

st.title('Book Recommender')
Selected_book_name = st.selectbox('Select Book', book_list)


def recommend(book_title, df, vector):
    try:
        book_index = df[df['title'] == book_title].index[0]
    except IndexError:
        return [], []

    similarity_scores = cosine_similarity(vector[book_index:book_index + 1], vector)
    book_list_indices = list(enumerate(similarity_scores[0]))
    sorted_book_list = sorted(book_list_indices, reverse=True, key=lambda x: x[1])[1:]

    recommended = []
    recommended_books_posters = []
    for i in sorted_book_list[0:50]:
        recommended.append(df.iloc[i[0]].title)
        recommended_books_posters.append(df.iloc[i[0]].coverImg)

    return recommended, recommended_books_posters


# --- "Recommend" button's ONLY job is to calculate and store results ---
if st.button('Recommend'):
    names, posters = recommend(Selected_book_name, new_df, embeddings)
    st.session_state.recommendations = list(zip(names, posters))
    st.session_state.num_to_show = 10  # Reset view to 5

# --- 2. Display logic now uses st.columns() for layout ---
if st.session_state.recommendations:
    recs_to_display = st.session_state.recommendations[:st.session_state.num_to_show]
    num_columns = 5

    # Loop to create rows
    for i in range(0, len(recs_to_display), num_columns):
        cols = st.columns(num_columns)
        row_recs = recs_to_display[i: i + num_columns]

        # Loop to populate columns in the current row
        for j, rec in enumerate(row_recs):
            with cols[j]:
                name, poster = rec
                # Basic check for valid poster URL
                if isinstance(poster, str) and poster.startswith('http'):
                    # Sanitize the book title
                    clean_name = name.replace('"', '&quot;').replace("'", "&apos;")

                    # 3. Apply the custom HTML/CSS to each card INSIDE the column
                    st.markdown(f"""
                    <div class="book-card">
                        <img src="{poster}" class="book-img"/>
                        <div class="book-title">{clean_name}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # "Show More" button logic
    if st.session_state.num_to_show < len(st.session_state.recommendations):
        # Center the button
        st.markdown("<br>", unsafe_allow_html=True)  # Add some space
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            if st.button('Show More'):
                st.session_state.num_to_show += 10








