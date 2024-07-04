import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the pickle files
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('book_pivot.pkl', 'rb') as f:
    book_pivot = pickle.load(f)

with open('final_rating.pkl', 'rb') as f:
    final_dataframe = pickle.load(f)


# Define the recommend_book function
def recommend_book(book_name):
    try:
        book_id = np.where(book_pivot.index == book_name)[0][0]
    except IndexError:
        st.write(f"Book '{book_name}' not found in the dataset.")
        return [], []

    distances, suggestions = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)

    recommended_books = []
    recommended_images = []
    for i in range(len(suggestions)):
        x = book_pivot.index[suggestions[i]]
        for j in range(len(x)):
            if x[j] == book_name:
                continue
            else:
                recommended_books.append(x[j])
                # Get the corresponding image URL from final_dataframe
                image_url = final_dataframe.loc[final_dataframe['title'] == x[j], 'Image-URL-L'].values[0]
                recommended_images.append(image_url)
    return recommended_books, recommended_images


# Streamlit application
st.title('Book Recommender System')

book_list = book_pivot.index.tolist()
selected_book = st.selectbox('Select a book to get recommendations', book_list)

if st.button('Recommend'):
    recommendations, images = recommend_book(selected_book)
    if recommendations:
        st.write('Recommendations:')
        num_cols = 3  # Number of columns to display recommendations side by side
        col_width = int(12 / num_cols)
        num_recommendations = len(recommendations)

        # Display recommendations in columns
        for i in range(0, num_recommendations, num_cols):
            cols = st.columns(num_cols)
            for j in range(num_cols):
                if i + j < num_recommendations:
                    with cols[j]:
                        st.write(recommendations[i + j])
                        st.image(images[i + j], width=150)
    else:
        st.write('No recommendations found.')

