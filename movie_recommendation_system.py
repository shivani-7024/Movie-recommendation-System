print("Loading....")

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
nltk.download('punkt')

# Load the IMDb movie dataset
dataset_path = r"C:\Users\LENOVO\Downloads\imdb_real_200_movies.csv" 
movies = pd.read_csv(dataset_path)

# Preprocess text function
def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(str(text).lower())
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    return " ".join(filtered_words)

# Apply text preprocessing
movies["Processed_Description"] = movies["Description"].apply(preprocess_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(movies["Processed_Description"])

# Function to Recommend Movies by Genre
def recommend_movies_by_genre(genre, top_n=5):
    genre_movies = movies[movies["Genre"].str.contains(genre, case=False, na=False)]

    if genre_movies.empty:
        print("❌ No movies found for this genre!")
        return
    
    print(f"🎬 Top {top_n} movies in '{genre}':")
    for title in genre_movies["Title"].head(top_n):
        print(f"➡️ {title}")

# Genre Selection Menu
def genre_menu():
    genres = movies["Genre"].unique().tolist()
    
    print("\n📌 Select a Genre:")
    for idx, genre in enumerate(genres, start=1):
        print(f"{idx}. {genre}")

    try:
        choice = int(input("\nEnter the genre number: "))
        if 1 <= choice <= len(genres):
            selected_genre = genres[choice - 1]
            recommend_movies_by_genre(selected_genre)
        else:
            print("❌ Invalid choice! Please enter a valid number.")
    except ValueError:
        print("❌ Invalid input! Please enter a number.")

# Run the genre selection
genre_menu()