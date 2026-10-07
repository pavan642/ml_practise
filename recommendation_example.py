

#pip3 install pandas
#pip3 install sklearn

#
#python3 recommendation_example.py
#
#🎬 Movies similar to 'Interstellar':
#----------------------------------------
#• The Martian (Similarity Score: 0.48)
#• Inception (Similarity Score: 0.35)
#
#🎬 Movies similar to 'The Notebook':
#----------------------------------------
#• La La Land (Similarity Score: 0.50)
#• Interstellar (Similarity Score: 0.00)



import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. 🎬 SAMPLE MOVIE DATASET
movies_data = {
    'movie_id': [1, 2, 3, 4, 5, 6],
    'title': [
        'Interstellar', 
        'Inception', 
        'The Dark Knight', 
        'The Notebook', 
        'La La Land', 
        'The Martian'
    ],
    'genres_and_keywords': [
        'Sci-Fi Space Time-Travel Christopher Nolan Exploration',
        'Sci-Fi Action Mind-Bending Christopher Nolan Dreams',
        'Action Crime Superhero Christopher Nolan Dark',
        'Romance Drama Love Emotional Relationship',
        'Romance Music Drama Musical Love',
        'Sci-Fi Space Survival Mars Exploration'
    ]
}

df_movies = pd.DataFrame(movies_data)

# 2. 🔤 VECTORIZATION (Convert text into math vectors)
# TfidfVectorizer turns words into numerical scores based on importance
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_movies['genres_and_keywords'])

# 3. 📐 CALCULATE COSINE SIMILARITY MATRIX
# Calculates similarity score (0.0 to 1.0) between every pair of movies
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 4. 🧠 RECOMMENDATION FUNCTION
def recommend_movies(movie_title, top_n=3):
    # Find the index of the input movie
    idx = df_movies[df_movies['title'] == movie_title].index[0]
    
    # Get pairwise similarity scores for all movies with this movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the movies based on similarity scores (highest first)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get scores of the top N most similar movies (ignoring index 0 because it's the movie itself)
    sim_scores = sim_scores[1:top_n+1]
    
    # Print results
    print(f"\n🎬 Movies similar to '{movie_title}':")
    print("-" * 40)
    for i in sim_scores:
        similar_movie_title = df_movies.iloc[i[0]]['title']
        score = i[1]
        print(f"• {similar_movie_title} (Similarity Score: {score:.2f})")

# 5. 🎯 TEST THE SYSTEM
recommend_movies('Interstellar', top_n=2)
recommend_movies('The Notebook', top_n=2)
