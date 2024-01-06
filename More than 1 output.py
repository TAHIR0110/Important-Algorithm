#!/usr/bin/env python
# coding: utf-8

# In[58]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

data = {
    'Title': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
    'Genre': ['Action', 'Comedy', 'Drama', 'Action', 'Comedy']
}
movies_df = pd.DataFrame(data)

label_encoder = LabelEncoder()
movies_df['Genre_Label'] = label_encoder.fit_transform(movies_df['Genre'])

vectorizer = CountVectorizer()
genre_matrix = vectorizer.fit_transform(movies_df['Genre'])

# Train an XGBoost classifier
xgb = XGBClassifier()
xgb.fit(genre_matrix, movies_df['Genre_Label'])

def recommend_movies_by_genre(input_genre, num_recommendations=5):
    input_genre_matrix = vectorizer.transform([input_genre])
    predicted_genre_label = xgb.predict(input_genre_matrix)[0]
    
    # Reverse mapping to get movie titles from predicted genre labels
    predicted_movies = movies_df[movies_df['Genre_Label'] == predicted_genre_label]['Title']
    
    return predicted_movies[:num_recommendations].values

# Take user input for genre
input_genre = input("Enter a movie genre: ")  # User input for genre
recommendations = recommend_movies_by_genre(input_genre)
print(f"Top 5 movies in the {input_genre} genre:")
for i, movie in enumerate(recommendations, start=1):
    print(f"{i}. {movie}")
movies_df


# In[57]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

data = {
    'Title': ['avengers', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
    'Genre': ['Action', 'Comedy', 'Drama', 'Action', 'Comedy']
}
movies_df = pd.DataFrame(data)

label_encoder = LabelEncoder()
movies_df['Genre_Label'] = label_encoder.fit_transform(movies_df['Genre'])

rf = RandomForestClassifier()
rf.fit(movies_df[['Genre_Label']], movies_df['Genre_Label'])

def recommend_movies_by_genre(input_genre, num_recommendations=5):
    input_genre_label = label_encoder.transform([input_genre])[0]
    predicted_genre_label = rf.predict([[input_genre_label]])[0]
    
    predicted_movies = movies_df[movies_df['Genre_Label'] == predicted_genre_label]['Title']
    
    print(f"the top movies in the {input_genre} genre are: ")
    print(predicted_movies[:num_recommendations].values)

input_genre = 'Action'
recommendations = recommend_movies_by_genre(input_genre)
movies_df


# # The below is for more than 1 genre

# In[56]:


import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier

data = {
    'Title': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
    'Genre': [['Action', 'Thriller'], ['Comedy', 'Romance'], ['Drama'], ['Action', 'Adventure'], ['Comedy']]
}
movies_df = pd.DataFrame(data)

mlb = MultiLabelBinarizer()
genre_labels = mlb.fit_transform(movies_df['Genre'])

rf = RandomForestClassifier()
rf.fit(genre_labels, movies_df['Title'])

def recommend_movies_by_genres(input_genres, num_recommendations=2):
    input_genre_labels = mlb.transform([input_genres])[0]
    predicted_probabilities = rf.predict_proba([input_genre_labels])[0]
    
    top_movie_indices = predicted_probabilities.argsort()[::-1][:num_recommendations]
    
    recommended_movies = [movies_df.iloc[i]['Title'] for i in top_movie_indices]
    print(f"Top movies in the genres {input_genres}:")
    print(recommended_movies)

input_genres = ['Action', 'Adventure']
recommendations = recommend_movies_by_genres(input_genres)
movies_df


# In[ ]:





# In[15]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

data = {
    'Title': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
    'Genre': ['Action', 'Comedy', 'Drama', 'Action', 'Comedy'],
    'Name': ['Name1', 'Name2', 'Name3', 'Name4', 'Name5']
}
movies_df = pd.DataFrame(data)

label_encoder_genre = LabelEncoder()
movies_df['Genre_Label'] = label_encoder_genre.fit_transform(movies_df['Genre'])

label_encoder_name = LabelEncoder()
movies_df['Name_Label'] = label_encoder_name.fit_transform(movies_df['Name'])

rf = RandomForestClassifier()
rf.fit(movies_df[['Genre_Label', 'Name_Label']], movies_df['Title'])

def recommend_movies_by_features(input_genre, input_name, num_recommendations=2):
    input_genre_label = label_encoder_genre.transform([input_genre])[0]
    input_name_label = label_encoder_name.transform([input_name])[0]
    
    input_features = [[input_genre_label, input_name_label]]
    
    predicted_probabilities = rf.predict_proba(input_features)
    
    top_movie_indices = predicted_probabilities.argsort()[0][::-1][:num_recommendations]
    
    recommended_movies = [movies_df.iloc[i]['Title'] for i in top_movie_indices]
    print(f"Top movies based on genre '{input_genre}' and name '{input_name}':")
    print(recommended_movies)

input_genre = 'Action'
input_name = 'Name2'
recommendations = recommend_movies_by_features(input_genre, input_name)


# In[ ]:




