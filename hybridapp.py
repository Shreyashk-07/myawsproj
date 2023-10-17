import streamlit as st
import pickle
import pandas as pd
import numpy as np

popularity_data = pickle.load(open('popularity_df.pkl', 'rb'))
content_data = pickle.load(open('content_df.pkl','rb'))
similarity = pickle.load(open('sim.pkl','rb'))
def popularity(selectgenre, num_rating, topn=5):
    popularitydf = popularity_data.groupby(['genres', 'title']).agg({'rating': ["mean", "size"]}).reset_index()
    popularitydf.columns = ['genres', 'title', 'rating_mean', 'rating_count']

    topn_recommended = popularitydf[
        (popularitydf['genres'] == selectgenre) & (popularitydf['rating_count'] > num_rating)].sort_values(
        by='rating_mean', ascending=False).head(topn)

    topn_recommended['S.No.'] = list(range(1, len(topn_recommended) + 1))
    topn_recommended.index = range(len(topn_recommended))
    topn_recommended.columns = ['Genre', 'Movie_Title', 'Average_Movie_Rating', 'Number_Reviews', 'S.No']
    return topn_recommended[['Movie_Title', 'Average_Movie_Rating', 'Number_Reviews']]


def content_based(moviec, sim, movie_title, topn=5):
    titles = moviec['title']
    indices = pd.Series(moviec.index, index=moviec['title'])
    index = indices[movie_title]
    cosine_sim = list(enumerate(sim[index]))
    cosine_sim = sorted(cosine_sim, key=lambda x: x[1], reverse=True)
    cosine_sin = cosine_sim[1:topn + 2]  # Removed the '+2' since we only need 'topn' items
    matching = [i[0] for i in cosine_sin]
    matches = moviec.iloc[matching]
    matches = matches[matches['title'] != movie_title]  # Fixed the variable name 'matches_df'

    matches.rename(columns={'title': 'Movie_Title'}, inplace=True)
    matches['S.No'] = range(1, len(matches) + 1)
    matches.index = range(len(matches))

    return matches['Movie_Title'].head(topn)

def fetch_user_history(userid):
    user_rating=ratings[ratings['userId']==userid]
    user_history=pd.merge(user_rating,movie_col,how='inner',on='movieId').sort_values('movieId')
    return user_history

def generate_recommendations(target_user, p, k, topn):
    # Fetch user history for the target user
    user_history = fetch_user_history(target_user)

    # Find similar users based on their movie ratings
    similar_users = ratings[ratings['movieId'].isin(user_history['movieId'].tolist())]

    # Calculate the number of movies reviewed by other users
    num_review = similar_users.groupby('userId').agg({"movieId": "nunique"}).reset_index().rename(columns={'movieId': "num_movies_reviewed"}).sort_values(by='num_movies_reviewed', ascending=False)

    # Remove the target user's own ID
    num_review = num_review[num_review['userId'] != target_user]

    # Count the number of movies rated by the target user
    rated_by_target_user = user_history['movieId'].nunique()

    # Calculate the threshold as a percentage of movies rated by the target user
    threshold = int((p / 100) * rated_by_target_user)

    # Filter users who have reviewed more movies than the threshold
    num_review = num_review[num_review['num_movies_reviewed'] > threshold]

    # Get the ratings data of similar users
    above_threshold = similar_users[similar_users['userId'].isin(num_review['userId'].tolist())]

    # Create vectors for the target
    target_vector = user_history.pivot(index='userId', columns=['movieId'], values=['rating'])
    target_vector.columns = [str(each[1]) for each in target_vector.columns]

    # Create vector for the similar users
    user_vector = above_threshold.pivot(index='userId', columns=['movieId'], values=['rating']).fillna(0)
    user_vector.columns = [str(each[1]) for each in user_vector.columns]

    # Make sure the columns match between the target user and user vectors
    for each in set(target_vector.columns) - set(user_vector.columns):
        user_vector[each] = 0.0

    user_vector = user_vector[target_vector.columns]

    # Calculate similarity scores using TF-IDF and cosine similarity
    sim_df = pd.DataFrame(cosine_similarity(target_vector, user_vector)).T
    sim_df.index = user_vector.index
    sim_df.columns = ['similarity_score']

    # Find the indices of the top k similar users
    top_k_users = list(sim_df.sort_values('similarity_score', ascending=False).index[:k])

    # Select rating data of the top k users
    user_subset = ratings[ratings['userId'].isin(top_k_users)]

    # Select movies not rated by the target user
    user_subset = user_subset[~user_subset['movieId'].isin(user_history['movieId'])]

    # Calculate average ratings for each movie based on the ratings of the top k users
    average_ratings = user_subset.groupby('movieId').mean()['rating'].reset_index().sort_values('rating',
                                                                                                ascending=False)[:topn]

    # Merge with the movie_col dataframe to get movie titles
    recommendations = pd.merge(average_ratings, movie_col)
    recommendations.columns = ['Movie Id', 'Average Rating', 'Movie Title']
    recommendations['S.No'] = range(1, len(recommendations) + 1)

    # Sort the target user's movie history based on ratings
    user_history.rename(columns={'title': 'Movie Title'}, inplace=True)
    user_history = user_history.sort_values('rating', ascending=False)
    user_history['S.No'] = range(1, len(user_history) + 1)

    return recommendations['Movie Title']

col1, col2, col3 = st.columns([2,1,2])

col1.markdown("# MovieMate - Your Movie Matchmaker!")
col1.write('''Tired of the movie struggle? Meet MovieMate, your movie matchmaker! Say goodbye to endless scrolling and hello to instant movie recommendations.
   With MovieMate, picking a movie is a breeze! No more wasting time searching or reading reviews. Just sit back and let **MovieMate** work its magic!
   Our clever algorithm explores a world of films to find the perfect match for you. Whether you want action, romance, horror, or comedy, MovieMate 
   has it all!''')

# Popularity

genres=list(set(popularity_data['genres']))

col3.markdown("### Popularity Based Recommendation")
selectgenre = col3.selectbox('Genre',genres)

num_rating = col3.slider('Reviewed by people', 0, 320)
topn = col3.slider('Number of recommendations', 0, 50)

if col3.button('Search', key=1):
    rec=popularity(selectgenre,num_rating,topn)
    col3.dataframe(rec)

# Content

movies=list(content_data['title'])
col1.markdown("### Content Based Recommendation")

selected_title = col1.selectbox('Movie',movies)
content_top = col1.slider('Number of Recommendations', 0, 50)

if col1.button('Search', key=2):
    rec=content_based(content_data,similarity,selected_title,content_top)
    col1.dataframe(rec)

