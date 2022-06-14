import pandas as pd

metadata = pd.read_csv('~/Documents/Documents - MACQ199NQ496F/machine_learning/beginner_tutorial_recommender_systems_python_datacamp/the_movies_dataset/movies_metadata.csv', low_memory=False)

print(metadata.head(3))

C = metadata['vote_average'].mean()

print( C )

m = metadata['vote_count'].quantile(0.90)

print(m)

q_movies = metadata.copy().loc[metadata['vote_count'] >= m ]

print(q_movies.shape)

print(metadata.shape)

def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']

    return (v / (v + m) ) * R + ( v / (v + m) ) * C

q_movies['Score'] = q_movies.apply(weighted_rating, axis=1)

q_movies = q_movies.sort_values('Score', ascending=False)

print(q_movies[['title', 'vote_count', 'vote_average', 'Score']].head(20))

df = pd.DataFrame(q_movies[['title', 'vote_count', 'vote_average', 'Score']].head(20))

df.to_csv('top_20_movies.csv', index=False)
