import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import linear_kernel
import time
from re import sub

title_to_recommend = input('\nPlease write the name of the movie (cap sensitive): ')

top_x_number = input('\nPlease input how many movies you would like to be recommended: ')

print('\nFetching movies...')

def snake_case(s):
  return '_'.join(
    sub('([A-Z][a-z]+)', r' \1',
    sub('([A-Z]+)', r' \1',
    s.replace('-', ' '))).split()).lower()

metadata = pd.read_csv('~/Documents/Documents - MACQ199NQ496F/machine_learning/recommender_systems_python/the_movies_dataset/movies_metadata.csv', low_memory=False)

tfidf = TfidfVectorizer(stop_words='english')

metadata['overview'] = metadata['overview'].fillna('')

tfidf_matrix = tfidf.fit_transform(metadata['overview'])

#print(tfidf_matrix.shape)

#print(tfidf.get_feature_names_out()[5000:5010])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

#print(cosine_sim.shape)

indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()

#print(indices[:10])

def get_recommendations(title, top, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:int(top)]
    movie_indices = [i[0] for i in sim_scores]
    return metadata['title'].iloc[movie_indices]

print('\nFor the movie titled "' + str(title_to_recommend) + '" we recommend the top ' + str(top_x_number) + ' movies below. (The results are also available as a csv created in this folder.)\n')

print(get_recommendations(title_to_recommend, top_x_number))

pd.DataFrame(get_recommendations(title_to_recommend, top_x_number)).to_csv(str(snake_case(title_to_recommend)) + '_top_' + str(top_x_number) + '_recommendations.csv', index=False)

