import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print(pd.__version__)


class Recommender:
    def __init__(self, matrix, songs):
        self.matrix_similar = matrix
        self.songs = songs

    def _print_message(self, song, recom_song):
        rec_items = len(recom_song)

        display_list = []
        print(f'The {rec_items} recommended songs are:')
        for i in range(rec_items):
            display_dict = {}
            print(f"Number {i+1}:")
            print(f"{recom_song[i][1]} by {recom_song[i][2]}")
            print("--------------------")
            display_dict['Sr. No.'] = i+1
            display_dict['Song'] = recom_song[i][1]
            display_dict['Artist'] = recom_song[i][2]
            display_list.append(display_dict)
        display_text = pd.DataFrame(display_list)
        return display_text

    def recommend(self, recommendation):
        text_input = recommendation['text']
        number_songs = recommendation['number_songs']
        text_input = text_input.replace(r'\n', '')
        input_vector = tfidf.transform([text_input])
        similarities = cosine_similarity(input_vector, lyrics_matrix)
        similar_indices = similarities.argsort()[0][::-1][:number_songs]
        recom_song = [(similarities[0][x], self.songs['song'][x],
                       self.songs['artist'][x]) for x in similar_indices]

        return self._print_message(song="Input_Song", recom_song=recom_song)


with open('models/tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('models/lyrics_matrix.pkl', 'rb') as f:
    lyrics_matrix = pickle.load(f)

with open('models/recommender.pkl', 'rb') as f:
    recommender = pickle.load(f)

st.set_page_config(page_title="Lyrics Analysis and Music Recommendation with Pair Similarities",
                   page_icon=":musical_note:",
                   layout="wide")

st.title('Lyrics Analysis and Music Recommendation with Pair Similarities')
st.write('Enter text and number of songs to get recommendations.')

text_input = st.text_area('Enter a song lyric')

number_songs = st.number_input(
    'Number of songs to recommend', min_value=1, max_value=10, value=5)

if st.button('Recommend'):
    query = {
        "text": text_input,
        "number_songs": number_songs
    }

    recommendations = recommender.recommend(query)
    st.write(recommendations)
