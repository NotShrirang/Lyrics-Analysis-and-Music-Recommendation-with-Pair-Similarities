# Lyrics Analysis and Music Recommendation with Pair Similarities

This project combines lyrics analysis and music recommendation using pair similarities. The application leverages a dataset from Kaggle containing information about a million songs on Spotify.

## Dataset
- The dataset used in this project can be found on Kaggle: [Spotify Million Song Dataset](https://www.kaggle.com/datasets/notshrirang/spotify-million-song-dataset).

## Hosted App
- The web application is hosted on Streamlit and can be accessed here: [Lyrics Analysis and Music Recommendation](https://lyrics-analysis-and-music-recommendation-with-pair-similarities.streamlit.app/).

## Overview
The application allows users to input text, and based on pair similarities, it recommends other songs with similar lyrics text. The recommendation model is trained on the provided dataset and utilizes TF-IDF vectorization and cosine similarity.

## Usage
### Using the Recommender in Your Code
```python
import pickle

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

# Example recommendation
query = {
    "text": """We're only gettin' older, baby
    And I've been thinkin' about it lately
    Does it ever drive you crazy
    Just how fast the night changes?
    Everything that you've ever dreamed of
    Disappearing when you wake up
    But there's nothing to be afraid of
    Even when the night changes
    It will never change me and you""",
    "number_songs": 5
}

# Get recommendations
recommender.recommend(query)
```

## Installation:
To run this project locally, you will need Python and Streamlit installed on your system. You can install the required packages using the provided `requirements.txt` file.

1. Clone Repo:

    ```sh
    git clone https://github.com/NotShrirang/Lyrics-Analysis-and-Music-Recommendation-with-Pair-Similarities.git
    ```

2. Change project directory:

    ```sh
    cd Lyrics-Analysis-and-Music-Recommendation-with-Pair-Similarities
    ```

3. Get requirements:

    ```sh
    pip install -r requirements.txt
    ```

## Run Streamlit Web App:

```sh
streamlit run app.py
```

Feel free to explore and enjoy discovering new music based on lyrics!
