from dotenv import load_dotenv
load_dotenv()
import os
from flask import Flask, render_template, request
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import lyricsgenius as genius
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pickle

analyzer = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

app = Flask(__name__)
cid=os.getenv("cid")
secret=os.getenv("secret")
ind = [0]

client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

api = genius.Genius(os.getenv("genius"))

model_with_lyrics = pickle.load(open("hit_song_predictor_model.pkl", 'rb'))
model_without_lyrics = pickle.load(open("hit_song_predictor_model_without_lyrics.pkl", 'rb'))

def featuring(artist):
    if "featuring" in artist :
        return 1
    else :
        return 0

def featuring_substring(artist):
    if "featuring" in artist :
        return artist.split("featuring")[0]
    else :
        return artist
    
def artist_info(lookup) :

    try :
        artist = sp.search(lookup)
        artist_uri = artist['tracks']['items'][0]['album']['artists'][0]['uri']
        track_uri = artist['tracks']['items'][0]['uri']

        available_markets = len(artist['tracks']['items'][0]['available_markets'])
        release_date = artist['tracks']['items'][0]['album']['release_date']

        artist = sp.artist(artist_uri)
        total_followers = float(artist['followers']['total']) // 1000000 
        genres = artist['genres']
        popularity = artist['popularity']

        audio_features = sp.audio_features(track_uri)[0]

        acousticness = audio_features['acousticness']
        danceability = audio_features['danceability']
        duration_ms = audio_features['duration_ms']
        energy = audio_features['energy']
        instrumentalness = audio_features['instrumentalness']
        key = audio_features['key']
        liveness = audio_features['liveness']
        loudness = audio_features['loudness']
        speechiness = audio_features['speechiness']
        tempo = audio_features['tempo']
        time_signature = audio_features['time_signature']
        valence = audio_features['valence']

        return available_markets, release_date, total_followers, genres, popularity, acousticness, danceability, duration_ms, energy, instrumentalness, key, liveness, loudness, speechiness, tempo, time_signature, valence

    except :
        return [None]*17
    

def lookup_lyrics(song):
    try :
        return api.search_song(song).lyrics
    except :
        return None
    
def clean_txt(song):
    song = ' '.join(song.split("\n"))
    song = re.sub("[\[].*?[\]]", "", song)
    return song


def len_lyrics(song):
    return len(song.split())

def len_unique_lyrics(song):
    return len(list(set(song.split())))

def rmv_stop_words(song):
    song = [w for w in song.split() if not w in stop_words] 
    return len(song)

def rmv_set_stop_words(song):
    song = [w for w in song.split() if not w in stop_words] 
    return len(list(set(song)))




def model_prediction(artist, title):
    
    df_pred = pd.DataFrame.from_dict({
        "Artist":[artist], 
        "Title":[title]})
    
    model = ""

    df_pred["Featuring"] = df_pred.apply(lambda row: featuring(row['Artist']), axis=1)
    df_pred["Artist_Feat"] = df_pred.apply(lambda row: featuring_substring(row['Artist']), axis=1)
    
    df_pred['lookup'] = df_pred['Title'] + " " + df_pred["Artist_Feat"]

    df_pred['available_markets'], df_pred['release_date'], df_pred['total_followers'], df_pred['genres'], df_pred['popularity'], df_pred['acousticness'], df_pred['danceability'], df_pred['duration_ms'], df_pred['energy'], df_pred['instrumentalness'], df_pred['key'], df_pred['liveness'], df_pred['loudness'], df_pred['speechiness'], df_pred['tempo'], df_pred['time_signature'], df_pred['valence'] = zip(*df_pred['lookup'].map(artist_info))
    
    df_pred['release_date'] = pd.to_datetime(df_pred['release_date'])
    df_pred['month_release'] = df_pred['release_date'].apply(lambda x: x.month)
    df_pred['day_release'] = df_pred['release_date'].apply(lambda x: x.day)
    df_pred['weekday_release'] = df_pred['release_date'].apply(lambda x: x.weekday())
    df_pred['lookup'] = df_pred['Title'] + " " + df_pred["Artist"]



    df_pred['lyrics'] = df_pred['lookup'].apply(lambda x: lookup_lyrics(x))
    if df_pred["lyrics"].iloc[0] != None:
        df_pred['lyrics'] = df_pred['lyrics'].apply(lambda x: clean_txt(x))
        df_pred['len_lyrics'] = df_pred['lyrics'].apply(lambda x: len_lyrics(x))
        df_pred['len_unique_lyrics'] = df_pred['lyrics'].apply(lambda x: len_unique_lyrics(x))
        df_pred['without_stop_words'] = df_pred['lyrics'].apply(lambda x: rmv_stop_words(x))
        df_pred['unique_without_stop_words'] = df_pred['lyrics'].apply(lambda x: rmv_set_stop_words(x))
        df_pred['sentimentVaderPos'] = df_pred['lyrics'].apply(lambda x: analyzer.polarity_scores(x)['pos'])
        df_pred['sentimentVaderNeg'] = df_pred['lyrics'].apply(lambda x: analyzer.polarity_scores(x)['neg'])
        df_pred['sentimentVaderComp'] = df_pred['lyrics'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
        df_pred['sentimentVaderNeu'] = df_pred['lyrics'].apply(lambda x: analyzer.polarity_scores(x)['neu'])
        df_pred['Vader'] = df_pred['sentimentVaderPos'] - df_pred['sentimentVaderNeg']
        df_pred['Title_Length'] = df_pred['Title'].apply(lambda x: len(x.split(" ")))
        model = model_with_lyrics
    else:
        model = model_without_lyrics
        df_pred = df_pred.drop(["total_followers"], axis=1)

    X = df_pred.drop(["Artist_Feat", "Artist", "Title", "lookup", "release_date", "genres", "lyrics"], axis=1).astype(float)
    y_pred = model.predict_proba(X)
    
    return y_pred




@app.route('/')
def helloworld():
    return render_template("index.html", table_data=[], prediction_data=[[0], [0]])

@app.route('/getSongPrediction', methods =['POST'])
def search():
    name = request.form["name"]
    out = sp.search(name)
    table = []
    for x in range(len(out["tracks"]["items"])):
        table.append({"index": x, "name": out["tracks"]["items"][x]["name"], "artist": out["tracks"]["items"][x]["album"]["artists"][0]["name"], "date": out["tracks"]["items"][x]["album"]['release_date']})

    return render_template("index.html", table_data=table, prediction_data=[[0], [0]])



@app.route('/predict', methods =['POST', 'GET'])
def predict():
    name = request.args.get('name')
    artist = request.args.get('artist')
    ind[0] = request.args.get('index')
    arr = model_prediction(artist=artist, title=name)
    return render_template("index.html", table_data=[], prediction_data=arr)

if __name__ == '__main__' :
    app.run(debug= False,port=8080)