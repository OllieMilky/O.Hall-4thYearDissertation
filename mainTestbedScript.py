import aspect_based_sentiment_analysis as absa
from os import listdir
from os.path import isfile, join
import nltk
import numpy as np
import tensorflow as tf
import time
import sys
import matplotlib.pyplot as plt
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm
import tensorflow_text

lemmatizer = WordNetLemmatizer()

def VADERSentiment(searchWord, chunker):
    f = open("AllEpisodeReviews.txt", encoding="utf8")
    text = f.read()
    reviews = text.split("/////")
    reviews = reviews[1:]
    scores = []
    location = []
    searchSentences = []
    sid = SentimentIntensityAnalyzer()
    start_time = time.time()

    for review in reviews:
        identifier = review[:20]
        season = identifier[7:9]
        episode = identifier[18:20]
        if 'A' in episode:
            episode = episode.replace('A', ' ')
        if 'A' in season:
            season = season.replace('A', ' ')
        reviewSentences = nltk.sent_tokenize(review)
        searchSentences = [
            sent for sent in reviewSentences if searchWord in word_tokenize(sent.lower())]
        for sentence in searchSentences:
            location.append((((int(season)) * 24) + int(episode))/25)
            ss = sid.polarity_scores(sentence)
            scores.append(ss['compound']*10)

    scoreaverages = []
    locationaverages = []
    scores = [round(x) for x in scores]
    chunks = [scores[x:x+chunker] for x in range(0, len(scores), chunker)]
    for chunk in chunks:
        scoreaverages.append((sum(chunk)/len(chunk))*2)

    locationchunks = [location[x:x+chunker]
                      for x in range(0, len(location), chunker)]
    for chunk in locationchunks:
        locationaverages.append(sum(chunk)/len(chunk))
    print("Number of sentences evaluated: ")
    print(len(scores))
    print(location)
    final_time = time.time() - start_time
    print("Time taken: "+str(final_time))

    return locationaverages, scoreaverages

def kerasSentiment(searchWord, chunker):
    f = open('AllEpisodeReviews.txt', encoding="utf8")
    text = f.read()
    reviews = text.split("/////")
    reviews = reviews[1:]
    scores = []
    location = []
    searchSentences = []
    model = tf.keras.models.load_model('kerasModel')
    start_time = time.time()

    for review in reviews:
        identifier = review[:20]
        season = identifier[7:9]
        episode = identifier[18:20]
        if 'A' in episode:
            episode = episode.replace('A', ' ')
        if 'A' in season:
            season = season.replace('A', ' ')
        reviewSentences = nltk.sent_tokenize(review)
        searchSentences = [
            sent for sent in reviewSentences if searchWord in word_tokenize(sent.lower())]
        for sentence in searchSentences:
            location.append((((int(season)) * 24) + int(episode))/25)
            prediction = model.predict(np.array([sentence]))
            if float(prediction[0])>1:
                prediction[0] = 1
            scores.append(int(prediction[0]*10))

    scoreaverages = []
    locationaverages = []
    scores = [round(x) for x in scores]
    chunks = [scores[x:x+chunker] for x in range(0, len(scores), chunker)]
    for chunk in chunks:
        scoreaverages.append((sum(chunk)/len(chunk))*2)

    locationchunks = [location[x:x+chunker]
                      for x in range(0, len(location), chunker)]
    for chunk in locationchunks:
        locationaverages.append(sum(chunk)/len(chunk))
    print("Number of sentences evaluated: ")
    print(len(scores))
    final_time = time.time() - start_time
    print("Time taken: "+str(final_time))

    return locationaverages, scoreaverages

def absaSentiment(searchWord, chunker):
    f = open('AllEpisodeReviews.txt', encoding="utf8")
    text = f.read()
    reviews = text.split("/////")
    reviews = reviews[1:]
    scores = []
    location = []
    searchSentences = []
    nlp = absa.load()
    start_time = time.time()

    for review in tqdm(reviews):
        identifier = review[:20]
        season = identifier[7:9]
        episode = identifier[18:20]
        if 'A' in episode:
            episode = episode.replace('A', ' ')
        if 'A' in season:
            season = season.replace('A', ' ')
        reviewSentences = nltk.sent_tokenize(review)
        searchSentences = [
            sent for sent in reviewSentences if searchWord in word_tokenize(sent.lower())]
        for sentence in searchSentences:
            location.append((((int(season)) * 24) + int(episode))/25)
            prediction = nlp(text=sentence, aspects=[searchWord])
            entityScore = prediction.examples
            valueScores = entityScore[0].scores

        ### Score aggregation methods

            neu = valueScores[0]
            neg = valueScores[1]
            pos = valueScores[2]

            finalScore = ((pos - neg) / (1 - neu))
            scores.append((finalScore*10))

    scoreaverages = []
    locationaverages = []
    scores = [round(x) for x in scores]
    chunks = [scores[x:x+chunker] for x in range(0, len(scores), chunker)]
    for chunk in chunks:
        scoreaverages.append((sum(chunk)/len(chunk))*2)

    locationchunks = [location[x:x+chunker]
                      for x in range(0, len(location), chunker)]
    for chunk in locationchunks:
        locationaverages.append(sum(chunk)/len(chunk))
    print("Number of sentences evaluated: ")
    print(len(scores))
    print(location)
    final_time = time.time() - start_time
    print("Time taken: "+str(final_time))

    return locationaverages, scoreaverages

def VADERStats():
    sid = SentimentIntensityAnalyzer()
    posCounter = 0
    negCounter = 0
    positiveFiles = ['positiveReviews/' + f for f in listdir('positiveReviews/') if isfile(join('positiveReviews/', f))]
    negativeFiles = ['negativeReviews/' + f for f in listdir('negativeReviews/') if isfile(join('negativeReviews/', f))]
    for filename in tqdm(positiveFiles):
        f = open(filename, encoding="utf8")
        text = f.read()
        ss = sid.polarity_scores(text)
        if ss['compound'] > 0:
            posCounter+=1
    for filename in tqdm(negativeFiles):
        f = open(filename, encoding="utf8")
        text = f.read()
        ss = sid.polarity_scores(text)
        if ss['compound'] < 0:
            negCounter+=1
    print('Positive Accuracy: ' +str((posCounter/12500)*100)+'%')
    print('Negative Accuracy: ' +str((negCounter/12500)*100)+'%')

def KerasStats():
    model = tf.keras.models.load_model('kerasModel')
    posCounter = 0
    negCounter = 0
    positiveFiles = ['positiveReviews/' + f for f in listdir('positiveReviews/') if isfile(join('positiveReviews/', f))]
    negativeFiles = ['negativeReviews/' + f for f in listdir('negativeReviews/') if isfile(join('negativeReviews/', f))]
    for filename in tqdm(positiveFiles):
        f = open(filename, encoding="utf8")
        text = f.read()
        prediction = model.predict(np.array([text]))
        if float(prediction[0])>1:
            prediction[0] = 1
        if int(prediction[0]) > 0:
            posCounter+=1
    for filename in tqdm(negativeFiles):
        f = open(filename, encoding="utf8")
        text = f.read()
        prediction = model.predict(np.array([text]))
        if float(prediction[0])>1:
            prediction[0] = 1
        if int(prediction[0])< 0:
            negCounter+=1
    print('Positive Accuracy: ' +str((posCounter/12500)*100)+'%')
    print('Negative Accuracy: ' +str((negCounter/12500)*100)+'%')

def ABSAStats():
    nlp = absa.load()
    posCounter = 0
    negCounter = 0
    positiveFiles = ['positiveReviews/' + f for f in listdir('positiveReviews/') if isfile(join('positiveReviews/', f))]
    negativeFiles = ['negativeReviews/' + f for f in listdir('negativeReviews/') if isfile(join('negativeReviews/', f))]
    for filename in tqdm(positiveFiles):
        f = open(filename, encoding="utf8")
        text = f.read()
        prediction = nlp(text=text, aspects=["film"])
        entityScore = prediction.examples
        valueScores = entityScore[0].scores
        neu = valueScores[0]
        neg = valueScores[1]
        pos = valueScores[2]
        finalScore = ((pos - neg) / (1 - neu))
        if finalScore > 0:
            posCounter+=1
    for filename in tqdm(negativeFiles):
        f = open(filename, encoding="utf8")
        text = f.read()
        prediction = nlp(text=text, aspects=["film"])
        entityScore = prediction.examples
        valueScores = entityScore[0].scores
        neu = valueScores[0]
        neg = valueScores[1]
        pos = valueScores[2]
        finalScore = ((pos - neg) / (1 - neu))
        if finalScore < 0:
            negCounter+=1
    print('Positive Accuracy: ' +str((posCounter/12500)*100)+'%')
    print('Negative Accuracy: ' +str((negCounter/12500)*100)+'%')

def plot(vaderx, vadery, kerasx, kerasy, absax, absay, searchWord):
    plt.plot(vaderx, vadery, color='r', label='VADER')
    plt.plot(kerasx, kerasy, color='b', label='Keras')
    plt.plot(absax, absay, color='g', label='ABSA')
    plt.xlabel("Season Number")
    plt.ylabel("Semantic Score")
    plt.title("Semantic score of "+searchWord)
    plt.xlim(1, 10)
    plt.ylim(-10, 10)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()

def menu():

    text_choice = int(input(
        '''
       Enter the number input for the text you would like to use:
       1) All Season Reviews
       2) All Episode Reviews
       '''
    ))

    user_input = int(input(
        '''
        Enter the number input for what you would like to do:
        1. VADER Sentiment Analysis
        2. Keras Sentiment Analysis
        3. ABSA Sentiment Analysis
        4. Comparative Sentiment Analysis
        5. VADER P/R
        6. Keras P/R
        7. ABSA P/R
        0. Exit
        '''
    ))

    if user_input == 1:
        searchword = str(
            input("Enter a word to evaluate from the text: ")).lower()
        chunker = int(input("Enter a chunk length: "))
        vaderx, vadery = VADERSentiment(searchword, chunker)
        plot(vaderx, vadery, 0, 0, 0, 0, searchword)
    
    if user_input == 2:
        searchword = str(
            input("Enter a word to evaluate from the text: ")).lower()
        chunker = int(input("Enter a chunk length: "))
        kerasx, kerasy = kerasSentiment(searchword, chunker)
        plot(0, 0, kerasx, kerasy, 0, 0, searchword)    
    
    if user_input == 3:
        searchword = str(
            input("Enter a word to evaluate from the text: ")).lower()
        chunker = int(input("Enter a chunk length: "))
        absax, absay = absaSentiment(searchword, chunker)
        plot(0, 0, 0, 0, absax, absay, searchword)    
        
    if user_input == 4:
        searchword = str(input("Enter a word to evaluate from the text: "))
        chunker = int(input("Enter a chunk length: "))
        vaderx, vadery = VADERSentiment(searchword, chunker)
        kerasx, kerasy = kerasSentiment(searchword, chunker)
        absax, absay = absaSentiment(searchword, chunker)
        plot(vaderx, vadery, kerasx, kerasy, absax, absay, searchword)

    if user_input == 5:
        VADERStats()
    
    if user_input == 6:
        KerasStats()

    if user_input == 7:
        ABSAStats()

    if user_input == 0:
        sys.exit(0)

while True:
    menu()
