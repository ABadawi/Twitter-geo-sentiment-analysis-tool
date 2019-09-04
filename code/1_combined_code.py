
                                           
                                            ## Modules & libraries:
import module_manager
module_manager.review()
##


import numpy as np
import scipy
import matplotlib.pyplot as plt 
from textblob import TextBlob
import sys 
import random 
import IPython 
import operator

## 
from geotext import GeoText
##

import tweepy           
import pandas as pd     
import numpy as np      

##
import tweepy
from tweepy import OAuthHandler
import json
import datetime as dt
import time
import os
import sys

## 
import string 
import pprint

##

import nltk
lemma = nltk.wordnet.WordNetLemmatizer()

##
from textblob.classifiers import NaiveBayesClassifier

##
from textblob import TextBlob



                                                    ## Code:



## Fetching Tweets from Twitter API:
'''
The first step in building a twitter sentiment analysis application is to create
an account and to set up a twitter API. An API can be thought of as a gateway 
from which you obtain web-based data. Twitter will assign a set of keys and 
authentication codes for each twitter API. Using those keys, one can simply 
fetch tweets with a couple of lines of code using  Python. 

'''

consumerKey = "8ucwj9obzhH73nS1mSl8LBWV4"
consumerSecret = "ypkhFP3ra7MNROJOnnqnnxvcXt39Lqw4j10DYYtxoKmIZxLlvx"
accessToken = "984806878231388160-e0A9yGVL7m78lMCcC78VHjJXZvYZmfm"
accessTokenSecret = "kXtBBgHNvpGfRicSsX40L7C0MHK8rFUuxL2OrVU6KCFvp"


'''
Tweepy is a pyhton module that fetches the tweets.
Sometime, when the rate of tweets about a certain 
topic is really high, teepy will throw a timeout
error. In order to avoid that, uncomment line 81.
For the sake of time, the function call in line
83 should suffice for a maximum number of tweets 
below 70 to 100.
'''

auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
auth.set_access_token(accessToken, accessTokenSecret)
# api = tweepy.API(auth, wait_on_rate_limit=True)
api = tweepy.API(auth)



## Setting up the limit/count of tweets:
'''
Enter the maximum number of tweets to be fethced 
'''
limit = int(input("\n Enter the maximum number of tweets: "))



## Select Search Mode: By topic vs. timeline
'''
Enter y to search tweets by topic. Or enter n, to search 
my timeline. More parameters can be added to the Tweepy cursor
to limit the search within a time range. You could also
search by city and raduis.
'''
searchByTerm = input('\n Do you want to search by topic? type: y or n: ')

##
if searchByTerm == 'y':
    searchTerm = input("\n Enter topic: ")
    
    
## Choose sentiment analysis type:
'''
User can choose between 3 sentiment analysis types:

1. Basic sentiment analysis
    
2. Dope sentiment analysis
    
    This function determines sentiment by analyzing the tweet
    word by word. It's trained with two files. One file for 
    positive sentiment words and the other for negative sent
    words. Thus this classifier is binary
 
3. Naive Bayes sentiment analysis 

    This classifier is trained using a databas of 7086 tweets
'''
analysisType = input('\n Enter sentiment anaysis type: \n 1. Basic\n 2. Dope\n 3. Naive\n  >> ' )
    


## Helper function to create a list of negative words:
'''
This function takes the negative words text file and dumps its
contents into a list to be used for sentiment analysis
'''
def createNegList():
    negL = []
    # Read txt file: 
    with open('neg.txt', encoding="utf8") as inputfile:
        
        for line in inputfile:
            
            word = line.strip()
            word = word.strip('\n')
            word = word.strip('\ufeff')
            negL.append(word)
        
        # print(negL[:3])       
        return negL
        
        

## Helper function to create a list of positive words:
'''
This function takes the positive words text file and dumps its
contents into a list to be used for sentiment analysis
'''
def createPosList():
    posL = []
    # Read txt file: 
    with open('pos.txt', encoding="utf8") as inputfile:
        
        for line in inputfile:
            
            word = line.strip()
            word = word.strip('\n')
            word = word.strip('\ufeff')
            posL.append(word)
        
        # print(posL[:3])       
        return posL




## Helper function to get the sentiment of tweet based on word by word sentiment:
'''
This classifier captures the sentiment by tweets analyzing word by word
The classification is based on the number of negative words vs. the
number of positive words in a tweet.
'''
def dopeClassify(tweet):
    
    sentiment = 0 # a tweet is neutral by default
    negL = createNegList()
    posL = createPosList()
     
    pos = 0 
    neg = 0 
    
    
    for word in str(tweet).split():
        
        if word in posL:
            pos += 1
            
        elif word in negL:
            neg += 1
    
    if pos > neg:
        sentiment = 1
        
    elif pos < neg:
        sentiment = -1
            
    
    return sentiment 
    
    
## Helper function to train naive bayes classifier:

## Create train list: ##


## Binatry data:
'''
This function creates a training list using the tweets dataset mentioned 
earlier (7086 tweets)


For training I am using a data set of tweets provided by kaggle.
The data was collected and labeled by University of Michigan
The data set contains 7086 tweets:
no. of pos tweets =  779
no. of neg tweets =  632

lables:
1 - pos
0 - neg 

The data was initailly provided in a txt. format. For my convenience, 
I am fetching the data from txt. into a list in order to use that to
train my classifiers.

'''
def createTrainlist():
    
    L = []
    # Read txt file: 
    with open('trainData.txt', encoding="utf8") as inputfile:
        
        for line in inputfile:
            
            # basic cleansing:
            text = line[2:] #remove 0 or 1
            text = text.strip() # remove space
            text = text.strip('\n') 
            text = text.strip('\t')
           
                
            # examine line by line:
            if '1' in line:
                sentiment = 1
                pair = (text, sentiment)
                L.append(pair)
                
            elif '0' in line:
                sentiment = -1
                pair = (text, sentiment)
                L.append(pair)
            
        # ## for testing purpose : 
        # 
        # # first 5 lines 
        # for pair in L[:5]:
        #     # print(pair)
        #     
        # # last 5 lines    
        # bottomFive = len(L) - 5
        # for pair in L[bottomFive:]:
        #     # print(pair)
        #     
        # # check total lenght of L:
        # # print('size of training data set =',  len(L))
        # 
        # # number of pos & neg tweets:
        S = set(L)
        
        posCount = 0 
        negCount = 0
        
        for pair in S:
            if pair[1] == 1:
                posCount += 1 
                
            elif pair[1] == -1:
                negCount += 1
                
        # print('no. of pos tweets = ', posCount)
        # print('no. of neg tweets = ', negCount)
            
        return L
        
# createTrainlist()

## Train it:

train = createTrainlist()
naive = NaiveBayesClassifier(train)




## Helper function to strip punctuations:
'''
This function performs basic cleansing of the tweet
'''
def stripPunc(s):
    
    ## str.replace(old,new) 
    
    for c in s:
        if c in set('&,<,>'):
            s = s.replace('c','') 
            # creating an empty string and only joining allowed characters 
            s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    return s
    
    
    
## Helper function to get long and lat given city name:


'''function citation: https://geopy.readthedocs.io/en/stable/

The original python code for this module geotext.py was throwing
an encoding error and thus  was slightly modified to 
suit my code. 

This function is important because only very few people choose to geo-tag 
their tweets. More people however are comfortable adding a place tag.
If a tweet is neither geo-tagged ot place-tagged, the code accesses
the free text feild title location. Since this is a free text feild,
it could contain meaningless string such as ''my parent's basement'
or 'Mars'. Therefore in this code, we first check if the text is 
a city or a country and then get the coordinates

'''

from geopy import geocoders 
from geopy.geocoders import Nominatim

def getCoor(s): 

    ## inser the username for geopy account: 
    gn = geocoders.GeoNames(username='abadawi')
    geolocator = Nominatim()
    location = geolocator.geocode(s, timeout=10) #takes the string as an input 
    
    
    # location = geolocator.geocode(my_address, timeout=10)
    
    return (location.latitude, location.longitude) 





## Main function ##
def extractTweetsCoor():
    '''
    This is the main function that loops over tweets and extrats relevant information for our
    analysis
    
    1. latitude, longitude
    2. sentiment 
    3. tweet text body
    
    '''
    
    
    ### Searching by topic: ###
    if searchByTerm == 'y':
        
        
        tweetSet = set()
        tweets = tweepy.Cursor(api.search, q=searchTerm, lang = 'en', tweet_mode='extended').items(limit)
        
        with open('tweets.json', 'w', encoding='utf8') as file:
        
            for tweet in tweets:
                tweetList = [] # resets to empty for every new tweet
                
                
                
                
                                                    ##########
                
                
                
                    
                ## Adding coordinates:
                
                ## if point location: geo tagged using GPS:
                if tweet._json["geo"] != None:
                    coor = (tweet._json["geo"]["coordinates"]) #tuple it 
                    tweetList.append(coor[0])
                    tweetList.append(coor[1])
                    
                    ## Adding sentiment:
                        
                    text = tweet._json["full_text"]
                    
                    if analysisType == 'Basic':
                        sentiment = TextBlob(text).sentiment.polarity
                        # tweetList.append(sentiment)
                        
                    elif analysisType == 'Naive':
                        sentiment = naive.classify(text)
                        # tweetList.append(sentiment)
                        
                    elif analysisType == 'Dope': 
                        sentiment = dopeClassify(text)
                        # tweetList.append(sentiment)
                        
                    tweetList.append(sentiment)
                    
                    
                    ## Adding tweet text:
                    tweetList.append(text)
                    
                            # ex: [lat, long, sentiment, text]
                    
                    
                
                ## if polygon: tagged using place id by twitter
                elif tweet._json['place'] != None:
                    # coor in bounding box are reversed
                
                
                    flippedCoor = tweet._json['place']['bounding_box']['coordinates'][0][0] 
                    #accessing coor as numbers
                    
                    coor = list(flippedCoor[::-1]) # put coord inside a list it  # not now 
                    tweetList.append(coor[0])
                    tweetList.append(coor[1])
                    
                    ## Adding sentiment:
                        
                    text = tweet._json["full_text"]
                    
                    if analysisType == 'Basic':
                        sentiment = TextBlob(text).sentiment.polarity
                        # tweetList.append(sentiment)
                        
                    elif analysisType == 'Naive':
                        sentiment = naive.classify(text)
                        # tweetList.append(sentiment)
                        
                    elif analysisType == 'Dope': 
                        sentiment = dopeClassify(text)
                        # tweetList.append(sentiment)
                    
                    tweetList.append(sentiment)
                    
                    ## Adding tweet text:
                    tweetList.append(text)
                    
                            # ex: [lat, long, sentiment, text]
                            
                            
                            
                # if none of that is available: get twitter profile general location
                elif tweet._json['user']['location'] != None or \
                'location' in tweet._json['user']:
                    
                    
                    profileLocation = tweet._json['user']['location']
                    
                    if GeoText(profileLocation).cities != []: 
                        profileLocation = GeoText(profileLocation).cities[0]
                        # print(profileLocation)
                        coor = getCoor(profileLocation)
                        tweetList.append(coor[0])
                        tweetList.append(coor[1])
                    
                    
                    
                    elif GeoText(profileLocation).countries != []: 
                        profileLocation = GeoText(profileLocation).countries[0]
                        # print(profileLocation)
                        coor = getCoor(profileLocation)
                        tweetList.append(coor[0])
                        tweetList.append(coor[1])
                        
                    else: continue 
                        
                        
                    
                    ## Adding sentiment:
                        
                    text = tweet._json["full_text"]
                    
                    if analysisType == 'Basic':
                        sentiment = TextBlob(text).sentiment.polarity
                        # tweetList.append(sentiment)
                        
                    elif analysisType == 'Naive':
                        sentiment = naive.classify(text)
                        # tweetList.append(sentiment)
                        
                    elif analysisType == 'Dope': 
                        sentiment = dopeClassify(text)
                        # tweetList.append(sentiment)
                    
                    
                    tweetList.append(sentiment)
                    
                    ## Adding tweet text:
                    tweetList.append(text)
                    
                            # ex: [lat, long, sentiment, text]
                            
                        
            
                
                else: continue
                # only considering tweets with coor
                # only if tweet has coor >> get sentiment 
            
                tweetTuple = tuple(tweetList)
                tweetSet.add(tweetTuple)
            
                
                    
                    
                ## Dump tweets into json:    
                json.dump(tweet._json , file, indent=4, sort_keys=True)
                
            
            
            # print('final list = ', tweetSet)     
            return tweetSet
                    
                    
                    
                    
                                                    ####### 
                                                    
             
                        
                    
                    
                    
                                                    #######
              
    
    
    
    
    
    
    
    
    ### Searching my timeline: ###    
    else: 
        tweetSet = set()
        timeLineCoorSet = set() # a list to store tweets coor
        timeLineTweets = tweepy.Cursor(api.user_timeline, tweet_mode='extended').items(limit)
        
        with open('timeLineTweets.json', 'w', encoding='utf8') as file:
        
            for tweet in timeLineTweets:
                tweetList = []
                posNumber = 0
                negNumber = 0
                
                ## Adding coordinates:
                
                ## if point location: geo tagged using GPS:
                if tweet._json["geo"] != None:
                    coor = (tweet._json["geo"]["coordinates"]) #tuple it 
                    tweetList.append(coor[0])
                    tweetList.append(coor[1])
                    
                    ## Adding sentiment:
                        
                    
                    text = tweet._json["full_text"]
                    
                    if analysisType == 'Basic':
                        sentiment = TextBlob(text).sentiment.polarity
                        # tweetList.append(sentiment)
                        
                    elif analysisType == 'Naive':
                        sentiment = naive.classify(text)
                        # tweetList.append(sentiment)
                        
                    elif analysisType == 'Dope': 
                        sentiment = dopeClassify(text)
                        # tweetList.append(sentiment)
                   
                    tweetList.append(sentiment)
                    
                    ## Adding tweet text:
                    tweetList.append(text)
                    
                            # ex: [lat, long, sentiment, text]
                     
                    
                
                ## if polygon: tagged using place id by twitter
                elif tweet._json['place'] != None:
                    # coor in bounding box are reversed
                   
                   
                    flippedCoor = tweet._json['place']['bounding_box']['coordinates'][0][0] 
                    #accessing coor as numbers
                    
                    coor = list(flippedCoor[::-1]) # put coord inside a list it  # not now 
                    tweetList.append(coor[0])
                    tweetList.append(coor[1])
                    
                    ## Adding sentiment:
                        
                        
                    text = tweet._json["full_text"]
                    
                    if analysisType == 'Basic':
                        sentiment = TextBlob(text).sentiment.polarity
                        # tweetList.append(sentiment)
                        
                    elif analysisType == 'Naive':
                        sentiment = naive.classify(text)
                        # tweetList.append(sentiment)
                        
                    elif analysisType == 'Dope': 
                        sentiment = dopeClassify(text)
                        # tweetList.append(sentiment)
                    
                    
                    tweetList.append(sentiment)
                    
                    ## Adding tweet text:
                    tweetList.append(text)
                    
                            # ex: [lat, long, sentiment, text]
                        
                        
                else: continue
                # only considering tweets with coor
                # only if tweet has coor >> get sentiment 
            
            
                tweetTuple = tuple(tweetList)
                tweetSet.add(tweetTuple)
                        
                 
    
                ## Dump tweets into json"
                json.dump(tweet._json , file, indent=4, sort_keys=True)
                
            
            # print('final list = ', tweetSet) 
            return tweetSet
            
            
extractTweetsCoor()
            

            
                            ################ VISULAIZATION ################

import plotly 
import plotly.plotly as py
import pandas as pd
import numpy as np
import plotly.tools as tls 


##


import plotly.plotly as py
import pandas as pd


### Importing tweets from file: Tweet Extraction ###


# Assign returns from tweetsExtraction to a new variable 
setOfCoor = extractTweetsCoor()


'''
This function takes a set of tuple coordinates and returns a dictionary with 
two keys: long & lat (for longitude and latitude respectively). It's responsible 
for converting the output of the previous function (fectching function) into
a format that suits visualization function
'''
def listToDict(S):
    
    # create empty dict 
    d = dict()
    
    # initiate the features of the dict as two empty lists 
    d['latitude'] = []
    d['longitude'] = []
    d['sentiment'] = []
    d['tweet'] = []
    
    
    for pair in S:
        
        # populate the lists
        d['latitude'].append(pair[0])
        d['longitude'].append(pair[1])
        d['sentiment'].append(pair[2])
        d['tweet'].append(pair[3])
        
    return d 
    
# print(listToDict(setOfCoor))
   


def visulaize():
    '''
    This function visulaizes tweets by sentiment and location on map
    '''
    
    ## Enter projection type:
    visulaizeType = input('\n Enter visulaization/projection type:  \n 1. Mercator \n 2. orthographic \n 3. melloweide \n >>  ')
    
    ## Enter map's scope:
    scope = input('\n Zoom in to a conteninent of choice:  \n 1. world \n 2. africa \n 3. asia \n 4. north america \n 5. south america \n 6. europe \n 7. usa \n >>  ')
    
    ## Visulaization dictionary:
    df = listToDict(setOfCoor)
    
    
    
    ## symbol type:
    symbol = input('\n Enter symbol shape: \n 1. circle \n 2. square \n >>  ')
    
   
    scl = [[-1,"r"],[1,"dodgerblue"]]
    
    
    df['text'] = df['sentiment'] 
    
    data = [ dict(
            type = 'scattergeo',
            locationmode = 'USA-states',
            lon = df['longitude'],
            lat = df['latitude'],
            # text = str(df['tweet']) + ' ' + str(df['sentiment']),
            text = df['text'],
            mode = 'markers',
            marker = dict(
                size = 10,
                opacity = 1,
                reversescale = True,
                autocolorscale = True,
                symbol = symbol,
                colorscale = scl,
                cmin = -1,
                color = df['sentiment'],
                cmax = 1,
                colorbar=dict(
                title="Sentiment polarity"),
                line = dict(
                    width=2,
                    color='black' # color of the outline of the circle 
                )
                
                
            ))]
    
    if searchByTerm == 'y':
    
        layout = dict(
                title = 'How the' + ' ' + str(scope) + ' ' + 'feels about' + ' ' + str(searchTerm),
                geo = dict(
                    scope=scope,
                    projection=dict(type=visulaizeType),
                    showland = True,
                    landcolor = "rgb(250, 250, 250)", #"gainsboro", # green 121, 190, 37
                    showcountries = True,
                    subunitcolor = "lime" , # land border
                    countrycolor = "rgb(150, 150, 150)", #"black", #borders between countries 
                    countrywidth = 0.6,
                    subunitwidth = 0.6
                ),
            )
            
    else:
        
        layout = dict(
                title = 'My timeline sentiment',
                geo = dict(
                    scope=scope,
                    projection=dict(type=visulaizeType),
                    showland = True,
                    landcolor = "rgb(250, 250, 250)",
                    showcountries = True,
                    subunitcolor = "lime" , # land border
                    countrycolor = "rgb(150, 150, 150)", #"black", #borders between countries 
                    countrywidth = 0.6,
                    subunitwidth = 0.6
                ),
            )
        
        
        
    plotly.offline.plot({"data":data, "layout":layout},filename = 'insertNameHere2_map.html')
    
    
visulaize()



## Bar Chart:
'''
This function visualizes the sentiment of the tweets using bar charts
'''

from tkinter import *

def getPercnetage():
    
    pos = 0
    neg = 0
    neu = 0
    
    tweetSet = extractTweetsCoor()
    
    for tweet in tweetSet:
        sentiment = tweet[2]
        if  sentiment > 0:
            pos += 1 
            
        elif sentiment < 0:
            neg += 1 
            
        else: 
            neu += 1
            
    return (pos, neg, neu)
    

def draw(canvas, width, height):
    
    result = getPercnetage()
    
    pos = result[0]
   
    neg = result[1]
    
    neu = result[2] 
    
    all = pos + neg + neu
    
    posPercent = pos/all
    negPercent = neg/all
    neuPercent = neu/all
    
    strPosPercent = str(round(posPercent*100,2))
    strNegPercent = str(round(negPercent*100,2))
    strNeuPercent = str(round(neuPercent*100,2))
    
    factorPos = posPercent
    # print('pos fac', factorPos)
    factorNeg = negPercent
    # print('neg fac', factorNeg)
    factorNeu = neuPercent
    
    # Blue bar 
    canvas.create_rectangle(0, height//4 - 50, width*factorPos, height//4 + 50,
                    fill='blue')
                    
    # Red bar 
    canvas.create_rectangle(0, 2*height//4 - 50, width*factorNeg, 2*height//4 + 50,
                    fill='red')
                    
    
    #grey bar 
    canvas.create_rectangle(0, 3*height//4 - 50, width*factorNeu, 3*height//4 + 50,
                    fill='grey')
                    
    
    # blue label                
    canvas.create_rectangle(1*width//4, height - 50, 1*width//4+25, height - 20,
                    fill='blue')
     
    # blue text                 
    canvas.create_text(1*width//4, height - 10, text = 'Positive' + ' ' + strPosPercent + '%')
    
    
    
    
    # red label
    canvas.create_rectangle(2*width//4, height - 50, 2*width//4+25, height - 20,
                    fill='red')
    
    # red text                
    canvas.create_text(2*width//4, height - 10, text = 'Negative' + ' ' + strNegPercent + '%')
    
    # grey label
    canvas.create_rectangle(3*width//4, height - 50, 3*width//4+25, height - 20,
                    fill='grey')
    
    # grey text                
    canvas.create_text(3*width//4, height - 10, text = 'Neutral' + ' ' + strNeuPercent + '%')
    
    # Title               
    canvas.create_text(250, 50, text = 'Tweet Sentiment Distribution', font='20')
    
'''    
def draw2(canvas, width, height):
    
    result = getPercnetage()
    
    pos = result[0]
    # print('pos', pos)
    neg = result[1]
    # print('neg', neg) 
    
    all = pos + neg
    # print('all', all)
    
    # round(a, 2)
    # 
    posPercent = pos/all
    negPercent = neg/all
    
    strPosPercent = str(round(posPercent*100,2))
    strNegPercent = str(round(negPercent*100,2))
    
    factorPos = posPercent
    # print('pos fac', factorPos)
    factorNeg = negPercent
    
    # Title               
    canvas.create_text(250, 20, text = 'Tweet Sentiment Distribution', font='18')
    
    
    # pos circle:
    canvas.create_oval(100, 100, 100+pos*10, 100+pos*10, fill="blue")
    
    # neg circle:
    canvas.create_oval(width - 200, height-200, width - 200+neg*10, width - 200+neg*10, fill="red")
    
    
## countries of tweets ???     
    
'''
def runDrawing(width=300, height=300):
    root = Tk()
    canvas = Canvas(root, width=width, height=height)
    canvas.pack()
    draw(canvas, width, height)
    # draw2(canvas, width, height)
    root.mainloop()
    print("bye!")

runDrawing(500, 500)
'''
def runDrawing2(width=300, height=300):
    root = Tk()
    canvas = Canvas(root, width=width, height=height)
    canvas.pack()
    # draw(canvas, width, height)
    draw2(canvas, width, height)
    root.mainloop()
    print("bye!")

runDrawing2(500, 500)
'''    
'''    
def mostFreq():
    
    
    tweetSet = extractTweetsCoor()
    
    d = dict()
    
    for tweet in tweetSet[:10]:
        text = tweet[3]
        text = stripPunc(text)
        for word in text.split():
            
            if word not in d:
                d[word] = 1
                
            elif word in d:
                d[word] += 1
    

    sorted_d = sorted(d.items(), key=operator.itemgetter(1))
        
                
    print(print(sorted_d[len(sorted_d)-5:]))             
    return sorted_d[len(sorted_d)-5:]
    
mostFreq()
    
   
def drawPieChart():
    
    
    
    d = mostFreq()
    
    #create labels:
    labels = []
    for key in d:
        lables.append(key)
        
    # create sizes:
    for key in d:
        sizes.append(d[key])
    
    
    # labels = ['Cookies', 'Jellybean', 'Milkshake', 'Cheesecake']
    # sizes = [38.4, 40.6, 20.7, 10.3]
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
    patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=90)
    plt.legend(patches, labels, loc="best")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    
drawPieChart()
                
'''
    


    
    
    
























