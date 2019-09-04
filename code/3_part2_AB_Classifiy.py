
                                           
                                           
                            ################ FECTCHING ################
                                           
                                           
                                           
                                           
                                            ## Modules & libraries:
import module_manager
module_manager.review()
##


import numpy as np
import sklearn
import scipy
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt 
import pandas as pd  
from textblob import TextBlob
import sys 
import nltk 
import random 
from nltk.corpus import movie_reviews
import seaborn 
import csv
import IPython 

## 
from geotext import GeoText

##

import tweepy           
import pandas as pd     
import numpy as np      


from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns


##
import tweepy
from tweepy import OAuthHandler
import json
import datetime as dt
import time
import os
import sys
import preprocessor 

## 
import string 
import pprint

##

from textblob import TextBlob

##
# $ pip3 install newspaper3k
# import geograpy



##

# use keys as variables
import credentials
from credentials import * 

# print('Installation Complete!')

                                                    ## References:














                                                    ## Code:

## Good Idea:

'''
add to a dict instead of appending to a list >> faster

'''



## USE MORE ADVANCED SEARCH parameteres: like range in time
'''
tweepy.Cursor(api.search,  
              q="Giraffes",
              since="2015-10-10", 
              until="2015-10-11",
              count=100).items())

'''

## Fetching Tweets from Twitter API:
consumerKey = "8ucwj9obzhH73nS1mSl8LBWV4"
consumerSecret = "ypkhFP3ra7MNROJOnnqnnxvcXt39Lqw4j10DYYtxoKmIZxLlvx"
accessToken = "984806878231388160-e0A9yGVL7m78lMCcC78VHjJXZvYZmfm"
accessTokenSecret = "kXtBBgHNvpGfRicSsX40L7C0MHK8rFUuxL2OrVU6KCFvp"

auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
auth.set_access_token(accessToken, accessTokenSecret)
# api = tweepy.API(auth, wait_on_rate_limit=True)
api = tweepy.API(auth)

## Helper function to create a list of negative words:

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

def abGetSent(tweet):
    
    sentiment = 0 # a tweet is neutral by default
    negL = createNegList()
    # print(negL)
    posL = createPosList()
     
    pos = 0 
    neg = 0 
    
    
    for word in str(tweet).split():
        
        # for each tweet:  
        
        if word in posL:
            pos += 1
            
        elif word in negL:
            neg += 1
    
    
    # print('pos', pos)
    # print('neg', neg)
            
    if pos > neg:
        sentiment = 1
        
    elif pos < neg:
        sentiment = -1
            
    
    return sentiment 




## Helper function to strip punctuations:
def stripPunc(s):
    
    ## str.replace(old,new) 
    
    for c in s:
        if c in set('&,<,>'):
            s = s.replace('c','') 
            # creating an empty string and only joining allowed characters 
            s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    return s
    
    
    
## Helper function to get long and lat given city name:
'''This function takes a string of a city's name and returns long and lat as a tuple'''

'''citation: https://geopy.readthedocs.io/en/stable/'''

from geopy import geocoders 
from geopy.geocoders import Nominatim

def getCoor(s): 

    ## inser the username for geopy account: 
    gn = geocoders.GeoNames(username='abadawi')
    geolocator = Nominatim()
    location = geolocator.geocode(s) #takes the string as an input 
    
    return (location.latitude, location.longitude) 



'''
This way, this file is responsible for extracting tweet coor.
Similarly, with slight modification, we can have a similar file 
to do each of the following:

1. sentiment using text blob
1. sentiment using advanced classifier 
2. collect and present in a table
4. text cloud  
3. pie chart



'''

## SIDE NOTE  
'''
1. Twitter point (GPS) 
2. Twitter place (polygon) - place_type should not be country. 
Unless your goal is to track which country    the tweet is from
as opposed to which city 
''' 

## Setting up the limit/count of tweets:
limit = int(input("\n Enter the maximum number of tweets: "))



## Select Search Mode: By topic vs. timeline
searchByTerm = input('\n Do you want to search by topic? type: y or n: ')

##
if searchByTerm == 'y':
    searchTerm = input("\n Enter topic: ")
    
    


## Main function ##
def extractTweetsCoor():
    
    
    ### Searching by topic: ###
    if searchByTerm == 'y':
        
        
        tweetSet = set()
        # tweets = tweepy.Cursor(api.search, q=searchTerm, since="2018-4-24", until="2018-4-28", tweet_mode='extended').items(limit)
        
        # Since we are doing sentiment analysis we are limiting lang option to ENG:
        # this limits the no. of tweets even further 
        # in a different file we just plot no sent
        
        tweets = tweepy.Cursor(api.search, q=searchTerm, lang = 'en', tweet_mode='extended').items(limit)
        
        with open('tweets.json', 'w', encoding='utf8') as file:
        
            for tweet in tweets:
                tweetList = [] # resets to empty for every new tweet
                
                ## Adding coordinates:
                
                ## if point location: geo tagged using GPS:
                if tweet._json["geo"] != None:
                    coor = (tweet._json["geo"]["coordinates"]) #tuple it 
                    tweetList.append(coor[0])
                    tweetList.append(coor[1])
                    
                    ## Adding sentiment:
                        
                    text = tweet._json["full_text"]
                    # sentiment = TextBlob(text).sentiment.polarity
                    sentiment = abGetSent(text)
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
                    sentiment = abGetSent(text)
                    # sentiment = TextBlob(text).sentiment.polarity
                    tweetList.append(sentiment)
                    
                    ## Adding tweet text:
                    tweetList.append(text)
                    
                            # ex: [lat, long, sentiment, text]
                            
                            
                            
                # if none of that is available: get twitter profile general location
                elif tweet._json['user']['location'] != None or \
                'location' in tweet._json['user']:
                    
                    
                    profileLocation = tweet._json['user']['location']
                    # print(profileLocation)
                    
                    if GeoText(profileLocation).cities != []: 
                        profileLocation = GeoText(profileLocation).cities[0]
                        coor = getCoor(profileLocation)
                        tweetList.append(coor[0])
                        tweetList.append(coor[1])
                    
                    
                    
                    elif GeoText(profileLocation).countries != []: 
                        profileLocation = GeoText(profileLocation).countries[0]
                        coor = getCoor(profileLocation)
                        tweetList.append(coor[0])
                        tweetList.append(coor[1])
                        
                    else: continue 
                        
                        
                    
                    ## Adding sentiment:
                        
                    text = tweet._json["full_text"]
                    sentiment = abGetSent(text)
                    # sentiment = TextBlob(text).sentiment.polarity
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
                
            
            
            print('final list = ', tweetSet)     
            return tweetSet
                    
            
    
    
    
    
    
    
    
    
    
    
    ### Searching my timeline: ###    
    else: 
        tweetSet = set()
        timeLineCoorSet = set() # a list to store tweets coor
        timeLineTweets = tweepy.Cursor(api.user_timeline, tweet_mode='extended').items(limit)
        
        with open('timeLineTweets.json', 'w', encoding='utf8') as file:
        
            for tweet in timeLineTweets:
                tweetList = []
                
                ## Adding coordinates:
                
                ## if point location: geo tagged using GPS:
                if tweet._json["geo"] != None:
                    coor = (tweet._json["geo"]["coordinates"]) #tuple it 
                    tweetList.append(coor[0])
                    tweetList.append(coor[1])
                    
                    ## Adding sentiment:
                        
                    
                    text = tweet._json["full_text"]
                    sentiment = abGetSent(text)
                    # sentiment = TextBlob(text).sentiment.polarity
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
                    sentiment = abGetSent(text)
                    # sentiment = TextBlob(text).sentiment.polarity
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

# api key:
# A1JcTlz2Xwt5oaRc0Yz2

# username:
# abadawi

##

# plotly.tools.set_credentials_file(username='abadawi', api_key='A1JcTlz2Xwt5oaRc0Yz2')
import plotly.plotly as py
import pandas as pd


### Importing tweets from file: Tweet Extraction ###

# from TRY_add_text_pol_to_list_file_2_extract_polarity_en import *
# from TweetsExtraction_TheRealDeal_3 import extractTweetsCoor

# Assign returns from tweetsExtraction to a new variable 
setOfCoor = extractTweetsCoor()
# lisrOfCoor is a list 
# we want to convert it to a dict

'''
This function takes a set of tuple coordinates and returns a dictionary with 
two keys: long & lat (for longitude and latitude respectively) 
'''


## VERY GOOD IDEA
'''
In order to make the text appear on the map, we could 
add the tweet text to this dict and visulaize it 
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
   

'''
Instead of reading from a csv file and accessing the right cells. We could turn 
our coordinates lists we extracted into dictionaries with keys as:
1. long for longitute
2. lat for latitude 
'''




# df = pd.read_csv('C:\\Users\\ahmed\\Desktop\\15112\\TP 112\\TP1\\TP2 Deliverables\\Working demo 100 pts\\cities.csv')
# df.head()

def visulaize():
    
    ## Enter projection type:
    visulaizeType = input('\n Enter visulaization/projection type:  \n 1. Mercator \n 2. orthographic \n 3. melloweide \n >>  ')
    
    ## Enter map's scope:
    scope = input('\n Zoom in to a conteninent of choice:  \n 1. world \n 2. africa \n 3. asia \n 4. north america \n 5. south america \n 6. europe \n 7. usa \n >>  ')
    
    ## Visulaization dictionary:
    df = listToDict(setOfCoor)
    
    
    ## Data type:
    # scatter geo and others????
    
    
    
    ## symbol type:
    symbol = input('\n Enter symbol shape: \n 1. square \n 2. circle \n >>  ')
    
    
    
    # df['text'] = df['airport'] + '' + df['city'] + ', ' + df['state'] + '' + 'Arrivals: ' + df['cnt'].astype(str)
    # 
    # scl = [ [0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
    #     [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"] ]
    
    
    ## WE COULD LOOK INTO CHOROPLETH LATER 
    ## type = 'scattergeo', type = 'scattergeo',choropleth
    
    
    # rgb(255, 51, 0) RED
    # rgb(255, 153, 51) orange 
    # rgb(255, 204, 102) washed orange 
    # rgb(255, 255, 153) almost white orange 
    # rgb(204, 255, 153)
    # rgb(153, 255, 153)
    # rgb(102, 255, 153)
    # rgb(0, 255, 153)
    # 
    # Green rgb(50,205,50)
    
    # scl = [ [-1,"rgb(255, 51, 0)"],[-0.75,"rgb(255, 153, 51)"],[-0.5,"rgb(255, 204, 102)"],\
    # [-0.25,"rgb(255, 255, 153)"], [0,"rgb(204, 255, 153)"],[0.25,"rgb(153, 255, 153)"],       [0.75,"rgb(102, 255, 153)"],[1,"rgb(0, 255, 153)"]]
    
    # scl = [ [-1,"r"],[-0.75,"darkorange"],[-0.5,"gold"],[-0.25,"khaki"], [0,"w"],[0.25,"greenyellow"],[0.5,"springgreen"], [0.75,"cyan"],[1,"dodgerblue"]]
    
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
                
                # color = df['cnt'],
                # cmax = df['cnt'].max(),
                # colorbar=dict(
                #     title="Incoming flightsFebruary 2011"
                
            ))]
    
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
        
        # 'albers usa'
        # type='Mercator'
        
        ## 1st: usa 
        #scope = usa
        #type = albers usa 
        
        ## 2nd: world
        #scope = does not matter 
        # type = Mercator 
        
        ## 3rd: Africa
        # scope = 'africa', europe, north america, asia, south america 
        # type = Mercator 
        
        
        
        
        ## you can add two plots (enlarged part) by adding 
    
    plotly.offline.plot({"data":data, "layout":layout},filename = 'insertNameHere2_map.html')
    
    
visulaize()

    
    
    
























