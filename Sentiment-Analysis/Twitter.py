#################################################################
#                  Analisis de sentimientos
#################################################################

# Autor: Marcos Catunta Cachi 

# Se realizará un análisis de la opinión que tiene la población acerca de las principales
# entidades bancarias del Perú en un rango de fechas determinado por medio de Twitter.

# Importar librerias
from tweepy import API
from tweepy import Cursor
from tweepy import OAuthHandler
from textblob import TextBlob
from googletrans import Translator

import pandas as pd
import numpy as np
import re
import json
import matplotlib.pyplot as plt

# Definir credenciales de Twitter
consumer_key = 'consumer_key'
consumer_secret = 'consumer_secret'
access_token = 'access_token'
access_token_secret = 'access_token_secret'

# Definir parametros de busqueda
start_date='2020-08-17'
end_date='2020-08-18'
entities=['BCP','SCOTIABANK','INTERBANK','BBVA']

# TWITTER client
class TwitterClient():
    def __init__(self):
        self.auth = TwitterAuthenticator().authenticate_twitter_app()
        self.twitter_client = API(self.auth)

    def get_twitter_client_api(self):
        return self.twitter_client
    
    def get_tweets(self, search, language, start_date, end_date):
        tweets=pd.DataFrame(columns=['text','date','retweet_count','like_count'])
        i=1
        for tweet in Cursor(self.twitter_client.search, q='\"{}\" -filter:retweets'.format(search),
                            tweet_mode='extended', lang=language, since=start_date, until=end_date).items():
            if 'retweeted_status' in tweet._json:
                try:
                    tweets.loc[i,'text']=tweet._json['retweeted_status'].extended_tweet['full_text']
                except AttributeError:
                    tweets.loc[i,'text']=tweet._json['retweeted_status']['full_text']
            else:
                try:
                    tweets.loc[i,'text']=tweet.extended_tweet['full_text']
                except AttributeError:
                    tweets.loc[i,'text']=tweet.full_text
            tweets.loc[i,'date']=tweet.created_at.strftime("%d-%b-%Y")
            tweets.loc[i,'retweet_count']=tweet.retweet_count
            tweets.loc[i,'like_count']=tweet.favorite_count
            i+=1            
        return tweets


# TWITTER authentication
class TwitterAuthenticator():
    def authenticate_twitter_app(self):
        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        return auth

# TWITTER analyzer
class TwitterAnalyzer:
    def CleanTweet(self, tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())
    
    def Translate(self, tweets):
        translator = Translator()
        translations = translator.translate(tweets, src='es', dest='en')
        return [i.text for i in translations]
    
    def Analyze(self, tweets):        
        polarity=[]
        for tweet in tweets:
            tweet_clear=self.CleanTweet(tweet.lower())
            tweet_analysis=TextBlob(tweet_clear)
            polarity.append(tweet_analysis.sentiment.polarity)
        return polarity
    
    def SaveTweets(self,tweets,entity):
        with open('D:/{}.json'.format(entity), 'w') as f:
            json.dump(tweets.values.tolist(), f)

# Sentiment Analyzer
class SentimentAnalyzer:
    def AnalyzeEntity(self,entity,save):
        entity_analysis={}
        twitter_client = TwitterClient()
        tweets=twitter_client.get_tweets(entity,'es', start_date, end_date)
        tweets['text_eng']=TwitterAnalyzer().Translate(tweets.text.tolist())
        tweets['polarity']=TwitterAnalyzer().Analyze(tweets.text_eng)
        
        total_good_likes,total_bad_likes,total_good_retweets,total_bad_retweets=0,0,0,0
        total_tweets_positives,total_tweets_negatives=0,0
        for index, row in tweets.iterrows():
            if row['polarity']>0:
                total_good_likes+=row.like_count
                total_good_retweets+=row.retweet_count
                total_tweets_positives+=1
            else:
                total_bad_likes+=row.like_count
                total_bad_retweets+=row.retweet_count
                total_tweets_negatives+=1
        entity_analysis['entity']=entity
        entity_analysis['count_tweets']=total_tweets_positives+total_tweets_negatives
        entity_analysis['total_good_likes']=total_good_likes
        entity_analysis['total_good_retweets']=total_good_retweets
        entity_analysis['total_tweets_positives']=total_tweets_positives
        entity_analysis['total_bad_likes']=total_bad_likes
        entity_analysis['total_bad_retweets']=total_bad_retweets
        entity_analysis['total_tweets_negatives']=total_tweets_negatives
        
        if save:
            TwitterAnalyzer().SaveTweets(tweets,entity)
        
        return entity_analysis
                
if __name__ == '__main__':

    # Obtención de datos
    sentiment_analysis=[]
    for entity in entities:
        sentiment_analysis.append(SentimentAnalyzer().AnalyzeEntity(entity,True))
    
    df_result=pd.DataFrame(sentiment_analysis)
    
    # Analisis de resultados
    
    # Cantidad de tweets por entidad
    plt.bar(df_result.entity,df_result.count_tweets)
    plt.ylabel('Tweets')
    plt.show()
    
    # Tweets positivos por entidad
    plt.bar(df_result.entity,df_result.total_tweets_positives)
    plt.ylabel('Tweets positivos')
    plt.show()
    
    # Tweets positivos y negativos por entidad
    positive=[row.total_tweets_positives/row.count_tweets for index,row in df_result.iterrows()]
    negative=[row.total_tweets_negatives/row.count_tweets for index,row in df_result.iterrows()]
    plt.bar(df_result.entity,positive,color='blue',label='positive')
    plt.bar(df_result.entity,negative,bottom=positive,color='red',label='negative')
    plt.ylabel('Porcentage')
    plt.legend()
    plt.show()
    
    # Likes positivos y negativos por entidad
    positive=[row.count_tweets/row.total_good_likes if row.total_good_likes>0 else 0 for index,row 
              in df_result.iterrows()]
    negative=[row.count_tweets/row.total_bad_likes if row.total_bad_likes>0 else 0 for index,row
              in df_result.iterrows()]

    fig, ax = plt.subplots()
    index = np.arange(df_result.entity.count())
    bar_width = 0.35
    opacity = 0.8    
    plt.bar(index, positive, bar_width, alpha=opacity, color='blue', label='positive')
    plt.bar(index + bar_width, negative, bar_width, alpha=opacity, color='red', label='negative')    
    plt.ylabel('Likes por tweet')
    plt.xticks(index + bar_width, df_result.entity)
    plt.legend()    
    plt.tight_layout()
    plt.show()
    
    # Retweets positivos y negativos por entidad    
    positive=[row.count_tweets/row.total_good_retweets if row.total_good_retweets>0 else 0 for index,row 
              in df_result.iterrows()]
    negative=[row.count_tweets/row.total_bad_retweets if row.total_bad_retweets>0 else 0 for index,row
              in df_result.iterrows()]

    fig, ax = plt.subplots()
    index = np.arange(df_result.entity.count())
    bar_width = 0.35
    opacity = 0.8    
    plt.bar(index, positive, bar_width, alpha=opacity, color='blue', label='positive')
    plt.bar(index + bar_width, negative, bar_width, alpha=opacity, color='red', label='negative')    
    plt.ylabel('Retweets por tweet')
    plt.xticks(index + bar_width, df_result.entity)
    plt.legend()    
    plt.tight_layout()
    plt.show()
    
    
   
    