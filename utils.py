import re
import string
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

def preprocess_tweet(tweet):
    """Process tweet function.
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet

    """
    
    #lower case
    tweet = tweet.lower()
    
    #remove retweet text "RT"
    tweet = re.sub(r'RT[\s]+','',tweet)

    #remove href
    tweet = re.sub(r'https?:\/\/.*[\r\n]*','',tweet)
    
    #removing #
    tweet = re.sub(r'#', ' ', tweet)
    
    tokenizer = TweetTokenizer(preserve_case=True, strip_handles=True,reduce_len=True)
    tweet_tokens     = tokenizer.tokenize(tweet)
    
    stop_words = stopwords.words('english')
    punctuations = string.punctuation
    
    tweet_tokens_clean = []
    for word in tweet_tokens:
        if (word not in stop_words) and (word not in punctuations):
            tweet_tokens_clean.append(word)
    
    lemmatizer = WordNetLemmatizer()
    tweet_tokens_clean = []
    for word in tweet_tokens:
        tweet_tokens_clean.append(lemmatizer.lemmatize(word))            
    return tweet_tokens_clean


def build_freqs(tweets, ys):
    """Build frequencies.
    Input:
        tweets: a list of tweets
        ys: an m x 1 array with the sentiment label of each tweet
            (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its
        frequency
    """
    # Convert np array to list since zip needs an iterable.
    # The squeeze is necessary or the list ends up with one element.
    # Also note that this is just a NOP if ys is already a list.
    y_labels = np.squeeze(ys).tolist()

    # Start with an empty dictionary and populate it by looping over all tweets
    # and over all processed words in each tweet.
    freqs = {}
    for y, tweet in tqdm(zip(y_labels, tweets)):
        for word in preprocess_tweet(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs
