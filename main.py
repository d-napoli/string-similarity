import sys

from fuzzywuzzy import fuzz
import statistics

# text_distance
import textdistance

# euclides
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances

def dif_between_str_lev(t1, t2):   
    result = []
    result.append( fuzz.partial_ratio( t1.lower(), t2.lower() ) )
    result.append( fuzz.ratio( t1.lower(), t2.lower() ) )
    result.append( fuzz.token_sort_ratio( t1, t2 ) )
    result.append( fuzz.token_set_ratio( t1, t2) )
    return result

def text_distance(t1, t2):
    return textdistance.hamming.normalized_similarity(t1, t2)

def euclides(t1, t2):
    corpus = [t1, t2]

    vectorizer = CountVectorizer()
    features = vectorizer.fit_transform(corpus).todense()

    for f in features:
        result = euclidean_distances(features[0], f)
    return result[0][0]

main()
