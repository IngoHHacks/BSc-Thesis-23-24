import numpy as np
import globals.consts as g
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, ndcg_score
from sklearn.preprocessing import MultiLabelBinarizer

import tools.games as games
import sample_collection.steamcsv as sc

SEED = g.SEED

def train_knn(k=10, use='both', vectorizer=None, vectorizer_name=''):
    # Rewrite these
    pass

def train_knn_collab(k=10):
    # Rewrite these
    pass