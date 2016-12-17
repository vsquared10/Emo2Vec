
# coding: utf-8
"""Supporting model logic for predicting emotional content of user input.
"""
import pandas as pd
import gensim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.manifold import TSNE
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

#load data for emo2vec
loc = '../NRC-Emotion-Lexicon-v0.92-Annotator-and-Sense-Level/NRC-Emotion-Lexicon-v0.92-Annotator-and-Sense-Level.txt'
print("loading & cleaning the data...")
em_words = pd.read_csv(loc, sep='\t', names=['annotator_id',
                                               'remove',
                                               'word',
                                               'joy',
                                               'sadness',
                                               'fear',
                                               'anger',
                                               'trust',
                                               'disgust',
                                               'surprise',
                                               'anticipation',
                                               'POS'])

em_words.drop('remove', axis=1, inplace=True)

em_words['word'], em_words['synonym'] = em_words['word'].str.split('--').str

em_words['toss1'], em_words['joy'] = em_words['joy'].str.split('-').str
em_words['toss2'], em_words['sadness'] = em_words['sadness'].str.split('-').str
em_words['toss3'], em_words['fear'] = em_words['fear'].str.split('-').str
em_words['toss4'], em_words['anger'] = em_words['anger'].str.split('-').str
em_words['toss5'], em_words['trust'] = em_words['trust'].str.split('-').str
em_words['toss6'], em_words['disgust'] = em_words['disgust'].str.split('-').str
em_words['toss7'], em_words['surprise'] = em_words['surprise'].str.split('-').str
em_words['toss8'], em_words['anticipation'] = em_words['anticipation'].str.split('-').str
em_words['toss9'], em_words['POS'] = em_words['POS'].str.split('-').str
em_words.head()

em_words.drop(['toss1','toss2','toss3','toss4','toss5','toss6','toss7','toss8','toss9'],
              axis=1,
              inplace=True)

new_cols = ['annotator_id',
                    'word','synonym',
                    'joy',
                    'sadness',
                    'fear',
                    'anger',
                    'trust',
                    'disgust',
                    'surprise',
                    'anticipation',
                    'POS']
em_words = em_words.reindex_axis(new_cols, axis=1)
em_words.head()

emotions = em_words[['joy',
                     'sadness',
                     'fear',
                     'anger',
                     'trust',
                     'disgust',
                     'surprise',
                     'anticipation']]

em_words[emotions.columns] = em_words[emotions.columns].apply(pd.to_numeric)

affected = em_words[emotions.columns].groupby([em_words['word']], sort=False).mean().reset_index()

# Load Google's pre-trained Word2Vec model.
print('training the word2vec model from google\'s corpus')
model = gensim.models.Word2Vec.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True)

# create list of word indicies to drop to avoid keyerrors with Google's pre-trained model.
idx_to_drop = []
def dropper():
    for ex in affected['word']:
        if ex not in model.vocab:
            idx_to_drop.append(affected.loc[affected.word == ex].index[0])

# drop words from affected that are not in google's model
dropper()
affected = affected.drop(idx_to_drop, axis=0)

G_vectors = lambda x: model[x]
affected['word_vectors'] = affected['word'].apply(G_vectors)

affected['label_vectors'] = affected[emotions.columns].values.tolist()

affected['binary'] = (affected[emotions.columns] > 0).astype(int).values.tolist()

df1 = affected[emotions.columns].rank(method='max', axis=1).rank(method='first', axis=1)
ma = df1.max().max()
affected['label'] = (df1== ma).astype(int).values.tolist()
affected['target'] = affected['label'].apply(lambda x: x.index(1))
label_dict = {0 : 'joy', 1 : 'sadness', 2 : 'fear', 3 : 'anger', 4 : 'trust', 5 : 'disgust', 6 : 'surprise', 7 : 'anticipation'}
affected['label_name'] = affected['target'].apply(lambda x: label_dict[x])

emo2vec = affected[['word_vectors', 'label_vectors', 'binary', 'label', 'target', 'label_name']]

# # Model Testing
print("splitting into train/test groups...")
emo_X, emo_y = list(emo2vec.word_vectors), list(emo2vec.target)
emo_X_train, emo_X_test, emo_y_train, emo_y_test = train_test_split(emo_X, emo_y, random_state=42)


# ### OnevsRest with LinearSVC (best score)

print("creating a model with the best stuff we've got...")
OVR = OneVsRestClassifier(LinearSVC(random_state=0), n_jobs=-1)
emo_model = OVR.fit(emo_X_train, emo_y_train)
