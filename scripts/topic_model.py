
# !pip install ripser
# !pip install persim
# !pip install pyshp
# !pip install kde-gpu
# !pip install pyLDAvis#==2.1.2
# !pip install --upgrade pandas==1.3
# !pip install nltk
# !pip install gensim

# !pip install spacy
# !pip install pyLDAvis

# !python -m spacy download en_core_web_sm

import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

from pathlib import Path
import tarfile
import json
import tempfile

import nltk; nltk.download('stopwords')
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim
# from pyLDAvis import gensim

import spacy

def getData(data_path, exp_date):
    writedir = tempfile.TemporaryDirectory()
    writedir.name

    for fil in os.listdir(data_path + exp_date):
        tar = tarfile.open(data_path + exp_date + fil, 'r:gz')
        tar.extractall(writedir.name)
        tar.close()

    twitter_data = []
    temp_path = writedir.name + '/data/' + exp_date
    for file in os.listdir(temp_path):
        with open(temp_path+file, "r", encoding="utf8") as f:
            [twitter_data.append(json.loads(line)) for line in f.readlines()]

    twitter_df = pd.DataFrame(twitter_data)
    return twitter_df

# Util functions for cleaning
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    # NLTK Stop words
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts, bigram_mod):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts, trigram_mod, bigram_mod):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def cleanData(df):
    # Enable logging for gensim - optional
    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

    import warnings
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import regex as re

    link_regex = "(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)"

    data = df.text.values.tolist()
    print(f"Sample tweets: {data[:5]}")

    data = [re.sub(link_regex, "", sent) for sent in data]


    data_words = list(sent_to_words(data))
    print(data_words[:1])
    
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # See trigram example
    print(trigram_mod[bigram_mod[data_words[0]]])

    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops, bigram_mod)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    # Obsolete
#     nlp = spacy.load('en', disable=['parser', 'ner'])

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    print(data_lemmatized[:1])
    
    return data_lemmatized

# Building model and extracting topics
def buildLDA(data_lemmatized, k, passes=10, alpha='auto'):
    # Build LDA model
    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    # View
    print(corpus[:1])

    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=k, 
                                               random_state=100,
                                               update_every=1,
                                               chunksize=100,
                                               passes=10,
                                               alpha='auto',
                                               per_word_topics=True)
    
    from pprint import pprint
    # Print the Keyword in the 10 topics
    pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]
    
    # Compute Perplexity
    print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)
    
    return (id2word, corpus, lda_model)
    
def get_doc_topic_dists(lda_model):
    topic_dists = []
    for (doc, dists) in lda_model.print_topics():
        dist_values = [float(value.split("*")[0]) for value in dists.split(' + ')]
        topic_dists.append(dist_values)
    return topic_dists