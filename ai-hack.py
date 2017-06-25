#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from textblob import TextBlob
import os
import chardet
import re

import numpy as np
from scipy import stats, integrate
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(color_codes=True)

fake_news = pd.read_csv('fake.csv',encoding='utf-8')

def count_char(x):
    if isinstance(x,int):
        x = str(x)
    if isinstance(x,float):
        x = str(x)
    return sum(1 for c in x if c.isupper())

paths = []
for root, dirs, files in os.walk(".", topdown=False): 
    for name in files:
        path = (os.path.join(root, name))
        if 'bbc' in path and path.endswith('.txt'):
            paths.append(path)
bias_df = fake_news.loc[fake_news['type'] == 'bias'].copy()
bs_df = fake_news.loc[fake_news['type'] == 'bs'].copy()
text_blobs = []
i = 0
for path in paths:
    with open(path, 'rb') as f:
        title = re.sub('[£—""…\n\"%]', ' ',f.readline()).strip()
        text = ' '.join([re.sub('[£—""…\n\"%]', ' ', x.strip().replace('"', ' ')) for x in f.readlines()])
        text_blob = TextBlob(text)
        text_blobs.append({'title':title,
                           'text':text,
                           'polarity':text_blob.sentiment[1],
                           'subjectivity':text_blob.sentiment[0],
                           'length':len(text),
                           'caps':count_char(title)})
onion_df = pd.read_csv('onion/onion_output.csv')
beaver_df = pd.read_csv('beaverton.csv')

import xml.etree.cElementTree as ET
from kitchen.text.converters import getwriter, to_bytes, to_unicode

wiki_file = open('enwiki-20140116.xml')
wiki_articles = []

def de_dupe(old_list):
    return [dict(t) for t in set([tuple(d.items()) for d in old_list])]

count = 0
for event, elem in ET.iterparse(wiki_file):
    if event == 'end':
        if elem.tag == 'title':
            title = elem.text
            title = to_bytes(title, 'utf-8')
            # print(type(title))
        if elem.tag == 'content':
            text = elem.text
            text = to_bytes(text, 'utf-8')
            # print(type(text))
        wiki_articles.append({'title':title,'text':text})
    elem.clear() # discard the element
    
wiki_articles = de_dupe(wiki_articles)
bbc_news = pd.DataFrame(text_blobs)


wiki_news = pd.DataFrame(wiki_articles)


bias_df['subjectivity'] = bias_df['text'].apply(lambda x: TextBlob(x).sentiment[1])
bias_df['polarity'] = bias_df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
bias_df['length'] = bias_df['text'].apply(lambda x: len(x))
bias_df['caps'] = bias_df['title'].apply(count_char)

bs_df['subjectivity'] = bs_df['text'].apply(lambda x: TextBlob(x).sentiment[1])
bs_df['polarity'] = bs_df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
bs_df['length'] = bs_df['text'].apply(lambda x: len(x))
bs_df['caps'] = bs_df['title'].apply(count_char)

onion_df['subjectivity'] = onion_df['text'].astype(str).apply(lambda x: TextBlob(x.decode('utf-8')).sentiment[1])
onion_df['polarity'] = onion_df['text'].astype(str).apply(lambda x: TextBlob(x.decode('utf-8')).sentiment.polarity)
onion_df['length'] = onion_df['text'].apply(lambda x: len(str(x)))
onion_df['caps'] = onion_df['title'].apply(count_char)

beaver_df['subjectivity'] = beaver_df['text'].astype(str).apply(lambda x: TextBlob(x.decode('utf-8')).sentiment[1])
beaver_df['polarity'] = beaver_df['text'].astype(str).apply(lambda x: TextBlob(x.decode('utf-8')).sentiment.polarity)
beaver_df['length'] = beaver_df['text'].apply(lambda x: len(str(x)))
beaver_df['caps'] = beaver_df['title'].apply(count_char)

wiki_news['subjectivity'] = wiki_news['text'].astype(str).apply(lambda x: TextBlob(x.decode('utf-8')).sentiment[1])
wiki_news['polarity'] = wiki_news['text'].astype(str).apply(lambda x: TextBlob(x.decode('utf-8')).sentiment.polarity)
wiki_news['length'] = wiki_news['text'].apply(lambda x: len(str(x)))
wiki_news['caps'] = wiki_news['title'].apply(count_char)

# get sub features from fake
f_sub = list(bias_df['subjectivity'])
f_sub.extend(bs_df['subjectivity'].values)
f_sub.extend(onion_df['subjectivity'].values)
f_sub.extend(beaver_df['subjectivity'].values)
# print(len(f_sub))

#get polarity features from news
f_pol = list(bias_df['polarity'].values)
f_pol.extend(bs_df['polarity'].values)
f_pol.extend(onion_df['polarity'].values)
f_pol.extend(beaver_df['polarity'].values)

f_len = list(bias_df['length'].values)
f_len.extend(bs_df['length'].values)
f_len.extend(onion_df['length'].values)
f_len.extend(beaver_df['length'].values)

f_caps = list(bias_df['caps'].values)
f_caps.extend(bs_df['caps'].values)
f_caps.extend(onion_df['caps'].values)
f_caps.extend(beaver_df['caps'].values)

fake = np.array([f_sub,f_pol])
fake = fake.transpose()
# print	(fake.shape)

r_sub = list(bbc_news['subjectivity'].values)
r_sub.extend(wiki_news['subjectivity'].values)

r_pol = list(bbc_news['polarity'].values)
r_pol.extend(wiki_news['polarity'].values)

r_len = list(bbc_news['length'].values)
r_len.extend(wiki_news['length'].values)

r_caps = list(bbc_news['caps'].values)
r_caps.extend(wiki_news['caps'].values)

real = np.array([r_sub,r_pol])
real = real.transpose()
import scipy

x = np.vstack((fake,real))
y = np.hstack(((np.ones((fake.shape[0],), dtype=np.int)),(np.zeros((real.shape[0],), dtype=np.int))))

from sklearn import model_selection
test_size = 0.25
seed = 7
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=test_size, random_state=seed)

from sklearn import svm
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create SVM classification object 
model = svm.SVC(kernel='linear', C=1, gamma=2) 
# there is various option associated with it, like changing kernel, gamma and C value. Will discuss more # about it in next section.Train the model using the training sets and check score
model.fit(x_train, y_train)
model.score(x_train, y_train)
import cpickle as pickle 
print "pickling"
with open('final_model.pkl', 'wb') as file:
	pickle.dump(model, file)