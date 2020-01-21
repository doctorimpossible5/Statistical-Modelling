#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Configuration for JupyterLab
get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')

# Importing Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
import warnings
import os
warnings.filterwarnings("ignore")


# In[3]:


# Read in the Data from a CSV
users_csv = pd.read_csv('data/Users.csv', encoding='ISO-8859-1', delimiter=';')
books_csv = pd.read_csv('data/Books.csv', error_bad_lines=False, encoding='ISO-8859-1', delimiter=';')
ratings_csv = pd.read_csv('data/Ratings.csv', error_bad_lines=False, encoding='ISO-8859-1', delimiter=';')


# In[4]:


# Create the Dataset
dataset = pd.merge(ratings_csv, users_csv, on='User-ID', how='inner')
dataset = pd.merge(dataset, books_csv, on='ISBN', how='inner')
dataset.drop(columns=['Image-URL-M', 'Image-URL-L', 'Image-URL-S'], inplace=True)
dataset


# In[5]:


# Find Issues with Data
null_age = dataset['Age'].isnull().sum()
null_author = dataset['Book-Author'].isnull().sum()
null_publisher = dataset['Publisher'].isnull().sum()
null_rating = dataset['Book-Rating'].isnull().sum()
null_publish_date = dataset['Year-Of-Publication'].isnull().sum()


# In[6]:


# Handle Null Data
dataset['Book-Rating'] = dataset['Book-Rating'].replace(0, None)
dataset['Publisher'] = dataset['Publisher'].fillna('Unknown')
dataset['Book-Author'] = dataset['Book-Author'].fillna('Unknown')


# In[7]:


# Handle Outliers
upper_range = dataset['Age'].median() + dataset['Age'].std()
lower_range = dataset['Age'].median() - dataset['Age'].std()
possible_ages = np.random.randint(lower_range, upper_range, size=null_age)
age_copy = dataset['Age'].copy()
age_copy[pd.isnull(age_copy)] = possible_ages
dataset['Age'] = age_copy.astype(int)
dataset


# In[7]:


# Remove Extraneous Information in Location
location_copy = dataset['Location'].copy()
location_copy.shape
for item in range(location_copy.shape[0]):
    location_copy[item] = str(location_copy[item]).split(',')[-1]
    if(location_copy[item] == None or location_copy[item]=="") :
        location_copy[item] = "usa"
dataset['Location'] = location_copy


# In[8]:


# Create Training and Testing
X_train, X_test, y_train, y_test = train_test_split(dataset['User-ID'].unique(), dataset['User-ID'].unique(), test_size=0.25, shuffle=True)
train_dataset = dataset[dataset['User-ID'].isin(X_train)]
test_dataset = dataset[dataset['User-ID'].isin(X_test)]


# In[52]:


# Create searchable dictionary
all_books = test_dataset[['ISBN', 'Book-Title']].copy()
all_books.drop_duplicates(inplace=True, subset='ISBN', keep='last')
all_books_dict = all_books.groupby('ISBN')['Book-Title'].apply(list).to_dict()


# In[9]:


# Load Word2Vec Model if already Generated
model = Word2Vec.load("embedding_model_1.model")


# In[26]:


# Data for Training Embeddings
reading_training = list()
for i in tqdm(train_dataset['User-ID'].unique()):
    reading_training.append(train_dataset[train_dataset['User-ID'] == i]['ISBN'].tolist())


# In[27]:


# Data for Validating Embeddings
reading_validation = list()
for i in tqdm(test_dataset['User-ID'].unique()):
    reading_validation.append(test_dataset[test_dataset['User-ID'] == i]['ISBN'].tolist())


# In[30]:


# Train Embeddings
model = Word2Vec(window=9,sg=1,hs=0,negative=10,alpha=.0290,min_alpha=.0008)
model.build_vocab(reading_training, progress_per=300)
model.train(reading_training, total_examples=model.corpus_count, epochs=15, report_delay=1)


# In[38]:


# Save Model for Later Usage
model.save("embedding_model_1.model")


# In[10]:


# Precompute L2-normalized vectors.
model.init_sims(replace=True)


# In[75]:


# Extract the Vectors of Items in the Vocab 
vocab = model[model.wv.vocab]


# In[55]:


# Test Reccomendations
dataset[dataset['Book-Title'].str.contains('Dreamcatcher')].sample()


# In[77]:


similar_isbns = model.similar_by_vector('0743467523')
similar_books = []
for isbn in similar_isbns:
    pair = (all_books_dict[isbn[0]][0], isbn[1])
    similar_books.append(pair)
print("Books similar to Dreamcatcher by Stephen King")
similar_books


# In[ ]:





# In[ ]:





# In[ ]:




