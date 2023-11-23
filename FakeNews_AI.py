#!/usr/bin/env python
# coding: utf-8

# In[7]:


import sys
print(sys.executable)


# In[2]:


from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

import numpy as np #linear Algebra
import pandas as pd #Data Processing

import os
import re
import nltk


# In[3]:


train=pd.read_csv(r"C:\Users\Supriya Batabyal\Downloads\train.csv")
test=pd.read_csv(r"C:\Users\Supriya Batabyal\Downloads\test.csv")


# In[4]:


print(train.shape, test.shape)


# In[5]:


train.head()


# In[6]:


#Count null data
print(train.isnull().sum())
print("***********")
print(test.isnull().sum())


# In[7]:


test=test.fillna(' ') #Replacing all the null values with space
train=train.fillna(' ')
#Now we need to merge all the data like title,author,text etc as we are working with textual data
test['total']=test['title']+' '+test['author']+test['text']
train['total']=train['title']+' '+train['author']+test['text']


# # CREATING WORDCLOUD VISUALS(Wordcloud is basically a visualization technique to represent the frequency of words in a text where the size of the word represents its frequency.)

# In[8]:


real_words = ' '
fake_words = ' '
stopwords = set(STOPWORDS)

#iterate through csv file
for val in train[train['label']==1].total:
    
    #split the value
    tokens = str(val).split()
    
    #Convert each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
        
    real_words +=" ".join(tokens)+" "
    
for val in train[train['label']==0].total:
    
    #split the values
    tokens = str(val).split()
    
    #Convert each token into lowercase
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
        
    fake_words +=" ".join(tokens)+" "


# In[54]:


wordcloud = WordCloud(width = 1000, height = 1000,
                     background_color ='white',
                     stopwords = stopwords,
                     min_font_size = 10).generate(fake_words)

#plot the wordcloud image
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)

plt.show()


#  #  Cleaning And Preprocessing
#  
#  # 1.Regex

# In[55]:


#Remove Puncution From the String
s = "!</> $$ </>^!!!%%&&%$@@@be^^^&&!& </>*@# &&\\ @@@n##%^^&!@# %%$"


# In[56]:


import re
s = re.sub(r'[^\\w\\s]','',s)


# # 2.Tokenization
# 

# In[57]:


#Downloading nltk data
nltk.download('punkt')
#(In NLTK, PUNKT is an unsupervised trainable model, which means it can be trained on unlabeled data (Data that has not been 
#tagged with information identifying its characteristics, properties, or categories is referred to as unlabeled data.)


# In[58]:


nltk.word_tokenize("hello how are you")


# # 3.StopWords

# In[25]:


#Stop Words: A stop word is a commonly used word (such as “the”, “a”, “an”, “in”) that a search engine has been programmed to
    #ignore, both when indexing entries for searching and when retrieving them as the result of a search query.
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
print(stop_words)


# In[14]:


stop_words.append('helo')
stop_words


# In[11]:


sentence = "I am going to play the match day after tomorrow"


# In[12]:


words= nltk.word_tokenize(sentence)
words=[w for w in words if w not in stop_words]


# In[13]:


words


# # 4.Lemmatization
# # Using Lemmatization we can reduce a word into the smallest dictionary form
# # There is another form is present,called Stemming but is follows only the given rule irrespective of it's meaning

# In[30]:


from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()

input_str="Are you running everyday morning in the fields"


# In[21]:


import nltk
nltk.download('wordnet')


# In[31]:


#Tokenize the sentence
input_str=nltk.word_tokenize(input_str)

#Lematize each word
for word in input_str:
    print(lemmatizer.lemmatize(word))


# # Let's Apply 

# In[61]:


lemmatizer=WordNetLemmatizer()
for index,row in train.iterrows():
    filter_sentence = ''
        
    sentence = row['total']
    #sentence = re.sub(r'[^\\w\\s]','',str(sentence)) #cleaning
        
    words = nltk.word_tokenize(str(sentence)) #tokenization
        
    words = [w for w in words if not w in stop_words]  #stopwords removal
        
    for word in words:
        filter_sentence = filter_sentence + ' ' + str(lemmatizer.lemmatize(word)).lower()
            
    train.loc[index,'total'] = filter_sentence


# In[62]:


train.head()


# In[63]:


train = train[['total','label']]


# In[ ]:




