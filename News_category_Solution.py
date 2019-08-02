#!/usr/bin/env python
# coding: utf-8

# # News Category Hackathon Solution

# In[17]:


import pandas as pd


# In[18]:


train_data = pd.read_excel("Data_Train.xlsx")


# In[19]:


train_data.head(10)


# In[20]:


train_data.info()


# In[21]:


train_data.SECTION = train_data.SECTION.astype('str')


# In[22]:


train_data.info()


# In[25]:


train_data.describe()


# In[27]:


train_data.drop_duplicates(inplace = True)


# In[34]:


train_data.shape


# In[36]:


train_data.groupby("SECTION").describe()


# ## Training Data Corpus

# In[32]:


import nltk


# In[51]:


from nltk.corpus import stopwords
import string


# In[49]:


stopwords.words('english')


# In[110]:


all_punctuations = string.punctuation + '‘’,:”][],'


# In[111]:


def punc_remover(raw_text):
    no_punct = "".join([i for i in raw_text if i not in all_punctuations])
    return no_punct
    


# In[145]:


def stopword_remover(no_punc_text):
    words = no_punc_text.split()
    no_stp_words = " ".join([i for i in words if i not in stopwords.words('english')])
    
    return no_stp_words


# In[163]:


lemmer = nltk.stem.WordNetLemmatizer()
def lem(words):
    return " ".join([lemmer.lemmatize(word,'v') for word in words.split()])


# In[ ]:





# In[164]:


def text_cleaner(raw):
    cleaned_text = stopword_remover(punc_remover(raw))
    return lem(cleaned_text)


# In[ ]:





# In[114]:


#text_cleaner("Hi! I am Amal. If you dont know about me. Look me up on LinkedIn")


# In[166]:


train_data['STORY'].head(5).apply(text_cleaner).values


# In[167]:


train_data['CLEAN_STORY'] = train_data['STORY'].apply(text_cleaner)


# In[169]:


train_data.values


# In[170]:


from sklearn.feature_extraction.text import CountVectorizer


# In[195]:


bow_dictionary = CountVectorizer().fit(train_data['CLEAN_STORY'])


# In[196]:


len(bow_dictionary.vocabulary_)


# In[197]:


bow_dictionary.vocabulary_


# In[184]:


train_data['CLEAN_STORY'][0]


# In[190]:


print(bow.transform([train_data['CLEAN_STORY'][0]]).shape)


# In[192]:


print(bow.transform([train_data['CLEAN_STORY'][0]]))


# In[199]:


bow_dictionary.get_feature_names()[25627]


# In[201]:


bow = bow_dictionary.transform(train_data['CLEAN_STORY'])


# In[205]:


print(bow.shape)


# In[207]:


bow.nnz


# In[211]:


sparsity = bow.nnz/(bow.shape[0] * bow.shape[1])
print(sparsity)


# ## TFIDF

# In[213]:


from sklearn.feature_extraction.text import TfidfTransformer


# In[215]:


tfidf_transformer = TfidfTransformer().fit(bow)


# In[ ]:





# In[221]:


tfidf_transformer.idf_[bow_dictionary.vocabulary_['university']]


# In[222]:


storytfidf = tfidf_transformer.transform(bow)


# In[224]:


from sklearn.naive_bayes import MultinomialNB
classfier = MultinomialNB().fit(storytfidf, train_data['SECTION'])


# In[226]:


train_p = classfier.predict(storytfidf)


# In[232]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(train_p,train_data['SECTION'])


# In[239]:


cm


# In[237]:


acc = cm.diagonal().sum()/cm.sum()


# In[240]:


acc


# ## Test Set

# In[241]:


test_data = pd.read_excel("Data_Test.xlsx")


# In[243]:


test_data.head(10)


# In[245]:


test_data['CLEAN_STORY'] = test_data['STORY'].apply(text_cleaner)


# In[248]:


#test_data.values


# In[249]:


from sklearn.pipeline import Pipeline


# In[253]:


pipe = Pipeline([
    ('bow', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])


# In[254]:


pipe.fit(train_data['CLEAN_STORY'], train_data['SECTION'])


# In[256]:


test_preds = pipe.predict(test_data['CLEAN_STORY'])


# In[259]:


list(test_preds)


# In[262]:


pd.DataFrame(test_preds, columns = ['SECTION']).to_excel('News_category_soln1.xlsx', index = False)

