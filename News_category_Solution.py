
import pandas as pd

train_data = pd.read_excel("Data_Train.xlsx")

"""
Data Exploration
train_data.head(10)
train_data.info()
train_data.SECTION = train_data.SECTION.astype('str')
train_data.info()
train_data.describe()
train_data.drop_duplicates(inplace = True)
train_data.shape
train_data.groupby("SECTION").describe()
"""


#Data Preprocessing

#Training Data Corpus

import nltk
from nltk.corpus import stopwords
import string
#stopwords.words('english')

all_punctuations = string.punctuation + '‘’,:”][],' # added additional unidentifiable symbols

# func to remove punctuations
def punc_remover(raw_text):
    no_punct = "".join([i for i in raw_text if i not in all_punctuations])
    return no_punct
 
# func to remove stopwords
def stopword_remover(no_punc_text):
    words = no_punc_text.split()
    no_stp_words = " ".join([i for i in words if i not in stopwords.words('english')])
    return no_stp_words

# func to lemmatize
lemmer = nltk.stem.WordNetLemmatizer()
def lem(words):
    return " ".join([lemmer.lemmatize(word,'v') for word in words.split()])

# func to clean dataframe
def text_cleaner(raw):
    cleaned_text = stopword_remover(punc_remover(raw))
    return lem(cleaned_text)


#text_cleaner("Hi! I am Amal. If you dont know about me. Look me up on LinkedIn")
#train_data['STORY'].head(5).apply(text_cleaner).values


train_data['CLEAN_STORY'] = train_data['STORY'].apply(text_cleaner)

#train_data.values


#Vectorizing
from sklearn.feature_extraction.text import CountVectorizer

#bag of words model
bow_dictionary = CountVectorizer().fit(train_data['CLEAN_STORY'])
#len(bow_dictionary.vocabulary_)
bow_dictionary.vocabulary_

#Random Tests
#print(bow.transform([train_data['CLEAN_STORY'][0]]).shape)
#print(bow.transform([train_data['CLEAN_STORY'][0]]))
#bow_dictionary.get_feature_names()[25627]


#creating bag of words
bow = bow_dictionary.transform(train_data['CLEAN_STORY'])


#print(bow.shape)
#bow.nnz

#Checking sparsity
sparsity = bow.nnz/(bow.shape[0] * bow.shape[1])
print(sparsity)


#TFIDF
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer().fit(bow)

#tfidf_transformer.idf_[bow_dictionary.vocabulary_['university']]

storytfidf = tfidf_transformer.transform(bow)


# Classifier
from sklearn.naive_bayes import MultinomialNB
classfier = MultinomialNB().fit(storytfidf, train_data['SECTION'])

#train_p = classfier.predict(storytfidf)

# Training_set accuracy
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(train_p,train_data['SECTION'])
#print(cm)
acc = cm.diagonal().sum()/cm.sum()
print(acc)

#Predicting for Test Set

test_data = pd.read_excel("Data_Test.xlsx")

#test_data.head(10)

test_data['CLEAN_STORY'] = test_data['STORY'].apply(text_cleaner)

from sklearn.pipeline import Pipeline
pipe = Pipeline([
    ('bow', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])


pipe.fit(train_data['CLEAN_STORY'], train_data['SECTION'])

test_preds = pipe.predict(test_data['CLEAN_STORY'])

list(test_preds)

pd.DataFrame(test_preds, columns = ['SECTION']).to_excel('News_category_soln1.xlsx', index = False)

