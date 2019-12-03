
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt
import io
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# LOADING DEPENDANT AND INDEPENDANT VARIABLES
X_all=pd.read_json('/home/ajay/Documents/independant_variables.json', typ='series');
Y_all=pd.read_json('/home/ajay/Documents/dependant_variables.json', typ='series');

# print(df.groupby('is_sarcastic').size());
# print(df[df.is_sarcastic==0]);

# SPLITTING DATA
X_train, X_test, Y_train, Y_test=train_test_split(X_all, Y_all,test_size=0.25, random_state=42069);

# FEATURE EXTRACTION
vectorizer= TfidfVectorizer(stop_words='english');
tfidf_train= vectorizer.fit_transform(X_train);
# print(tfidf_train.shape);
tfidf_test= vectorizer.transform(X_test);
# print(tfidf_test.shape);

# FITTING AND PREDICTION
clf=MultinomialNB();
clf.fit(tfidf_train, Y_train);
pred= clf.predict(tfidf_test);
score=accuracy_score(Y_test, pred );
print("accuracy:   %0.3f" % score);