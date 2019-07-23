import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

import re #limpio el tweet dejando solo letras
import nltk
nltk.download('stopwords') #lista de palabras irelevante
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#importo el dataset
dataset = pd.read_csv('TWEETS_LISTOS.tsv', delimiter= '\t', quoting= 3)

#aqui se limpia el texto de palabras que no aportan

tweets_limpios = []
for i in range(0, 225):
    tweet = re.sub('[^a-zA-Z]', ' ',dataset['text'][i])
    tweet = tweet.lower()
    tweet = tweet.split()
    # stopwords.words('spanish') stopword en español
    ps = PorterStemmer()  #root word cambia la palabra por una mas "simple" (creo que cambia los verbos a infinitivo)
    tweet = [ps.stem(word) for word in tweet if not word in stopwords.words('english')]
    tweet = ' '.join(tweet)
    tweets_limpios.append(tweet)

#aqui se crea un modelo de bolsa de palabras


cv = CountVectorizer()  

X = cv.fit_transform(tweets_limpios).toarray() # se crear una matriz donde cada fila represrna un tweet u cada columna representa las plabras del conjunto de plabras de todo el dataset de tweets, 
y = dataset.iloc[:,1].values

#entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state =0)

#estandariza los datos para no darle mas relevacia a ninguna colomna (no es necesario en nuestro caso de tweets)
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

#entrenamos un modelo de naive bayes 
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


#matriz de confusion 
cm = confusion_matrix(y_test, y_pred)
print (cm)