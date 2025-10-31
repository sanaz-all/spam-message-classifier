#ُMasoud janfeshan #sanaz allayhari

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
from sklearn.model_selection import train_test_split

# Reading the data
df = pd.read_csv("D:/Machine L projects/Masoud/train.csv",encoding='latin-1')

df = df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
df = df.rename(columns={'v1':'label','v2':'Text'})
df['label_enc'] = df['label'].map({'ham':0,'spam':1})

X, y = np.asanyarray(df['Text']), np.asanyarray(df['label_enc'])
new_df = pd.DataFrame({'Text': X, 'label': y})
X_train, y_train = new_df['Text'], new_df['label']


# Find average number of tokens in all sentences
avg_words_len = round(sum([len(i.split()) for i in df['Text']])/len(df['Text'])) 
#tedad kalame haye barname ro bar tedad jomleha taghsim mikone 

# Finding Total no of unique words in corpus # همه کلمات 1 بار اومده 
s = set()
for sent in df['Text']:
    for word in sent.split():
	    s.add(word)
total_words_length=len(s)

# Building the model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,accuracy_score

MAXTOKENS=total_words_length
OUTPUTLEN=avg_words_len

text_vec = TextVectorization( # یک لایه از کتابخونه کراس 
	max_tokens=MAXTOKENS,
	standardize='lower_and_strip_punctuation',
	output_mode='int',
	output_sequence_length=OUTPUTLEN
)
text_vec.adapt(X_train)

# Embedding layer
embedding_layer = layers.Embedding(
	input_dim=MAXTOKENS,
	output_dim=128,
	embeddings_initializer='uniform',
	input_length=OUTPUTLEN
)
#model_1
input_layer = layers.Input(shape=(1,), dtype=tf.string)
vec_layer = text_vec(input_layer)
embedding_layer_model = embedding_layer(vec_layer)
x = layers.GlobalAveragePooling1D()(embedding_layer_model)
x = layers.Flatten()(x)
x = layers.Dense(32, activation='relu')(x)
output_layer = layers.Dense(1, activation='sigmoid')(x)
model_1 = keras.Model(input_layer, output_layer)

model_1.compile(optimizer='adam', loss=keras.losses.BinaryCrossentropy(
	label_smoothing=0.5), metrics=['accuracy'])

history_1 =model_1.fit(X_train, y_train, epochs=5)

from sklearn.metrics import precision_score, recall_score, f1_score

def compile_model(model):
	'''
	simply compile the model with adam optimzer
	'''
	model.compile(optimizer=keras.optimizers.Adam(),
				loss=keras.losses.BinaryCrossentropy(),
				metrics=['accuracy'])

def fit_model(model, epochs, X_train=X_train, y_train=y_train,):
	'''
	fit the model with given epochs, train 
	and test data
	'''
	history = model.fit(X_train,
						y_train,
						epochs=epochs,
						)
	return history


def user_example():
    z=(input("test kon :"))
    df_user = pd.DataFrame ({'Text': z, 'sasas' :y})
    _, Z_test  = train_test_split (df_user['Text'], test_size=1)
    print (model_1.predict(Z_test))
    p = model_1.predict(Z_test)
    if np.round(p)==1:
         print('spam')
    else:
         print ('Not Spam')

def dataset_result():
    df_2 = pd.read_csv(input("Adress file ra vared konid :"),encoding='latin-1')
    #ex: D:/Machine L projects/test.csv
    df_2 = df_2.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
    df_2 = df_2.rename(columns={'v2':'Text'})
    U = np.asanyarray(df_2['Text'])
    new_df_2 = pd.DataFrame({'Text': U})
    X_test = new_df_2['Text']
    predictions = model_1.predict(X_test)
    result_df = pd.DataFrame({
    'Text': df_2['Text'],
    'Prediction': np.round(predictions).flatten()
    })
    result_df['Prediction'] = result_df['Prediction'].replace({0: 'Not Spam ', 1: 
    'Spam'})
    result_df.to_csv('output.csv', index=False)
    


while True  :
    us = input("Do you want the results of your dataset or do you have another example? [ex/ds]")
    if us=="ex" :
        user_example()
        break
	
    elif us=="ds" :
        dataset_result()
        break
    else: 
        print("Please enter either ex for an example or ds for a dataset.")




