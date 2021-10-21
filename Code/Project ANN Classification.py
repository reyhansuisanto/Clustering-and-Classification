#!/usr/bin/env python
# coding: utf-8

# # Project ANN Classification
# Reyhan Suisanto - 2301872980 <br>
# Ryan Razaan Gunawan - 2301878290

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA


# In[2]:


#Load Dataset
dataset = pd.read_csv('classification.csv')
dataset


# In[3]:


dataset['cap-shape'].replace({
    'b': 1,
    'c': 2,
    'x': 3,
    'f': 4,
    'k': 5,
    's': 6
}, inplace=True)

dataset['odor'].replace({
    'a': 1,
    'l': 2,
    'c': 3,
    'y': 4,
    'f': 5,
    'm': 6,
    'n': 7,
    'p': 8,
    's': 9
}, inplace=True)

dataset['habitat'].replace({
    'g': 1,
    'l': 2,
    'm': 3,
    'p': 4,
    'u': 5,
    'w': 6,
    'd': 7
}, inplace=True)

dataset


# In[4]:


features = dataset[['cap-shape', 'cap-color', 'odor', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-color', 'ring-number', 'habitat']]
features


# In[5]:


label = dataset['class']
label = np.array(label)
label = label.reshape(-1,1)
label


# In[6]:


#Preprocessing Dataset

scaler = MinMaxScaler()
features = scaler.fit_transform(features)

encoder = OneHotEncoder()
label = encoder.fit_transform(label).toarray()


# In[7]:


pca = PCA(n_components = 4)
pca.fit(features)
pca_data = pca.transform(features)


# In[8]:


#Variabel pembantu untuk model

layer = {
    'input' : 4,
    'hidden' : 64,
    'output' : 2
}

weight = {
    'th' : tf.Variable(tf.random_normal([layer['input'], layer['hidden']])),
    'to' : tf.Variable(tf.random_normal([layer['hidden'], layer['output']]))
}

bias = {
    'th' : tf.Variable(tf.random_normal([layer['hidden']])),
    'to' : tf.Variable(tf.random_normal([layer['output']]))
}


# In[9]:


#Split Dataset

x_train, x_rest, y_train, y_rest = train_test_split(pca_data, label, test_size = 0.3)
x_valid, x_test, y_valid, y_test = train_test_split(x_rest, y_rest, test_size = 0.33)

x = tf.placeholder(tf.float32, [None, layer['input']])
label = tf.placeholder(tf.float32, [None, layer['output']])


# In[10]:


#Function Forward Pass // Prediction

def forward_pass():
    wx_b1 = tf.matmul(x, weight['th']) + bias['th']
    y1 = tf.nn.sigmoid(wx_b1)

    wx_b2 = tf.matmul(y1, weight['to']) + bias['to']
    y2 = tf.nn.sigmoid(wx_b2)

    return y2


# In[11]:


#Isi value dari prediction kita

output = forward_pass()

#Variable yang membantu kita dalam training dan testing

epoch = 2500
alpha = 0.75

#MSE Function

error = tf.reduce_mean(0.5 * (label-output) ** 2)
optimizer = tf.train.GradientDescentOptimizer(alpha)
train = optimizer.minimize(error)


# In[12]:


saver = tf.train.Saver()


# In[13]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    best_error = float('inf')
    for i in range(epoch):
        sess.run(
            train,
            feed_dict={
                x : x_train,
                label : y_train
            }
        )

        if i % 25 == 0 :
            current_error = sess.run(
                error,
                feed_dict = {
                    x: x_train,
                    label : y_train
                }
            )
            print(f'epoch : {i+1} | Error : {current_error:.4f}')
        if i % 125 == 0 :
            valid_error = sess.run(
                error,
                feed_dict = {
                    x: x_valid,
                    label : y_valid
                }
            )
            print(f'epoch : {i+1} | Valid Error : {valid_error:.4f}')
            if valid_error < best_error :
                best_error = valid_error
                saver.save(sess, './final_model.ckpt')
                
            
    true_prediction = tf.equal(tf.argmax(output,axis=1), tf.argmax(label, axis=1))
    accuracy = tf.reduce_mean(tf.cast(true_prediction, tf.float32))
    saver.restore(sess, './final_model.ckpt')
    accuracy = sess.run(
        accuracy,
        feed_dict = {
        x : x_test,
        label : y_test
        }
    )
    print(f'Accuracy : {accuracy:.4f}')


# In[ ]:




