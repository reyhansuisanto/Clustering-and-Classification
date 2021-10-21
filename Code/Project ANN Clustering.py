# -*- coding: utf-8 -*-
"""Project ANN Clustering.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xQhl7hxnA4jfrRmscc8oICiwoCgNOsbr

# Project ANN Clustering
Reyhan Suisanto - 2301872980 <br>
Ryan Razaan Gunawan - 2301878290
"""

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#Pertama, load dataset dari clustering.csv

data = pd.read_csv('clustering.csv')
data

#Kemudian pisahkan data menjadi 2 bagian yaitu feature dan label

feature = data[['bruises', 'odor', 'stalk-shape', 'veil-type', 'spore-print-color']]
label = data['class']

#Lalu untuk features derivide saya menggunakan function replace

feature['odor'].replace({
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

feature['stalk-shape'].replace({
    'e': 1,
    't': 2
}, inplace=True)

feature['veil-type'].replace({
    'p': 1,
    'u': 2
}, inplace=True)

#Normalize dan PCA

scaler = StandardScaler()
feature = scaler.fit_transform(feature)

pca = PCA(n_components = 3)
principalComponents = pca.fit_transform(feature)

#SOM

class SOM:
  
  def __init__(self, width, height, n_features, learning_rate):
    self.width = width
    self.height = height
    self.n_features = n_features
    self.learning_rate = learning_rate

    self.cluster = []
    for i in range(height):
      self.cluster.append([])

    self.weight = tf.Variable(
        tf.random.normal(
            [width * height, n_features]
        ),
        tf.float32
    )
    self.input = tf.placeholder(tf.float32, [n_features])
    self.location = []
    for y in range(height):
      for x in range(width):
        self.location.append(tf.cast([y, x], tf.float32))
    
    self.bmu = self.get_bmu()
    
    self.update = self.update_neighbor()

  def get_bmu(self):
    distance = tf.sqrt(
        tf.reduce_mean((self.weight - self.input) ** 2, axis=1)
    )

    bmu_index = tf.argmin(distance)

    bmu_location = tf.cast([
        tf.div(bmu_index, self.width),
        tf.mod(bmu_index, self.width)
    ], tf.float32)

    return bmu_location

  def update_neighbor(self):
    distance = tf.sqrt(
        tf.reduce_mean((self.bmu - self.location) ** 2, axis=1)
    )

    sigma = tf.cast(tf.maximum(self.width, self.height) / 2, tf.float32)
    neighbor_strength = tf.exp(-(distance**2) / (2 * sigma ** 2))
    rate = neighbor_strength * self.learning_rate

    stacked_rate = []
    for i in range(self.width * self.height):
      stacked_rate.append(
          tf.tile(
            [rate[i]], 
            [self.n_features]
        )
      )
      
    
    delta = stacked_rate * (self.input - self.weight)
    new_weight = self.weight + delta

    return tf.assign(self.weight, new_weight)
  
  def train(self, dataset, num_epoch):
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())

      for epoch in range(num_epoch):
        print(f"Epoch {epoch}: ", end='')
        for data in dataset:
          dic = {
              self.input: data
          }
          sess.run(self.update, feed_dict = dic)
        print('Done')
      
      location = sess.run(self.location)
      weight = sess.run(self.weight)

      for i, loc in enumerate(location):
        self.cluster[int(loc[0])].append(weight[i])

#Visualization

som = SOM(8, 8, 3, 0.1)
som.train(principalComponents, 2500)
plt.imshow(som.cluster)
plt.show()