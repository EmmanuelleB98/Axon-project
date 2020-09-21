import os
import sys
import time
import numpy as np
import glob
import pickle
import cv2
import networkx as nx
from PIL import Image
import tensorflow.compat.v1 as tf
import tensorflow as tf

from tensorflow.compat.v1  import ConfigProto 
from tensorflow.compat.v1  import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True

session = InteractiveSession(config=config)
import keras
from skimage.io import imread
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import load_model
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Dropout
from keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from scipy.sparse import csr_matrix, lil_matrix
from visualize import visualize_nodes
from process_data import process
from gcn.train import run_training
from numpy import loadtxt



# Load training data 
TRAIN="./Synthetic Image/train/"
VAL="./Synthetic Image/val/"
LABEL_TRAIN="./Labels/train/"
LABEL_VAL="./Labels/val/"

import pickle
with open("splines.txt", "rb") as fp:   # Unpickling
    splines_points= pickle.load(fp)

with open("nodes.txt", "rb") as fp:  
    nodes_values= pickle.load(fp)

#Compute offset between the initialised spline points and labels nodes values

def compute_offset(splines_points, nodes_values):
	offsets = []
	for i in range(len(nodes_values)): #node_values.shape[0]
		min_dist, min_offset = None, None
		for j in range(len(splines_points)):
			node_value, splines_point = nodes_values[i], splines_points[j]
			if min_dist is None or np.linalg.norm(np.array(splines_point) - np.array(node_value)) < min_dist:
				min_dist = np.linalg.norm(np.array(splines_point) - np.array(node_value))
				min_offset = [splines_point[0] - node_value[0], splines_point[1] - node_value[1]]
		offsets.append(min_offset)
	return np.array(offsets)

# GCNs
epochs_per_image = 300
gcn_models = [None]*epochs_per_image 


for image,j in zip(glob.glob(os.path.join(TRAIN, '*.jpg')),range(len(splines_points))):



  
  
  img=imread(image)
  img = img.astype('float32')
  image=img/255

  
  
  #Nodes values initialisation 
  visualize_nodes(nodes_values, image)
  visualize_nodes(splines_points, image)
 

  resized_image = img_to_array(image)
  resized_image = np.resize(resized_image,(256, 256,1)) 
  resized_image_exp = np.expand_dims(resized_image, axis=0)

  # Compute feature map
  model=load_model('model.h5')
  embedding_model = Model(inputs=model.inputs, outputs=model.layers[30].output)
  feature_map = embedding_model.predict(resized_image_exp)

  # Define graph
  N = 30 
  G = nx.Graph()

 
  for i in range(N):
    if i-2 < 0:
      G.add_edge(N+(i-2), i)
    else:
      G.add_edge((i-2), i)
    if i-1 < 0:
      G.add_edge(N+(i-1), i)
    else:
      G.add_edge((i-1), i)
    G.add_edge((i+1)%N, i)
    G.add_edge((i+2)%N, i)
  
  #nx.draw(G)


  # Run training
  for epoch in range(epochs_per_image):
    print("Epoch {0} of {1}".format(epoch+1, epochs_per_image))
    # Bilinear interpolation for creating the input features 
    input_features = []

    for i in range(len(nodes_values)):
      fx = int(np.floor((nodes_values[i][0]/image.shape[1])*feature_map.shape[1]))
      fy = int(np.floor((nodes_values[i][1]/image.shape[0])*feature_map.shape[2])) 
      if fx >= feature_map.shape[1]: 
        fx = feature_map.shape[1] - 1
      if fy >= feature_map.shape[2]:
        fy = feature_map.shape[2] - 1
      input_feature = feature_map[0][fx][fy]
      input_feature = np.concatenate([input_feature, np.array(nodes_values[i])])
      input_features.append(input_feature)
    input_features = np.array(input_features) #shape (15x130)
    print('features shape',input_features.shape)

    # Define graph propagation
    
    offsets = compute_offset(splines_points, nodes_values)
    print(offsets)
    epoch_model = gcn_models[epoch]
    new_epoch_model, output_offsets = run_training(csr_matrix(nx.adjacency_matrix(G)),
      lil_matrix(input_features), offsets, offsets, offsets, np.ones((N), dtype=bool),
      np.ones((N), dtype=bool), np.ones((N), dtype=bool), 'gcn', epoch_model)
    gcn_models[epoch] = new_epoch_model
    nodes_values += output_offsets
    print('NODES PREDICTIONS', nodes_values)
    visualize_nodes(nodes_values, image)



  visualize_nodes(nodes_values, image)
  visualize_nodes(splines_points, image)
