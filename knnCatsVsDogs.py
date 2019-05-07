# -*- Coding: UTF-8 -*-
#coding: utf-8
#Authors:Vanessa Dantas, Valmir Ferreira
import numpy as np         # lidar com arrays
import cv2                 # trabalhando com imagens redimensionadas
import os                  # lidar com diretorios
from tqdm import tqdm      # barra de percentual para tarefas
from random import shuffle # inserir mistura nos dados para treinamento futuro
from sklearn.decomposition import PCA #importando a funcao PCA de skylearn
from sklearn import preprocessing #o pacote de preprocessamento do skylearn nos dá funções para escalar os dados antes de utilizar PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt #desenhar graficos
import pandas as pd #data science

TRAIN_DIR = './train'
TEST_DIR = './test'
IMG_SIZE = 50
LR = 1e-3

MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '2conv-basic') # os tamanhos devem ser equivalentes

def run_knn():
	# Se o dataset já tiver sido criado utilizar o comando
	train_data = np.load('train_data.npy')
	#Creating pandas dataframe from numpy array
	dataset = pd.DataFrame({'Column1':train_data[:,0],
							'Column2':train_data[:,1] 
							#'Column3':train_data[:,2],
							#'Column4':train_data[:,3],
							#'Column5':train_data[:,4],
							#'Column6':train_data[:,5],
							#'Column7':train_data[:,6],
							#'Column8':train_data[:,7]
							})
	print(dataset)	


knn_data = run_knn()