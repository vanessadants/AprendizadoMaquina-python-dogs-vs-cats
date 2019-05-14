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
from sklearn.model_selection import train_test_split  #dividir test and train
from sklearn.preprocessing import StandardScaler #escalonamento
from sklearn.neighbors import KNeighborsClassifier  #classificador knn
from sklearn.metrics import classification_report, confusion_matrix  #matriz de confusao
from sklearn.model_selection import cross_val_score #cross validation



START=9
STOP=15
STEP=3
K_FOLD=10

TRAIN_DIR = './train'
IMG_SIZE = 50
N_IMGS=len(os.listdir(TRAIN_DIR))
N_COMPONENTS=45

def run_knn():

	# Read dataset
	# Separating out the features
	target=['Classe']
	features = []
	for i in range(0,N_COMPONENTS):
	    features.append('PCA'+str(i))
	#names=features
	#names.append(target)

	dataset =np.genfromtxt('train_data.csv', dtype=float, delimiter=',', names=True)  
	
	# Separating features and target
	X=[]
	for f in features:
		X.append(dataset[f])
	X=np.asarray(X,dtype=float).T
	

	#X = X.view((float, len(X.dtype.names)))
	y = dataset[target].astype(int)
	 
	#train and test split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  #80% train e 20% test

	X_train=np.asarray(X_train);
	X_test=np.asarray(X_test);
	y_train=np.asarray(y_train);
	y_test=np.asarray(y_test);

	#escalonamento
	scaler = StandardScaler()  
	scaler.fit(X_train)

	X_train = scaler.transform(X_train)  
	X_test = scaler.transform(X_test) 

	#treinamento e predicoes
	neighbors = range(START, STOP, STEP)

	# empty list that will hold cv scores
	cv_scores = []

	# perform k-fold cross validation
	for k in neighbors:
	    knn = KNeighborsClassifier(n_neighbors=k)
	    scores = cross_val_score(knn, X_train, y_train, cv=K_FOLD, scoring='accuracy')
	    cv_scores.append(np.mean(scores)) #media da acuracia
	    
	    #classificacao
	    #knn.fit(X_train, y_train)  
		#y_pred = knn.predict(X_test) 

		#matriz de confusão
		#print(confusion_matrix(y_test, y_pred))  
		#print(classification_report(y_test, y_pred))  

	#plot the misclassification error versus K
	# changing to misclassification error
	MSE = [1 - x for x in cv_scores]

	# determining best k
	optimal_k = neighbors[MSE.index(min(MSE))]
	print ("The optimal number of neighbors is %d" % optimal_k)

	# plot misclassification error vs k
	plt.plot(neighbors, MSE)
	plt.xlabel('Number of Neighbors K')
	plt.ylabel('Misclassification Error')
	plt.show()

	
knn_model=run_knn()