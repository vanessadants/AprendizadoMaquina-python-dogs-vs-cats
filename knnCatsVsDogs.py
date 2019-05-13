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


start=3
stop=15
step=3

error = []

def run_knn(k):
	# Se o dataset já tiver sido criado utilizar o comando
	train_data = np.load('train_data.npy')
	#Creating pandas dataframe from numpy array
	dataset = pd.DataFrame({'Dados':train_data[:,0], 'Groundtruth':train_data[:,1] })
	print(dataset.head())	

	#Split dados e labels
	X = dataset.iloc[:, :-1].values  
	y = dataset.iloc[:, 1].values  

	#train and test split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  #80% train e 20% test

	#escalonamento
	scaler = StandardScaler()  
	scaler.fit(X_train)

	X_train = scaler.transform(X_train)  
	X_test = scaler.transform(X_test) 

	#treinamento e predicoes
	classifier = KNeighborsClassifier(n_neighbors=k)  
	classifier.fit(X_train, y_train)  

	y_pred = classifier.predict(X_test) 
	error.append(np.mean(y_pred != y_test))

	#matriz de confusão
	#print(confusion_matrix(y_test, y_pred))  
	#print(classification_report(y_test, y_pred))  

# Calculating error for K values between 1 and 40
for i in range(start, stop, step):  
   run_knn(i)

#plot error
plt.figure(figsize=(12, 6))  
plt.plot(range(start, stop, step), error, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Error')  