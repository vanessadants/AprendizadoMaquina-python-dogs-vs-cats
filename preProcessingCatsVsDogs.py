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

TRAIN_DIR = './train'
IMG_SIZE = 50

def label_img(img): 
    word_label = img.split('.')[-3]
    # conversao para um array [gato,cachorro]
    #                            [gato]
    if word_label == 'cat': return [1,0]
    #                             [cachorro]
    elif word_label == 'dog': return [0,1]
def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        #Preparacao para PCA
        scaled_training_data=StandardScaler().fit_transform(np.array(img).T) #centralizar e escalar os dados
        #PCA
        pca = PCA()
        pca.fit(scaled_training_data)
        pca_training_data=pca.transform(scaled_training_data)
        
        #per_var_training=np.round(pca.explained_variance_ratio_* 100, decimals=1)
        #labels = ['PC' +str(x) for x in range(1,len(per_var_training)+1)]
        #Desenhando gráfico para análise de componentes principais
        #plt.bar(range(1,len(per_var_training)+1),height=per_var_training)
        #plt.ylabel('Percentage of Explained Variance')
        #plt.xlabel('Principal Component')
        #plt.title('Scree Plot')
        #plt.show()


        training_data.append([np.array(pca_training_data),np.array(label)])
        
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


train_data = create_train_data()
# Se o dataset já tiver sido criado utilizar o comando
#train_data = np.load('train_data.npy')

