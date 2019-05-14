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
IMG_SIZE = 50
N_IMGS=len(os.listdir(TRAIN_DIR))
N_COMPONENTS=45


def label_img(img): 
    word_label = img.split('.')[-3]
    #                            [gato]
    if word_label == 'cat': return 0
    #                             [cachorro]
    elif word_label == 'dog': return 1
    #return word_label
def create_train_data():
    i=0
    my_array=np.zeros((N_IMGS,IMG_SIZE*IMG_SIZE+1),dtype=float)

    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))

        
        lin,col = (len(img), len(img[0]))
        img=img.reshape(1,lin*col)
        
        for j in range(0,len(img[0])-1):
            my_array[i,j]=img[0][j]
        my_array[i,len(img[0])]=label

        i=i+1

    return np.asarray(my_array,dtype=float)


train_data=create_train_data()


#calculate range
lin,col = (len(train_data), len(train_data[0]))
#print('{}X{}'.format(lin, col))

# Assign colum names to the dataset
names = []
for i in range(1,col):
    names.append('Atributo '+str(i))
names.append('Classe')

dataset = pd.DataFrame(train_data,dtype=float,columns=names)
print(dataset.head(1))

#dataset.to_csv('train_data.csv', index=False)

# Separating out the features
features = []
for i in range(1,col):
    features.append('Atributo '+str(i))
x = dataset.loc[:, features].values


# Separating out the target
y = dataset.loc[:,['Classe']].values

#PCA
pca = PCA(n_components=N_COMPONENTS)

principalComponents = pca.fit_transform(x)

columns = []
for i in range(0,N_COMPONENTS):
    columns.append('PCA'+str(i))

principalDf = pd.DataFrame(data = principalComponents, columns = columns)

finalDf = pd.concat([principalDf, dataset[['Classe']]], axis = 1)

print(finalDf.head(1))

finalDf.to_csv('train_data.csv', date_format=float, index=False)


#Desenhando gráfico para análise de componentes principais
pca_training_data=pca.transform(x)
        
per_var_training=np.round(pca.explained_variance_ratio_* 100, decimals=1)
labels = columns
labels.append('Classe')

plt.bar(range(1,len(per_var_training)+1),height=per_var_training)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()