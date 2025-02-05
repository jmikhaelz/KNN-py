import pandas as pd                 #Manejo, análisis y procesamiento de datos
from collections import Counter     #Te permite calcular la frecuencia
import numpy as np                  #Manejos matrices y arreglos multidimensionales
# Basado a la codificacion del video Clasificador KNN | Machine Learning | Aprendizaje Automático | Python
# Autor: Victor Romero
# Link : https://www.youtube.com/watch?v=qv0rb_A0f3M&t=1098s
class AlgKNN():
    def __init__(self,k):   #Inicializamos con la K = Numero de vecinos cercanos
        self.k = k
    def learn(self,Q,C):    #Metodo de Aprendizaje Q = Test, C = Donde es la clase de entrenamiento donde clasifica la tupla
        self.Q = Q
        self.C = C
        self.nm = Q.shape[0]        #Numero de columnas donde revisaremos las instancias para la distancia
    def clasf(self,P):              #Metodo de Clasificacion a partir de PLAY
        clase = []                   #Array para almacenar que resultado SI o NO
        for i in range(P.shape[0]): #Ciclo FOR para recorrer todas la array de pruebas (P0,..,Pn)
            d = np.empty(self.nm) #Array de distancias d(P,Qn)
            for n in range(self.nm):#Recorremos las tuplas de (Q0,..,Qn)
                d[n] = euclidiana(self.Q[n,:],P[i,:])   #Formula de distancia euclidiana
            k_d = np.argsort(d) #Ordenamos el array de menor a mayor, y mostramos solo el indice del array
            k_etq = self.C[k_d[:self.k]] #Sacamos los valores del array de PLAY/clasificacion donde se limitarian un rando de 0 a k
            cnt_n = 0
            cnt_y = 0
            for f in k_etq: #CONTAMOS LA VECES QUE APARECEN LAS CLASES
                if(f != "YES"):
                    cnt_n += 1
                else:
                    cnt_y += 1
            if(cnt_n > cnt_y):  #ASIGNAMOS EL VALOR DE LA CLASE QUE SE REPITE MAS
                clase.append("NO")
            if(cnt_y > cnt_n):
                clase.append("YES")
        return clase #Returnamos el array de play
def euclidiana(x,y):    #Funcion donde calculamos la distancia euclidiana
    return np.sqrt(np.sum((x-y)**2)) #Hacemos la sumatoria de las (x-y)^2 y despues le sacamos raiz