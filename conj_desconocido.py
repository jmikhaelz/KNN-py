import pandas as pd                                     #Manejo, análisis y procesamiento de datos
from sklearn.model_selection import train_test_split    #Creacion de instancia de la clase, establecer los hiperparámetros 
import numpy as np                                      #Manejos matrices y arreglos multidimensionales
import AlgKNN as knn                                    #Clase donde almacenamos el proceso del algoritmo
    #TRANSFORMACION (TABLA DE LA BASE DE DATOS, OPCION DE TRANSFORMACION)
def transformacion(tabla,opcion):
            '''
                Outlook hacemos la transformacion de Nominal a Numerica
                    Correspondiente agregamos los campos
                        Sunny / Overcast / Rain 
                        Dado si tiene una similitud con el campo se agregara un 1
                        y de lo contrario se agregara un 0
                Transformacion de la campo Windy
                    Donde:
                        FALSE = 1
                        TRUE = 0
            '''
            conversion = []
            for outk, w in zip(tabla['Outlook'],tabla['Windy']):
                    match outk:
                        case 'sunny':
                            if(w == False):
                                    conversion.append([1,0,0,1])
                            else:
                                    conversion.append([1,0,0,0])
                        case 'overcast':
                            if(w == False):
                                    conversion.append([0,1,0,1])
                            else:
                                    conversion.append([0,1,0,0])
                        case 'rainy':
                            if(w == False):
                                    conversion.append([0,0,1,1])
                            else:
                                    conversion.append([0,0,1,0])
            tabla.pop('Outlook')         #Eliminamos la columna Outlook, Windy
            tabla.pop('Windy')  
            tabla = pd.concat([tabla, pd.DataFrame (conversion)], axis = 1)    #afrefamos la matriz de las transformacion
            if(opcion == 0): #Revisamos la opcion si tiene la clase PLAY
                #Renombramos los campos para tener mayusculas
                tabla.rename(columns={'Temp': 'TEMP', 'Humidity' : 'HUMIDITY','Play':'PLAY', 0 : 'SUNNY', 1:'OVERCAST', 2:'RAIN',3:'WINDY'}, inplace=True)
                #Ordenamos las columnas
                tabla = pd.DataFrame.reindex(tabla,columns = ['SUNNY', 'OVERCAST', 'RAIN','TEMP', 'HUMIDITY','WINDY','PLAY'])
                return tabla
            else:
                #Renombramos los campos para tener mayusculas
                tabla.rename(columns={'Temp': 'TEMP', 'Humidity' : 'HUMIDITY', 0 : 'SUNNY', 1:'OVERCAST', 2:'RAIN',3:'WINDY'}, inplace=True)
                #Ordenamos las columnas visualmente
                tabla = pd.DataFrame.reindex(tabla,columns = ['SUNNY', 'OVERCAST', 'RAIN','TEMP', 'HUMIDITY','WINDY'])
                return tabla    
try:
    print("\nCONJUNTOS DE DATOS ORIGINAL\n")
    #Conversion del archivo txt a una tabla DataFrame
    print("_"*14+"GOLF_NOMINAL"+"_"*14)
    tabla = pd.read_csv("dataset/golf.txt", header=0)#Especificamos que la primera linea del archivo txt son las titulos de las columnas
    tabla = transformacion(tabla,0)
    print(tabla)
    print("¯"*54)
    
    print("\n APARATADO DE PRUEBA CON CONJUNTO DESCONOCIDO\n\n")
    k = int(input(" Ingresa el valor de k: "))
    directorio = input("\n Ingresa la dirección de la carpeta con los archivos de texto: ")
    test = pd.read_csv(directorio, header=0)
    
    directorio = directorio.split(sep='.')  #Separamos el tipo del archivo por el punto
    
    print("\n"+"_"*8+"TABLA DE "+directorio[0].upper()+"_"*8)
    print(test)
    print("¯"*54)
    
    
    test = transformacion(test,1)   #Transformacion de la tabla por la razon del Outlook y sus diferentes opciones
    print("\n APLICACION DEL ALGORITMO DE LA CLASIFICACION CORRESPONDIENTE A LA CLASE PLAY\n")
        
    play = tabla.pop('PLAY')    #Separamos la clase PLAY para el algoritmo KNN
    entre = tabla.to_numpy()    #Transformacion a Array
    play = play.to_numpy()
    test1 = test.to_numpy()
    
    #Realizamos el algoritmo de KNN
    operacion = knn.AlgKNN(k)  #Creamos el objeto con el valor de K
    operacion.learn(entre,play) #Agregamos la muestra de entrenamiento y la clase donde vamos a clasificar
    
    test['PLAY'] = pd.Series(operacion.clasf(test1)) #Referenciamos el array de test y el resultado de la clase PLAY la concatenamos a test
    
    print("\n"+"_"*14+"TABLA DE "+directorio[0].upper()+"_"*14)
    print(test)
    print("¯"*54)
except Exception as e: #Si hay un error en un archivo de identificara
    print(f"\n <!> : {str(e)}")