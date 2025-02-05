import random
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
#PRUEBAS(VECES QUE SE HARA EL ALGORITMO, NO. DE VECINOS, TABLA DE ENTRENAMIENTO, TABLA DE CLASE A CLASF, TABLA DE PRUEBA, INDICES DE LA TABLA DE PRUEBA)    
def prueba_k(it,k,tabla,visual): #Funcion para hacer las pruebas de K , y la base de datos
        print("\n PRUEBA SOBRE "+str(k)+" VECINOS CERCANOS CORRESPODIENTE K = "+str(k)+"\n")
        p = 0   #Variable contador para los porcentajes
        tabla_porc = []
        for x in range(it): #Rango de iteraciones
            tabla = tabla.sample(frac=1).reset_index(drop=True)
            #El método train_test_split() se utiliza para dividir nuestros datos en conjuntos de entrenamiento y de prueba.
            #El muestreo de las X's son las tablas son la clase PLAY , y e donde las Y's almacenan los valores de PLAY
            X_train, X_test, y_train, y_test= train_test_split(
                tabla.drop('PLAY', axis=1), tabla['PLAY'], stratify=tabla['PLAY'] , #Separamos la columna PLAY donde se hara los resultaso de KNN
                random_state=(len(tabla)+random.randint(0, x+1)+k), #Cuantas veces se van a revolver
                test_size=0.25)   #Tam de la muestra de test
            
            #Almacenamos el indice de la prueba , por cuestiones de transformacion a array, por no respecta el orden que tenia en el dataframe
            indices = []
            for id in X_test.index:
                indices.append(id)
            
            #Convertimos los dataframe a array y transformamos la clase PLAY
            entrenamiento = X_train.to_numpy()
            play_e = y_train.to_numpy()
            prueba = X_test.to_numpy()
            play_p = y_test.to_numpy()
            if(visual == 1):
                print("\n\n ITERACCION "+str(x+1))
                
                #Unimos las partes de cada subconjunto con su correspondiente clase play para imprimirla en consola
                X_train['PLAY'] = y_train
                X_test['PLAY'] = y_test
                print("\n"+"_"*14+"TABLA DE ENTRENAMIENTO"+"_"*14)
                print(X_train)
                print("¯"*54)
                print("\n"+"_"*14+"TABLA DE PRUEBA"+"_"*14)
                print(X_test)
                print("¯"*54)
            #Creamos el objeto de la clase AlgKNN, para realizar el algoritmo
            test = knn.AlgKNN(3)  #Creamos el objeto con el valor de K
            test.learn(entrenamiento,play_e) #Agregamos la muestra de entrenamiento y la clase donde vamos a clasificar
            resultado = test.clasf(prueba)  #Referenciamos el array de prueba y nos devolvera un array
            comparacion = []    #Array donde se almacenan los aciertos en las comparaciones
            count = 0   #Variable de los aciertos
            for i,j in zip(play_p,resultado):   #For para recorrer los array de la clase PLAY
                if(i == j):                     #Si son iguales agregara una paloma, si es lo contrario agrega una tache
                    comparacion.append("✔")
                    count += 1
                else:
                    comparacion.append("✘")
            p += (count/float(play_p.size))*100
            tabla_porc.append((count/float(play_p.size))*100) #ALMACENAMOS EL PORCENTAJES DE CADA ITERACCION
            if (visual == 1):
                #Creamos tabla con las comparaciones de cada clase
                tabla1 = pd.DataFrame({'PLAY DE PRUEBA':play_p, 'PLAY CON K-NN': resultado, 'COMPARACION': comparacion})
                print("\nCOMPARACION\n")
                tabla1.insert(0, 'PRUEBA', indices) #Agramos indices para saber a cuales corresponden cada prueba
                print(tabla1.to_string(index=False)) #Mostarmos la tabla sin los indices del dataframe
                print("\n Promedio de aciertos : "+str((count/float(play_p.size))*100)+"% en la iteracion "+str(it)+" con el algoritmo K-NN\n\n")
        tabla_porc = pd.DataFrame({'PROMEDIO DE ACIERTO':tabla_porc}) 
        print(tabla_porc) #MOSTRAMOS LA TABLA DE PROCENTAJES
        print("\n\n Promedio de aciertos : "+str(p/float(it))+"% despues de "+str(it)+" iteraciones con el algoritmo K-NN\n\n") #Imprision de los porcentajes     
try:
    print("\nCONJUNTOS DE DATOS ORIGINAL\n")
    #Conversion del archivo txt a una tabla DataFrame
    print("_"*14+"GOLF_NOMINAL"+"_"*14)
    tabla = pd.read_csv("dataset/golf.txt", header=0)#Especificamos que la primera linea del archivo txt son las titulos de las columnas
    tabla = transformacion(tabla,0)
    print(tabla)
    print("¯"*54)
    
    #Hacemos utilidad de la funcion prueba y con su correspondiente caso
    print("\n ENTRENAMIENTO DEL ALGORITMO K-NN\n")
    prueba_k(20,7,tabla,0)
    prueba_k(20,5,tabla,0)
    prueba_k(20,3,tabla,0)
except Exception as e: #Si hay un error en un archivo de identificara
    print(f"\n <!> : {str(e)}")