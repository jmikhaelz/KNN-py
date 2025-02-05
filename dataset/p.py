import numpy as np
import pandas as pd
import heapq
import os

 
datos=pd.read_csv("golf1.txt",header=0) # Lectura de la base de datos 
print( "\n \n " )
print( "\t\t\t  DATOS\n" ) #### titulo
print(datos)    #####imprime los datos que estan guardados
print("--------------------------------------------------------------")
print( "\n \n \n" )

############################ ordenando datos aleatoriamente 
datos=datos.to_numpy()
np.random.shuffle(datos) #ordena aleatoriamente 
#datos2=np.delete(dts, 6, axis=1) #elimina la fila 6
print( "datos aleatorios" ) 
print(datos)
print("---------------------------------------------------")
#print(datos2)# muestra los datos sin la fila 6 y aleatoria,ente

print( "\n \n " )
#########################################################     ENTRENAMIENTO = 10 
print ("entrenamiento")
entrenamiento1=datos[0:10].copy() #copia los 10  elementos  de entrenamiento ordenados aleatoriamente en  otro arreglo
entrenamiento2=np.delete(entrenamiento1, 6, axis=1) #ordena aleatoriamente y borra la fila 6
#print(entrenamiento1)
print(entrenamiento2)#----sin el yes o no
print("-------------------")
#########################################################     PRUEBA = 04
print ("prueba")
prueba1=datos[10:14].copy() #copia los 4 elementos  de prueba ordenados aleatoriamente en  otro arreglo
clas=np.delete(prueba1,[0,1,2,3,4,5], axis=1)  
prueba2=np.delete(prueba1, 6, axis=1)#ordena aleatoriamente y borra la fila 6
#for i in range(10,14): # muestra los datos de prueba 
#   posicion = datos[i]
print(prueba2)#----sin el yes o no
print("---------------------------------------------------")
print( "\n \n \n" )
ac = 0
#opc=input("\ningrese los k menores --  ")
opc=7
for j, i in zip(prueba2, clas):
    
    print("\n \t para P =",j,"-",i,"\n")
    ########################################################
    k=[]
    print('\tDistancias ')
    for i in range (len(entrenamiento2)):  #formula de las distancias 
        r = j - entrenamiento2[i]
        dist = np.sqrt(np.sum(np.square(r)))
        print(entrenamiento2[i],'-->',dist) #muestra las distancias con sus vectores 
        k.append(dist)
    #---------------------------------------------------------------3k
    ###########################################################
    print('\n')
    print("\t",opc,"k menores \n")
    kmenores=heapq.nsmallest(opc,k)  #detecta los k menores 
    for i in kmenores:
        print(i)  #muestra los k menores 
    print('\n')

    ###########################################################
    pos1=[]
    print("\tresultado\n")
    for i in kmenores:  # se crea un nuevo arreglo para saber  en que posicion estan los k menores 
        pos =k.index(i)  
        pos1.append(pos) ###

    for i in pos1:
        print(entrenamiento1[i]) #los datos de entrenamiento que estan en la misma posicion que los k menores 

    

    sol=np.delete(entrenamiento1,[0,1,2,3,4,5], axis=1)   
    #print (sol)
    yes = 0
    no = 0
    sol1=[] #guarda si y no
    for i  in pos1:
        sol1.append(sol[i])
        if(sol[i] == 'yes'):
            yes +=1
        else:
            no +=1
    print("\n")
    print("->",yes)
    print("->",no)
    if (yes>no):
        print(j,"--> yes")
        if(i == ' yes'):
            ac += 1
            print("acierto")
    if(yes<no):
        print(j,"--> no")
        if(i == ' no'):
            ac += 1
            print("acierto")
    if(yes==no):
        print(j,"--> igual")

print(ac)
print("\n---------------------------------------------------")
os.system(" Pause \n")
