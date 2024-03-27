import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Importar archivo csv 
data=pd.read_csv('Archivos\SVM.csv')
#print(datos)

######### ANALISIS EXPLORATORIO #############
####### ANALISIS DE CADA VARIABLE DE MANERA INDIVIDUAL 

print (data.info()) #Tipo de variable para cada columna 
print (data.describe()) #Extrae variables estadisticas descriptivas 

# Hay 1000










