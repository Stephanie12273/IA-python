import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
# Importar archivo csv 
data=pd.read_csv('Archivos\SVM.csv')
#print(datos)
######### ANALISIS EXPLORATORIO #############
####### ANALISIS DE CADA VARIABLE DE MANERA INDIVIDUAL
print(data.head(5))
print (data.info()) #Tipo de variable para cada columna 
print (data.describe()) #Extrae variables estadisticas descriptivas 
#Histogramas que muestren el comportamiento del Acceso a la pagina y el Tiempo en la pagina 
col_num = ['Acceso a la página', 'Tiempo en la página']
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(16,8))  # Ajustar el tamaño de la figura
fig.subplots_adjust(hspace=0.5)
# Definir los pasos para cada columna
steps = {'Acceso a la página': 1, 'Tiempo en la página': 2} 
for i, col in enumerate(col_num):
    sns.histplot(x=col, data=data, ax=ax[i], kde=True)
    ax[i].set_title(col)
    ax[i].set_xticks(range(0, int(data[col].max()) + 1, steps[col]))  # Establecer los ticks del eje x con los pasos definidos
    ax[i].set_xticklabels(ax[i].get_xticks(), rotation=90)  # Rotar las etiquetas del eje x
plt.show()
#grafico de barras para conocer comportamiento de Agregacion al carrito y compra de producto












