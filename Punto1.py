import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
# Importar archivo csv 
data=pd.read_csv('Archivos\SVM.csv')
#print(datos)

#Funcion para graficar tasas de conversion 
def graficar_tasas_conversion(var_predictora, var_predecir, type='line', order=None):
    x, y = var_predictora, var_predecir
    # agrupación (groupby), calcular tasa de conversión (mean), multiplicarla por 100
    grupo = data.groupby(x)[y].mean() * 100
    grupo = grupo.rename('tasa_conv').reset_index()

    

    if type == 'line':  #rangos continuos
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=var_predictora, y='tasa_conv', data=grupo)
        plt.grid()
        plt.xlabel(var_predictora)
        plt.ylabel('Tasa de conversión (%)')
        plt.title('Tasa de conversión en función de ' + var_predictora)
        plt.show()
    elif type == 'bar': #Datos categoricos
        plt.figure(figsize=(14, 6))
        sns.barplot(x=var_predictora, y='tasa_conv', data=grupo, order=order)
        plt.grid()
        plt.xlabel(var_predictora)
        plt.ylabel('Tasa de conversión (%)')
        plt.title('Tasa de conversión en función de ' + var_predictora)
        plt.show()
    elif type == 'scatter': #Datos categoricos
        plt.figure(figsize=(14, 6))
        sns.scatterplot(x=var_predictora, y='tasa_conv', data=grupo)
        plt.grid()
        plt.xlabel(var_predictora)
        plt.ylabel('Tasa de conversión (%)')
        plt.title('Tasa de conversión en función de ' + var_predictora)
        plt.show()

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
col_num1=['Agregación al carrito','Compra del producto']
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16,8))  # Ajustar el tamaño de la figura
fig.subplots_adjust(hspace=0.5)
for i, col in enumerate(col_num1):
    sns.countplot(x=col, data=data, ax=ax[i])
    ax[i].set_title(col)
    ax[i].tick_params(axis='x', labelrotation=90)  # Rotar las etiquetas del eje x
plt.show()
####### ANALISIS UNIVARIADO
#Relacion entre la variable numerica y la variable a predecir 
# graficar tasas de conversion Acceso a la página y Compra del producto
graficar_tasas_conversion('Acceso a la página','Compra del producto', type='line')
#Creamos subgrupos de accesos a la pagina y calculamos la tasa de conversion para cada caso
data.loc[:,'grupos de acceso']="0-20"
data.loc[(data['Acceso a la página']>20),'grupos de acceso']=">20"
graficar_tasas_conversion('grupos de acceso','Compra del producto',type='bar')
# graficar tasas de conversion Tiempo en la página y Compra del producto
graficar_tasas_conversion('Tiempo en la página','Compra del producto',type='scatter')
##Creamos subgrupos de accesos a la pagina y calculamos la tasa de conversion para cada caso
data.loc[:,'grupos de tiempo']="0-40"
data.loc[(data['Tiempo en la página']>=40)&(data['Tiempo en la página']<80),'grupos de tiempo']="40-80"
data.loc[(data['Tiempo en la página']>=80),'grupos de tiempo']=">=80"
graficar_tasas_conversion('grupos de tiempo','Compra del producto',type='bar')
# graficar tasas de conversion Tiempo en la página y Compra del producto
graficar_tasas_conversion('Agregación al carrito','Compra del producto',type='bar')
##Creamos subgrupos de accesos a la pagina y calculamos la tasa de conversion para cada caso


    





















