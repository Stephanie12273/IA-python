import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import numpy as np

# Importar archivo csv 
data=pd.read_csv('Archivos\SVM.csv')

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
        plt.figure(figsize=(10, 6))
        sns.barplot(x=var_predictora, y='tasa_conv', data=grupo, order=order)
        plt.grid()
        plt.xlabel(var_predictora)
        plt.ylabel('Tasa de conversión (%)')
        plt.title('Tasa de conversión en función de ' + var_predictora)
        plt.show()
    elif type == 'scatter': #Datos categoricos
        plt.figure(figsize=(10, 6))
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
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10,6))  # Ajustar el tamaño de la figura
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
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,6))  # Ajustar el tamaño de la figura
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

########## NORMALIZACION Y AJUSTE DE DATOS ##############
# Dividir los datos en características (predictoras) y variable objetivo
x1 = data.drop(['Compra del producto', 'grupos de acceso', 'grupos de tiempo'], axis=1)
y1 = data['Compra del producto']  # Variable objetivo
# Normalizar las características
scaler = StandardScaler()
x1_normalized = scaler.fit_transform(x1)
print(x1_normalized)
# Dividir los datos normalizados en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(x1_normalized, y1, test_size=0.2, random_state=42)
#imprimir los conjunto de datos de entrenamiento y prueba
print("Caracteristicas normalizada(entrenamiento):",X_train)
print("Compra del carrito(entrenamiento):",y_train)
print("Caracteristicas normalizada(prueba):",X_test)
print("Compra del carrito(prueba):",y_test)
# Entrenar el modelo SVM kernel lineal
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
# Predicciones en el conjunto de prueba
y_pred = svm_model.predict(X_test)
# Calcular métricas de rendimiento
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Accuracy lineal:", accuracy)
print("Recall lineal:", recall)
print("F1 Score lineal:", f1)
# Mostrar la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
#visualizar matriz de confusion
plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Verdadero')
plt.show()
# Entrenar el modelo SVM kernel poly
svm_modelpoly = SVC(kernel='poly')
svm_modelpoly.fit(X_train, y_train)
# Predicciones en el conjunto de prueba
y_pred = svm_modelpoly.predict(X_test)
# Calcular métricas de rendimiento
accuracyPoly = accuracy_score(y_test, y_pred)
recallPoly = recall_score(y_test, y_pred)
f1Poly = f1_score(y_test, y_pred)
print("Accuracy poly:", accuracyPoly)
print("Recall poly:", recallPoly)
print("F1 Score poly:", f1Poly)
# Mostrar la matriz de confusión
conf_matrixPoly = confusion_matrix(y_test, y_pred)
print("Confusion Matrix poly:")
print(conf_matrixPoly)
#visualizar matriz de confusion
plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrixPoly, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión poly')
plt.xlabel('Predicción')
plt.ylabel('Verdadero')
plt.show()
# Entrenar el modelo SVM kernel sigmoidal
svm_modelsig = SVC(kernel='sigmoid')
svm_modelsig.fit(X_train, y_train)
# Predicciones en el conjunto de prueba
y_pred = svm_modelsig.predict(X_test)
# Calcular métricas de rendimiento
accuracysig = accuracy_score(y_test, y_pred)
recallsig = recall_score(y_test, y_pred)
f1sig = f1_score(y_test, y_pred)
print("Accuracy sig:", accuracysig)
print("Recall sig:", recallsig)
print("F1 Score sig:", f1sig)
# Mostrar la matriz de confusión
conf_matrixsig = confusion_matrix(y_test, y_pred)
print("Confusion Matrix sig:")
print(conf_matrixsig)
#visualizar matriz de confusion
plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrixsig, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión sig')
plt.xlabel('Predicción')
plt.ylabel('Verdadero')
plt.show()

# Entrenar el modelo SVM kernel RBF
svm_modelrbf = SVC(kernel='rbf')
svm_modelrbf.fit(X_train, y_train)
# Predicciones en el conjunto de prueba
y_pred = svm_modelrbf.predict(X_test)
# Calcular métricas de rendimiento
accuracyrbf = accuracy_score(y_test, y_pred)
recallrbf = recall_score(y_test, y_pred)
f1rbf = f1_score(y_test, y_pred)
print("Accuracy rbf:", accuracyrbf)
print("Recall rbf:", recallrbf)
print("F1 Score rbf:", f1rbf)
# Mostrar la matriz de confusión
conf_matrixrbf = confusion_matrix(y_test, y_pred)
print("Confusion Matrix rbf:")
print(conf_matrixrbf)
#visualizar matriz de confusion
plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrixrbf, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión rbf')
plt.xlabel('Predicción')
plt.ylabel('Verdadero')
plt.show()












    





















