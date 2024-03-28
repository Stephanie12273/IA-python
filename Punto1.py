import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

# Importar archivo csv 
data = pd.read_csv('Archivos/SVM.csv')

# Funcion para graficar tasas de conversion 
def graficar_tasas_conversion(var_predictora, var_predecir, type='line', order=None):
    x, y = var_predictora, var_predecir
    grupo = data.groupby(x)[y].mean() * 100
    grupo = grupo.rename('tasa_conv').reset_index()
    if type == 'line':
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=var_predictora, y='tasa_conv', data=grupo)
    elif type == 'bar':
        plt.figure(figsize=(10, 6))
        sns.barplot(x=var_predictora, y='tasa_conv', data=grupo, order=order)
    elif type == 'scatter':
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=var_predictora, y='tasa_conv', data=grupo)
    plt.grid()
    plt.xlabel(var_predictora)
    plt.ylabel('Tasa de conversión (%)')
    plt.title('Tasa de conversión en función de ' + var_predictora)
    plt.show()

# ANALISIS EXPLORATORIO
print(data.info()) 
print(data.describe()) 

col_num = ['Acceso a la página', 'Tiempo en la página']
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
fig.subplots_adjust(hspace=0.5)
steps = {'Acceso a la página': 1, 'Tiempo en la página': 2} 
for i, col in enumerate(col_num):
    sns.histplot(x=col, data=data, ax=ax[i], kde=True)
    ax[i].set_title(col)
    ax[i].set_xticks(range(0, int(data[col].max()) + 1, steps[col]))
    ax[i].set_xticklabels(ax[i].get_xticks(), rotation=90)
plt.show()

col_num1=['Agregación al carrito', 'Compra del producto']
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
fig.subplots_adjust(hspace=0.5)
for i, col in enumerate(col_num1):
    sns.countplot(x=col, data=data, ax=ax[i])
    ax[i].set_title(col)
    ax[i].tick_params(axis='x', labelrotation=90)
plt.show()

# ANALISIS UNIVARIADO
graficar_tasas_conversion('Acceso a la página', 'Compra del producto', type='line')
data['grupos de acceso'] = pd.cut(data['Acceso a la página'], bins=[0, 20, float('inf')], labels=['0-20', '>20'])
graficar_tasas_conversion('grupos de acceso', 'Compra del producto', type='bar')
graficar_tasas_conversion('Tiempo en la página', 'Compra del producto', type='scatter')
data['grupos de tiempo'] = pd.cut(data['Tiempo en la página'], bins=[0, 40, 80, float('inf')], labels=['0-40', '40-80', '>=80'])
graficar_tasas_conversion('grupos de tiempo', 'Compra del producto', type='bar')
graficar_tasas_conversion('Agregación al carrito', 'Compra del producto', type='bar')

# NORMALIZACION Y AJUSTE DE DATOS
x1 = data.drop(['Compra del producto', 'grupos de acceso', 'grupos de tiempo'], axis=1)
y1 = data['Compra del producto']  
scaler = StandardScaler()
x1_normalized = scaler.fit_transform(x1)
X_train, X_test, y_train, y_test = train_test_split(x1_normalized, y1, test_size=0.2, random_state=42)

# MODELOS SVM
modelos = {'Lineal': SVC(kernel='linear'), 'Polinomial': SVC(kernel='poly'), 'Sigmoide': SVC(kernel='sigmoid'), 'RBF': SVC(kernel='rbf')}
for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Accuracy {nombre}: {accuracy}, Recall {nombre}: {recall}, F1 Score {nombre}: {f1}")
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix {nombre}:")
    print(conf_matrix)
    plt.figure(figsize=(10, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Compra', 'Compra'], yticklabels=['No Compra', 'Compra'])
    plt.title(f'Matriz de Confusión {nombre}')
    plt.xlabel('Predicción')
    plt.ylabel('Verdadero')
    plt.show()

# Seleccionando solo tres características para la visualización
X_3d = data[['Acceso a la página', 'Tiempo en la página', 'Agregación al carrito']]
y_3d = data['Compra del producto']
# Normalizando las características
scaler = StandardScaler()
X_3d_scaled = scaler.fit_transform(X_3d)
# Dividiendo el conjunto de datos en entrenamiento y prueba
X_train_3d, X_test_3d, y_train_3d, y_test_3d = train_test_split(X_3d_scaled, y_3d, test_size=0.2, random_state=42)
# Entrenando el modelo SVM
model_3d = SVC(kernel='linear')
model_3d.fit(X_train_3d, y_train_3d)
##VISUALIZACION
# Creamos la figura 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
# Coloreamos los puntos basados en su clasificación real
ax.scatter(X_train_3d[:, 0], X_train_3d[:, 1], X_train_3d[:, 2], c=y_train_3d, cmap='viridis', marker='o')
ax.set_xlabel('Acceso a la página')
ax.set_ylabel('Tiempo en la página')
ax.set_zlabel('Agregación al carrito')
plt.title('Visualización 3D de las Clasificaciones SVM')
plt.show()




















