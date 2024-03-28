import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Cargar el conjunto de datos
    dt = pd.read_csv('./Archivos/telecom_churn_data.csv')

    # Dividir el conjunto de datos en características (features) y variable objetivo (target)
    dt_features = dt.drop(['Churn'], axis=1) 
    dt_target = dt['Churn']

    # Normalización de características
    dt_features_normalized = StandardScaler().fit_transform(dt_features)

    # Convertir la matriz numpy de características normalizadas en un DataFrame de pandas
    dt_features_df = pd.DataFrame(dt_features_normalized, columns=dt_features.columns)
    
    # Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(dt_features_df, dt_target, test_size=0.3, random_state=42)

    # Modelado de Red Neuronal
    clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=100, activation='relu', solver='adam', random_state=42)
    clf.fit(X_train, y_train)

    # Predicciones de probabilidad de churn
    churn_probabilities = clf.predict_proba(X_test)[:, 1]  # Probabilidades de la clase positiva (churn)

    # Convertir las probabilidades de churn en porcentaje
    churn_probabilities_percentage = churn_probabilities * 100

    # Mostrar el resultado del churn en porcentaje
    print("Probabilidad de churn (%):")
    for prob in churn_probabilities_percentage:
        print("{:.2f}%".format(prob))

    # Evaluación del Modelo
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", confusion_mat)

    # Visualización de la distribución de características
    sns.set(style="whitegrid")
    for feature in dt_features_df.columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(dt_features_df[feature], kde=True, bins=20)
        plt.title(f'Distribución de {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frecuencia')
        plt.show()

    # Visualización de relaciones entre características y variable objetivo (Churn)
    sns.set(style="whitegrid")

    # Diagrama de dispersión
    for feature in dt_features_df.columns:
        sns.scatterplot(x=feature, y='Churn', data=pd.concat([dt_features_df, dt_target], axis=1))
        plt.title(f'Relación entre {feature} y Churn')
        plt.xlabel(feature)
        plt.ylabel('Churn')
        plt.show()

    # Diagrama de barras
    for feature in dt_features_df.columns:
        sns.barplot(x='Churn', y=feature, data=pd.concat([dt_features_df, dt_target], axis=1))
        plt.title(f'Distribución de {feature} por Churn')
        plt.xlabel('Churn')
        plt.ylabel(feature)
        plt.show()

    # Heatmap de correlación
    plt.figure(figsize=(10, 8))
    sns.heatmap(pd.concat([dt_features_df, dt_target], axis=1).corr(), annot=True, cmap='coolwarm')
    plt.title('Heatmap de correlación entre características y Churn')
    plt.show()

    # Visualización de la convergencia de la función de pérdida
    plt.plot(clf.loss_curve_)
    plt.title('Convergencia de la función de pérdida')
    plt.xlabel('Iteraciones')
    plt.ylabel('Pérdida')
    plt.show()

    # Visualizar la matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()
