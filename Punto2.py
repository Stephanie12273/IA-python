import pandas as pd 
import sklearn 
import matplotlib.pyplot as plt 

#importamos modulos 
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

from sklearn.linear_model import LogisticRegression 

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__== "__main__":
    dt= pd.read_csv('./Archivos/telecom_churn_data.csv')
    #print(dt.head(5))
    dt_features = dt.drop(['Churn'], axis=1) 
    dt_target = dt['Churn']

    dt_features = StandardScaler().fit_transform(dt_features)
    
    X_train, X_test, y_train, y_test = train_test_split(dt_features, dt_target, test_size=0.3, random_state=42)
    print(X_train.shape)
    print(y_train.shape)

    #n_components = min(n_muestras, n_features)
    #pca = PCA(n_components=)

    
