import pandas as pd 
import scipy as sp
from sklearn import preprocessing

DB = pd.read_csv('Archivos\ecommerce_data.csv')

x = sp.stats.describe(DB)
print (x)

