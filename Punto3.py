import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Importar archivo csv 
data = pd.read_csv('Archivos/ecommerce_data.csv')

print(data.info()) 
print(data.describe()) 