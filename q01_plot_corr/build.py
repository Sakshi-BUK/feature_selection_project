import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.pyplot import yticks, xticks, subplots, set_cmap

data = pd.read_csv('data/house_prices_multivariate.csv')
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
df=data
newdf = data.select_dtypes(include=numerics)

# Write your solution here:
def plot_corr(df, size=11):
    plt.set_cmap('YlOrRd')
    plt.figure(figsize=(size,size))
    sns.heatmap(df.corr())
#plot_corr(df, size=11)
