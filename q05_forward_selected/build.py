# %load q04_encoding/build.py
# Default imports
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import numpy as np


data = pd.read_csv('data/house_prices_multivariate.csv')
model=LinearRegression()
df=data

# Write your code here:
def forward_selected (df,model):
    var1=[]
    var2=[]
    np.random.seed(6)
    features=df.iloc[:,:-1]
    target=df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(features,target,test_size=0.3)
    features=['OverallQual','GrLivArea','BsmtFinSF1','GarageCars','KitchenAbvGr','1stFlrSF','YearRemodAdd','LotArea','MasVnrArea','WoodDeckSF']
    for i in features:
        var1.append(i)
        model.fit(X_train[var1],y_train)
        y_pred=model.predict(X_test[var1])
        acc=r2_score(y_test,y_pred)
        if not var2:
            var2.append(acc)
        elif acc>var2[-1]:
            var2.append(acc)
        else:
            var1.remove(i)


    return var1,var2

#forward_selected (df,model)
