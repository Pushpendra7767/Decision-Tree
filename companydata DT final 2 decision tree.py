# import pacakages
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.datasets import load_boston
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error

# print dataset.
comydata = pd.read_csv("C:\\Users\\ACER\\Desktop\\scrap\\Decision tree\\CompanyData.csv")
comydata.head()
colnames = list(comydata.columns)

# Convert string into unique integer values.
comydata.loc[comydata['ShelveLoc']=='Good','ShelveLoc']=0
comydata.loc[comydata['ShelveLoc']=='Bad','ShelveLoc']=1
comydata.loc[comydata['ShelveLoc']=='Medium','ShelveLoc']=2

comydata.loc[comydata['Urban']=='Yes','Urban']=1
comydata.loc[comydata['Urban']=='No','Urban']=0

comydata.loc[comydata['US']=='Yes','US']=1
comydata.loc[comydata['US']=='No','US']=0

# plot histogram
plt.hist(comydata['Urban'],edgecolor='k')
plt.grid(axis='y')
plt.show()

# check skewnee & kurtosis
# graph plot
comydata['Sales'].skew()
comydata['Sales'].kurt()
plt.hist(comydata['Sales'],edgecolor='k')
sns.distplot(comydata['Sales'],hist=False)
plt.boxplot(comydata['Sales'])

comydata['CompPrice'].skew()
comydata['CompPrice'].kurt()
plt.hist(comydata['CompPrice'],edgecolor='k')
sns.distplot(comydata['CompPrice'],hist=False)
plt.boxplot(comydata['CompPrice'])

# split train & test dataset
x = comydata.drop(['Sales'],axis=1)
y = comydata['Sales']
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# implement linear regression and plot graph.
comydata1 = LinearRegression()
comydata1.fit(X_train,Y_train)
pred = comydata1.predict(X_test)
comydata1.score(X_train,Y_train)

x_l = range(len(X_test))
plt.scatter(x_l, Y_test, s=5, color="green", label="original")
plt.plot(x_l, pred, lw=0.8, color="black", label="predicted")
plt.legend()
plt.show()

# implement rigid regression & plt scatter graph.
alphas = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1,0.5, 1]
for a in alphas:
   model = Ridge(alpha=a, normalize=True).fit(X_train,Y_train) 
   score = model.score(X_train,Y_train)
   pred_y = model.predict(X_test)
   mse = mean_squared_error(Y_test, pred_y) 
   print("Alpha:{0:.6f}, R2:{1:.3f}, MSE:{2:.2f}, RMSE:{3:.2f}"
    .format(a, score, mse, np.sqrt(mse)))

# print ypred, score, mse
ridge_mod=Ridge(alpha=0.01, normalize=True).fit(X_train,Y_train)
ypred = ridge_mod.predict(X_test)
score = model.score(X_test,Y_test)
mse = mean_squared_error(Y_test,ypred)
print("R2:{0:.3f}, MSE:{1:.2f}, RMSE:{2:.2f}"
   .format(score, mse,np.sqrt(mse)))

# plot scatter diagram
x_ax = range(len(X_test))
plt.scatter(x_ax, Y_test, s=5, color="blue", label="original")
plt.plot(x_ax, ypred, lw=0.8, color="red", label="predicted")
plt.legend()
plt.show()

































































