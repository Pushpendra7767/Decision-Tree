# import pacakages
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pylab as plt
import seaborn as sns
# print dataset.
frcheck = pd.read_csv("C:\\Users\\ACER\\Desktop\\scrap\\Decision tree\\Fraudcheck.csv")
frcheck.head()
colnames = list(frcheck.columns)
# Convert string into unique integer values.
frcheck.loc[frcheck['Undergrad']=='YES','Undergrad']=1
frcheck.loc[frcheck['Undergrad']=='NO','Undergrad']=0

frcheck.loc[frcheck['Marital.Status']=='Single','Marital.Status']=1
frcheck.loc[frcheck['Marital.Status']=='Married','Marital.Status']=2
frcheck.loc[frcheck['Marital.Status']=='Divorced','Marital.Status']=3

frcheck.loc[frcheck['Urban']=='YES','Urban']=1
frcheck.loc[frcheck['Urban']=='NO','Urban']=0

frcheck.loc[frcheck['Taxable.Income']<=30000,'Taxable.Income']=1
frcheck.loc[frcheck['Taxable.Income']>30000,'Taxable.Income']=0
# plot histogram
plt.hist(frcheck['Urban'],edgecolor='k')
plt.grid(axis='y')
plt.show()

# check skewnee & kurtosis
# graph plot
frcheck['City.Population'].skew()
frcheck['City.Population'].kurt()
plt.hist(frcheck['City.Population'],edgecolor='k')
sns.distplot(frcheck['City.Population'],hist=False)
plt.boxplot(frcheck['City.Population'])

frcheck['Work.Experience'].skew()
frcheck['Work.Experience'].kurt()
plt.hist(frcheck['Work.Experience'],edgecolor='k')
sns.distplot(frcheck['Work.Experience'],hist=False)
plt.boxplot(frcheck['Work.Experience'])

# count values 
frcheck['Taxable.Income'].value_counts()
# split train & test dataset
y = frcheck['Taxable.Income']
x = frcheck.drop(['Taxable.Income'],axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.33, random_state=42)
# implement decission tree classifier
# criterion = 'entropy',class_weight='balanced'
fr1 = DecisionTreeClassifier(criterion = 'entropy',class_weight='balanced',)
fr1.fit(X_train,Y_train)
pred = fr1.predict(X_test)
accuracy_score(Y_test,pred)
confusion_matrix(Y_test,pred,labels=[1,0])
# criterion = 'gini',class_weight='balanced',splitter='random',max_features='int'
fr2 = DecisionTreeClassifier(criterion = 'gini',class_weight='balanced',splitter='random',max_features='int')
fr2.fit(X_train,Y_train)
pred = fr2.predict(X_test)
accuracy_score(Y_test,pred)
confusion_matrix(Y_test,pred,labels=[1,0])

# Bagging classifier
# max_samples=0.5,max_features=1.0,n_estimators=10
fr4 = BaggingClassifier(DecisionTreeClassifier(),max_samples=0.5,max_features=1.0,n_estimators=10)
fr4.fit(X_train,Y_train)
pred4 = fr4.predict(X_test)
accuracy_score(Y_test,pred4)
confusion_matrix(Y_test,pred4,labels=[1,0])
# max_samples=0.5,max_features=1.0,n_estimators=10,random_state='int'
fr5 = BaggingClassifier(DecisionTreeClassifier(),max_samples=0.5,max_features=1.0,n_estimators=10,random_state='int')
fr5.fit(X_train,Y_train)
pred4 = fr5.predict(X_test)
accuracy_score(Y_test,pred4)
confusion_matrix(Y_test,pred4,labels=[1,0])


























































































