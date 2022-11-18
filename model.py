import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn import preprocessing

dataset=pd.read_csv('evdataset.csv')

dataset.replace({'Drive':{'Rear':2,'Front':0,'AWD':1}},inplace=True)

x1=dataset.drop(['link','Electric Range','id','Make','link','City - Cold Weather','Highway - Cold Weather','Combined - Cold Weather','City - Mild Weather','Highway - Mild Weather','Combined - Mild Weather','Seats'],axis=1)
y=dataset[['Electric Range','City - Cold Weather','Highway - Cold Weather','Combined - Cold Weather','City - Mild Weather','Highway - Mild Weather', 'Combined - Mild Weather']]

x = preprocessing.normalize(x1)

from sklearn.model_selection import train_test_split
x = pd.DataFrame(x)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LinearRegression
ml=LinearRegression()
ml.fit(x_train,y_train)
pk=open('model1.pkl','wb')
pickle.dump(ml,pk)

