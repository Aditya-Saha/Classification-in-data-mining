import pandas as pd
titanic =pd.read_csv("titanic.csv")

#print(titanic.head())

titanic.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'], axis=1, inplace=True)

#print(titanic.head())

predictor = titanic.drop('Survived',axis='columns')
target=titanic.Survived

#print(predictor)

#print(target)


predictor['Sex'].replace(['female','male'],[0,1],inplace=True)

print(predictor.head())

predictor.Age = predictor.Age.fillna(predictor.Age.mean())

print(predictor.head())

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(predictor,target,test_size=0.2)

#-----------------------------------------------------------------------------------
from sklearn.naive_bayes import GaussianNB
print("---------------------------------using gaussian naive bayes----------------------")
model=GaussianNB()
model.fit(x_train,y_train)
print(model.score(x_test,y_test))




