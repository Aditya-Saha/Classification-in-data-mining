import pandas as pd

df=pd.read_csv("titanic.csv")
print(df.columns)

df.drop(['PassengerId','Name','Sex','Pclass','SibSp','Parch','Ticket','Fare'],axis=1,inplace=True)

from sklearn import preprocessing

df['Age']=df['Age'].fillna(df['Age'].mean())

ob1=preprocessing.LabelEncoder()
ob2=preprocessing.LabelEncoder()
df['Embarked']= ob1.fit_transform(df['Embarked'])
df['Cabin']=ob2.fit_transform(df['Cabin'])

#print(df.head())

df['Cabin']=df['Cabin'].fillna(df['Cabin'].mean())

df['Embarked']=df['Embarked'].fillna(df['Embarked'].mean())

predictor=df.drop(['Survived'],axis=1)
#----------------------------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(predictor,df['Survived'],test_size=0.3)
#----------------------------------------------------------------------------------------------------
from sklearn.neural_network import MLPClassifier

model1 = MLPClassifier(max_iter=1000,hidden_layer_sizes=(30,16,8,4,2), activation='relu',solver= 'sgd',batch_size='auto' )#epoch is max_iter
#print(MLPClassifier)
model1.fit(x_train,y_train)
y_pred=model1.predict(x_test)

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,y_pred))
#print(classification_report(y_test,y_pred))

#--------------------------------------------------------------------------------
model2 = MLPClassifier(max_iter=1500,hidden_layer_sizes=(30,16,8,4,2), activation='logistic',solver= 'adam',batch_size=1000 )#epoch is max_iter
#print(MLPClassifier)
model2.fit(x_train,y_train)
y_pred=model2.predict(x_test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,y_pred))
#print(classification_report(y_test,y_pred))







#https://github.com/suganyamurthy/ML-Code/blob/d3fa601eb88c1c4ef238cf35bc85f3c1a826ab33/multi%20layer.ipynb
