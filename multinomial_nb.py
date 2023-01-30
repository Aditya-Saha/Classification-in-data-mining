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
from sklearn.naive_bayes import MultinomialNB
print("---------------------------------using gaussian naive bayes----------------------")
model=MultinomialNB()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(y_pred)

from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_pred)
print(f"The accuracy_score is {acc*100}%") 

#print(model.score(x_test,y_test))


#--------------------------------------------------------------------
#applying decision tree classifier model using different criterions and splitter.

from sklearn.tree import DecisionTreeClassifier
clf_gini=DecisionTreeClassifier(criterion='gini',splitter='random')
clf_gini.fit(x_train,y_train)
y_pred=clf_gini.predict(x_test)
acc=accuracy_score(y_test,y_pred)
print(f"The accuracy_score with gini random is {acc*100}%") 

from sklearn.metrics import confusion_matrix
c=confusion_matrix(y_test,y_pred)
print(c)

TP=c[1][1]
TN=c[0][0]
FP=c[0][1]
FN=c[1][0]


print(f"True negative [0,0] ={c[0][0]}");
print(f"True positive[1,1] ={c[1][1]}");
print(f"false positive [0,1] ={c[0][1]}");
print(f"false negative [1,0] ={c[1][0]}");

print(f"\n\nMiscalculation rate is {(FP+FN)/(TP+TN+FP+FN)}")

from sklearn.metrics import precision_score
p=precision_score(y_test,y_pred)
print(f"\n\nPrecision rate is {p}")

from sklearn.metrics import recall_score
r=recall_score(y_test,y_pred)
print(f"\n\nPrecision rate is {r}")

from sklearn.metrics import f1_score
f=f1_score(y_test,y_pred)
print(f"\n\nPrecision rate is {f}")

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test,y_pred)
print(f"mean square error is {mse}")


#----------------------------------------------------------------------------
from sklearn.tree import DecisionTreeClassifier
clf_gini=DecisionTreeClassifier(criterion='gini',splitter='best')
clf_gini.fit(x_train,y_train)
y_pred=clf_gini.predict(x_test)
acc=accuracy_score(y_test,y_pred)
print(f"The accuracy_score with gini best is {acc*100}%") 
#-------------------------------------------------------------------------
from sklearn.tree import DecisionTreeClassifier
clf_entropy=DecisionTreeClassifier(criterion='entropy',splitter='random')
clf_entropy.fit(x_train,y_train)
y_pred=clf_entropy.predict(x_test)
acc=accuracy_score(y_test,y_pred)
print(f"The accuracy_score with entropy random is {acc*100}%") 
#--------------------------------------------------------------------------

from sklearn.tree import DecisionTreeClassifier
clf_entropy=DecisionTreeClassifier(criterion='entropy',splitter='best')
clf_entropy.fit(x_train,y_train)
y_pred=clf_entropy.predict(x_test)
acc=accuracy_score(y_test,y_pred)
print(f"The accuracy_score with gini random is {acc*100}%") 


