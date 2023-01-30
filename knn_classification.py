import pandas as pd
df=pd.read_csv("haberman.csv")

print(df.head())

df1=df.drop(['Survival_years'],axis=1)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split( df1, df['Survival_years'],test_size=0.2)


from sklearn.neighbors import KNeighborsClassifier

knn= KNeighborsClassifier(n_neighbors=5,metric='minkowski')

knn.fit(x_train,y_train)

y_pred=knn.predict(x_test)

from sklearn.metrics import accuracy_score

acc=accuracy_score(y_test,y_pred)

print(acc)

