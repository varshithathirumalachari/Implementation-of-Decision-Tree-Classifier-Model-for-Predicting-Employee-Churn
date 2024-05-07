# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: VARSHITHA A T
RegisterNumber: 212221040176
*/
import pandas as pd
data=pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x = data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y = data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = "entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
print(accuracy)

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
## DATASET
![image](https://github.com/varshithathirumalachari/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/131793193/884bb491-a7f3-4979-8fb3-2e5592f5613e)
## data.info()
![image](https://github.com/varshithathirumalachari/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/131793193/24c8bfc1-a65c-48fa-814d-f1b9c714e29d)
## CHECKING IF NULL VALUES ARE PRESENT
![image](https://github.com/varshithathirumalachari/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/131793193/8b2f0cf2-f0d8-4c04-ac70-0e60ca545bc8)
## VALUE_COUNTS()
![image](https://github.com/varshithathirumalachari/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/131793193/d4ddc1e8-6ba9-4547-8f84-1e684914c834)
## DATASET AFTER ENCODING
![image](https://github.com/varshithathirumalachari/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/131793193/4e30197b-f044-41cc-b1e7-0b242ab48e25)
## X-VALUES
![image](https://github.com/varshithathirumalachari/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/131793193/1163489a-7ff8-42fd-acc0-2ae3e0f62fb1)
## ACCURACY
![image](https://github.com/varshithathirumalachari/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/131793193/3253cd2f-a5d0-4551-ab02-1d8fd10c88cc)
## dt.predict()
![image](https://github.com/varshithathirumalachari/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/131793193/9124e39e-f90a-4ba8-931a-d32754625eb2)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
