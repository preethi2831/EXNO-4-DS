```
NAME : Preethika N
REG NO : 212223040130
```

# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv("income.csv",na_values=[ " ?"])
data
```
![image](https://github.com/user-attachments/assets/2d6c361f-48f8-4b34-80fd-a06d17a0daa4)

```
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/62e6d0ad-de0a-41c5-960c-4116383d1d4a)
```
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/user-attachments/assets/155a7f79-96a3-4638-8dd6-732d26ce7f66)
```
data2=data.dropna(axis=0)
data2
```
![image](https://github.com/user-attachments/assets/ae6349e7-43c3-4f20-bd9a-84b6c8aca2fc)
```
sal=data["SalStat"]

data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/user-attachments/assets/e182875b-8f69-411a-9690-10618a9908ea)
```
sal2=data2['SalStat']

dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/user-attachments/assets/dee49677-2990-408e-b93f-1c07a66e712b)
```
data2
```
![image](https://github.com/user-attachments/assets/0c3a0e78-a377-460e-8d0d-cf001e77af1d)
```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![image](https://github.com/user-attachments/assets/b4b63d88-1d5b-4aa3-b6f1-ba23895ed80b)
```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/user-attachments/assets/1afe35af-357f-44cc-b4c3-ccb67be645d2)
```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/user-attachments/assets/a4922365-9be3-4ac4-b684-7eba15c85d3f)
```
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/user-attachments/assets/16a0d4ca-4ac2-469b-9d85-ef3f236f4782)
```
x=new_data[features].values
print(x)
```
![image](https://github.com/user-attachments/assets/cc99b19a-18d5-462b-b728-5ab14b66130b)
```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

KNN_classifier=KNeighborsClassifier(n_neighbors = 5)

KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/user-attachments/assets/ff6c8214-692f-4579-a1c1-3405e30cff10)
```
prediction=KNN_classifier.predict(test_x)

confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![image](https://github.com/user-attachments/assets/ea140901-1446-4c7f-87ca-32f4996d2dc8)
```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![image](https://github.com/user-attachments/assets/cd371055-7d98-4df7-bcf8-d7e22651cfbb)
```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![image](https://github.com/user-attachments/assets/a96487d7-c6ab-4fe1-a173-943f59dc0762)
```
data.shape
```
![image](https://github.com/user-attachments/assets/9083ea28-b6c5-4055-97e6-74e6496e110f)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]

selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)

selected_feature_indices=selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/44c71c58-c0b8-4545-a743-99e93f7d8b4b)
```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/user-attachments/assets/03f342e0-728b-4b6f-8d68-305f6a68af40)
```
tips.time.unique()
```
![image](https://github.com/user-attachments/assets/661f07d5-af2b-49bb-afb5-edbd978e1762)
```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![image](https://github.com/user-attachments/assets/ecb862dd-d2a8-4070-ae33-ad57879b7ea0)
```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
![image](https://github.com/user-attachments/assets/97875b1c-aa7a-43b1-aaee-eed6875b3e34)

# RESULT:
Thus the program to read the given data and perform Feature Scaling and Feature Selection process and save the data to a file is been executed.      
