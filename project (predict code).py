# Business Problem

#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset
df=pd.read_csv(r"D:\NIT\JANUARY\19 jan project\student_info.csv")

## Discover and visualize the data to gain insights
df.head()
df.tail()
df.info()
df.describe()


plt.scatter(x =df.study_hours, y = df.student_marks)
plt.xlabel("Students Study Hours")
plt.ylabel("Students marks")
plt.title("Scatter Plot of Students Study Hours vs Students marks")
plt.show()

df.isnull().sum()
df.mean()
df

## Prepare the data for Machine Learning algorithms 
df.fillna(df.mean(),inplace=True)
df.isnull().sum()

X = df.drop("student_marks", axis = "columns")
y = df.drop("study_hours", axis = "columns")


df.head()

print("shape of X = ", X.shape)
print("shape of y = ", y.shape)

# split dataset
from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state=0)


# Select a model and train it
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(X_train,y_train)


lr.coef_

lr.intercept_



m = 3.93
c = 50.44
y  = m * 4 + c 
y

m = 3.93
c = 50.44
y  = m * 9 + c 
y

y_pred  = lr.predict(X_test)
y_pred

## Fine-tune your model
from sklearn.metrics import r2_score

acc=r2_score(y_test,y_pred)

acc1=lr.score(X_test,y_test)
print(acc)
print(acc1)

pd.DataFrame(np.c_[X_test, y_test, y_pred], columns = ["study_hours", "student_marks_original","student_marks_predicted"])



plt.scatter(X_train,y_train)

plt.scatter(X_test,y_test)

plt.scatter(y_pred,y_test)



plt.scatter(X_test, y_test)
plt.plot(X_train, lr.predict(X_train), color = "r")



plt.scatter(X_train, y_train)
plt.plot(X_train, lr.predict(X_train), color = "r")

## Save Ml Model
import joblib
joblib.dump(lr, "student_mark_predictor.pkl")

model = joblib.load("student_mark_predictor.pkl")





































































































