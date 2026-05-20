# Machine-learning-
My daily machine  learning journey with notes, code, and mini projects in mobile phone 📱 

Day-1
Linear Regression Basics
We’ll predict salary from years of experience.
Go to terminal inside pydroid-3 app and install:
pip install pandas
pip install scikit-learn

# Day 1 - Linear Regression

# Import libraries
import pandas as pd
from sklearn.linear_model import LinearRegression

# Create dataset
data = {
    "Experience": [1, 2, 3, 4, 5],
    "Salary": [25000, 30000, 35000, 40000, 45000]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Input and output
X = df[["Experience"]]
y = df["Salary"]

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict salary for 6 years experience
prediction = model.predict([[6]])

# Output
print("Predicted Salary:", prediction[0])
Predicted Salary: 50000.0

Day 1 of my Machine Learning Journey 🚀

Built my first Linear Regression model using Python and scikit-learn.

Learned:

- datasets
- training models
- prediction
- model.fit()
- model.predict()

Starting small but staying consistent..


# Day 2 - Train Test Split and Graph

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Dataset
data = {
    "Experience": [1,2,3,4,5,6,7,8,9,10],
    "Salary": [25000,30000,35000,40000,45000,
               50000,55000,60000,65000,70000]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Input and output
X = df[["Experience"]]
y = df["Salary"]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Print predictions
print("Predictions:")
print(predictions)

# Accuracy score
score = model.score(X_test, y_test)

print("\nModel Accuracy:", score)

# Graph
plt.scatter(X, y)

# Regression line
plt.plot(X, model.predict(X))

plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Experience vs Salary")

plt.show()


# Day 3 - Student Pass Prediction

# Import libraries
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Dataset
data = {
    "Hours": [1,2,3,4,5,6,7,8,9,10],
    "Pass":  [0,0,0,0,1,1,1,1,1,1]
}

# Convert into DataFrame
df = pd.DataFrame(data)

# Input and output
X = df[["Hours"]]
y = df["Pass"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Create model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Predictions:", predictions)

print("Accuracy:", accuracy)

# Predict new student
new_prediction = model.predict([[7]])

if new_prediction[0] == 1:
    print("Student Will Pass")
else:
    print("Student Will Fail")

    # Day 4 - Model Evaluation Metrics

# Import libraries
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Dataset
data = {
    "Hours": [1,2,3,4,5,6,7,8,9,10],
    "Pass":  [0,0,0,0,1,1,1,1,1,1]
}

# DataFrame
df = pd.DataFrame(data)

# Input and output
X = df[["Hours"]]
y = df["Pass"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Create model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, predictions)

precision = precision_score(y_test, predictions)

recall = recall_score(y_test, predictions)

f1 = f1_score(y_test, predictions)

matrix = confusion_matrix(y_test, predictions)

# Print results
print("Accuracy:", accuracy)

print("Precision:", precision)

print("Recall:", recall)

print("F1 Score:", f1)

print("\nConfusion Matrix:")
print(matrix)
If output is:

[[1 0]
 [0 1]]
It means:
.correct fail prediction
.correct pass prediction
.zero mistakes

# Day 5 - KNN Classifier

# Import libraries
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Dataset
data = {
    "Hours": [1,2,3,4,5,6,7,8,9,10],
    "Pass":  [0,0,0,0,1,1,1,1,1,1]
}

# Create DataFrame
df = pd.DataFrame(data)

# Input and output
X = df[["Hours"]]
y = df["Pass"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Create KNN model
model = KNeighborsClassifier(n_neighbors=3)

# Train model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

# Print results
print("Predictions:", predictions)

print("Accuracy:", accuracy)

# Predict new student
new_prediction = model.predict([[2]])


# Day 6 - Decision Tree Classifier

# Import libraries
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Dataset
data = {
    "Hours": [1,2,3,4,5,6,7,8,9,10],
    "Pass":  [0,0,0,0,1,1,1,1,1,1]
}

# Create DataFrame
df = pd.DataFrame(data)

# Input and output
X = df[["Hours"]]
y = df["Pass"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Create Decision Tree model
model = DecisionTreeClassifier()

# Train model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

# Print results
print("Predictions:", predictions)

print("Accuracy:", accuracy)

# Predict new student
new_prediction = model.predict([[3]])

if new_prediction[0] == 1:
    print("Student Will Pass")
else:
    print("Student Will Fail")

Predictions: [1 0]
Accuracy: 1.0
Student Will Fail

Day 6 of my Machine Learning Journey 🚀

Built a Decision Tree classifier today 🌳

Learned:

- decision-based AI
- classification using trees
- model training
- prediction logic

Understanding how ML models make decisions step-by-step 

# Day 7 - Random Forest Classifier

# Import libraries
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Dataset
data = {
    "Hours": [1,2,3,4,5,6,7,8,9,10],
    "Pass":  [0,0,0,0,1,1,1,1,1,1]
}

# Create DataFrame
df = pd.DataFrame(data)

# Input and output
X = df[["Hours"]]
y = df["Pass"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=10,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

# Print results
print("Predictions:", predictions)

print("Accuracy:", accuracy)

# Predict new student
new_prediction = model.predict([[8]])

if new_prediction[0] == 1:
    print("Student Will Pass")
else:
    print("Student Will Fail")

# Day 8 - Naive Bayes Classifier

# Import libraries
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Dataset
data = {
    "Hours": [1,2,3,4,5,6,7,8,9,10],
    "Pass":  [0,0,0,0,1,1,1,1,1,1]
}

# Create DataFrame
df = pd.DataFrame(data)

# Input and output
X = df[["Hours"]]
y = df["Pass"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Create Naive Bayes model
model = GaussianNB()

# Train model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

# Print results
print("Predictions:", predictions)

print("Accuracy:", accuracy)

# Predict new student
new_prediction = model.predict([[6]])

if new_prediction[0] == 1:
    print("Student Will Pass")
else:
    print("Student Will Fail")
