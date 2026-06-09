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

# Day 9 - Support Vector Machine (SVM)

# Import libraries
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
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

# Create SVM model
model = SVC(kernel="linear")

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
new_prediction = model.predict([[5]])

if new_prediction[0] == 1:
    print("Student Will Pass")
else:
    print("Student Will Fail")

# Day 10 - KMeans Clustering

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

# Dataset
data = {
    "Age": [18,19,20,21,22,45,46,47,48,50],
    "Spending": [15,18,16,20,22,65,70,68,72,75]
}

# Create DataFrame
df = pd.DataFrame(data)

# Input features
X = df[["Age", "Spending"]]

# Create KMeans model
model = KMeans(
    n_clusters=2,
    random_state=42
)

# Train model
model.fit(X)

# Cluster predictions
clusters = model.predict(X)

# Add clusters to dataframe
df["Cluster"] = clusters

# Print dataset
print(df)

# Visualization
plt.scatter(df["Age"], df["Spending"], c=clusters)

plt.xlabel("Age")
plt.ylabel("Spending Score")
plt.title("KMeans Clustering")

plt.show()

# Day 11 - PCA (Principal Component Analysis)

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Dataset
data = {
    "Math":    [90,85,88,75,60,95,70,65],
    "Science": [92,80,85,78,65,98,72,60],
    "English": [85,78,82,70,58,96,68,62]
}

# Create DataFrame
df = pd.DataFrame(data)

# Scale data
scaler = StandardScaler()

scaled_data = scaler.fit_transform(df)

# Apply PCA
pca = PCA(n_components=2)

principal_components = pca.fit_transform(scaled_data)

# Create new DataFrame
pca_df = pd.DataFrame(
    data=principal_components,
    columns=["PC1", "PC2"]
)

# Print transformed data
print(pca_df)

# Visualization
plt.scatter(
    pca_df["PC1"],
    pca_df["PC2"]
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Visualization")

plt.show()

# Day 12 - Feature Scaling

# Import libraries
import pandas as pd

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler
)

# Dataset
data = {
    "Age": [18,20,25,30,35],
    "Salary": [20000,30000,50000,80000,120000]
}

# Create DataFrame
df = pd.DataFrame(data)

print("Original Data:\n")
print(df)

# -----------------------------
# Standard Scaling
# -----------------------------

standard_scaler = StandardScaler()

standard_scaled = standard_scaler.fit_transform(df)

standard_df = pd.DataFrame(
    standard_scaled,
    columns=df.columns
)

print("\nStandard Scaled Data:\n")
print(standard_df)

# -----------------------------
# MinMax Scaling
# -----------------------------

minmax_scaler = MinMaxScaler()

minmax_scaled = minmax_scaler.fit_transform(df)

minmax_df = pd.DataFrame(
    minmax_scaled,
    columns=df.columns
)

print("\nMinMax Scaled Data:\n")
print(minmax_df)

# Day 13 - Handling Missing Values

import pandas as pd

from sklearn.impute import SimpleImputer

# Dataset with missing values
data = {
    "Age": [22, 25, None, 30, 35],
    "Salary": [25000, None, 40000, 50000, None]
}

df = pd.DataFrame(data)

print("Original Dataset:\n")
print(df)

# Create imputer
imputer = SimpleImputer(strategy="mean")

# Fill missing values
df_filled = pd.DataFrame(
    imputer.fit_transform(df),
    columns=df.columns
)

print("\nDataset After Filling Missing Values:\n")
print(df_filled)

# Day 14 - Complete ML Pipeline

import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Dataset
data = {
    "Hours": [1, 2, 3, None, 5, 6, 7, 8, None, 10],
    "Pass":  [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

print("Original Dataset:\n")
print(df)

# Step 1: Handle Missing Values
imputer = SimpleImputer(strategy="mean")

df["Hours"] = imputer.fit_transform(df[["Hours"]])

# Step 2: Features and Target
X = df[["Hours"]]
y = df["Pass"]

# Step 3: Feature Scaling
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Step 4: Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# Step 5: Train Model
model = LogisticRegression()

model.fit(X_train, y_train)

# Step 6: Predictions
predictions = model.predict(X_test)

# Step 7: Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nPredictions:")
print(predictions)

print("\nAccuracy:")
print(accuracy)

# New Prediction
new_student = scaler.transform([[7]])

result = model.predict(new_student)

if result[0] ==

# Day 15 - Iris Flower Classification

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()

# Features
X = iris.data

# Target
y = iris.target

# Scale features
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

# Predict a new flower
sample = [[5.1, 3.5, 1.4, 0.2]]

sample_scaled = scaler.transform(sample)

result = model.predict(sample_scaled)

flower_name = iris.target_names[result[0]]

print("Predicted Flower:", flower_name)

Day 16 — Titanic Survival Prediction 🚢
This is one of the most famous ML datasets.
Goal:
Predict whether a passenger survived the Titanic disaster.
output :
0 → Did Not Survive
1 → Survived
code:

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
titanic = fetch_openml("titanic", version=1, as_frame=True)

df = titanic.frame

# Select columns
df = df[["sex", "age", "pclass", "survived"]]

# Remove missing values
df = df.dropna()

# Encode gender
encoder = LabelEncoder()

df["sex"] = encoder.fit_transform(df["sex"])

# Features and target
X = df[["sex", "age", "pclass"]]
y = df["survived"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = DecisionTreeClassifier()

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

# New passenger prediction
sample = [[0, 25, 1]]

result = model.predict(sample)

print("Survived Prediction:", result[0])

day17_house_price_prediction.py
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Dataset
data = {
    "Size": [1000, 1200, 1500, 1800, 2000, 2200, 2500],
    "Bedrooms": [2, 2, 3, 3, 4, 4, 5],
    "Age": [10, 8, 7, 5, 4, 3, 2],
    "Price": [50, 60, 75, 90, 100, 115, 130]
}

df = pd.DataFrame(data)

# Features
X = df[["Size", "Bedrooms", "Age"]]

# Target
y = df["Price"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = LinearRegression()

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Error
mae = mean_absolute_error(y_test, predictions)

print("Mean Absolute Error:", mae)

# New House Prediction
house = [[2100, 4, 3]]

predicted_price = model.predict(house)

print("Predicted Price:", predicted_price[0])

day18_spam_email_detector.py
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Dataset
emails = [
    "Win a free iPhone now",
    "Congratulations you won a lottery",
    "Claim your prize today",
    "Meeting at 4 PM",
    "Project submission tomorrow",
    "Let's discuss machine learning"
]

labels = [
    "Spam",
    "Spam",
    "Spam",
    "Not Spam",
    "Not Spam",
    "Not Spam"
]

# Convert text into numbers
vectorizer = CountVectorizer()

X = vectorizer.fit_transform(emails)

# Train model
model = MultinomialNB()

model.fit(X, labels)

# Test email
new_email = ["Free lottery prize waiting for you"]

new_email_vector = vectorizer.transform(new_email)

prediction = model.predict(new_email_vector)

print("Prediction:", prediction[0])
