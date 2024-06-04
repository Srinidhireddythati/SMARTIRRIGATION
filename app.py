import streamlit as st
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the data
url = 'Irrigation Scheduling.csv'  # Update the path if necessary
df = pd.read_csv(url)

# Streamlit app title
st.title("Smart Irrigation System")

# Display the dataframe
st.write("### Data Preview")
st.write(df.head())

# Display first and last five records
st.write("### First Five Records")
st.write(df.head())

st.write("### Last Five Records")
st.write(df.tail())

# Display class
st.write("### Class Column")
st.write(df['class'])

# Describe the data
st.write("### Data Description")
st.write(df.describe())

# Display info
st.write("### Data Info")
buffer = io.StringIO()
df.info(buf=buffer)
s = buffer.getvalue()
st.text(s)

# Number of unique values for each column
st.write("### Number of Unique Values for Each Column")
st.write(df.nunique())


# Dropping unnecessary column 
df.drop("id", axis=1, inplace=True)

# Filling the empty values(NaN,NAN,na) of column Altitude with average of all values of same column
df['altitude'].fillna(int(df['altitude'].mean()), inplace=True)

# Now checking the number of empty values (NaN,NAN,na) in Altitude column
st.write("### Number of NaN Values in Altitude Column After Filling")
st.write(df['altitude'].isna().sum())

# Visualize class distribution
st.write("### Class Distribution")
fig, ax = plt.subplots()
sns.countplot(x='class', data=df, ax=ax)
st.pyplot(fig)

# Encoding categorical data
st.write("### Encoding Categorical Data")
onehot_encoder = OneHotEncoder()
class_encoded = onehot_encoder.fit_transform(df[['class']]).toarray()

# Add encoded class to dataframe and remove original 'class' column
df_encoded = pd.concat([df.drop('class', axis=1), pd.DataFrame(class_encoded, columns=onehot_encoder.categories_[0])], axis=1)

# Correlation heatmap
st.write("### Correlation Heatmap")
# Exclude non-numeric columns from correlation calculation
numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
data = df[numeric_columns].corr()
fig, ax = plt.subplots()
sns.heatmap(data, annot=True, fmt='.0%', ax=ax)
st.pyplot(fig)

# Splitting data into X and Y
X = df.iloc[:, 0:4].values  # Independent dataset 
Y = df['class'].values  # Dependent dataset 

# Splitting dataset into training and testing sets
st.write("### Splitting Data into Training and Testing Sets")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

# Feature scaling
st.write("### Feature Scaling")
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train Logistic Regression model
st.write("### Training Logistic Regression Model")
model1 = LogisticRegression()
model1.fit(X_train, Y_train)

# Train Gaussian Naive Bayes model
st.write("### Training Gaussian Naive Bayes Model")
model2 = GaussianNB()
model2.fit(X_train, Y_train)

# Train SVM model
st.write("### Training SVM Model")
model3 = SVC(kernel='linear')
model3.fit(X_train, Y_train)

# Predictions
st.write("### Model Predictions")
pred1 = model1.predict(X_test)
pred2 = model2.predict(X_test)
pred3 = model3.predict(X_test)

# Model accuracy
st.write("### Model Accuracy")
st.write("Logistic Regression Accuracy: ", accuracy_score(Y_test, pred1))
st.write("Gaussian Naive Bayes Accuracy: ", accuracy_score(Y_test, pred2))
st.write("SVM Accuracy: ", accuracy_score(Y_test, pred3))

# Classification reports
st.write("### Classification Reports")
st.write("#### Logistic Regression")
st.text(classification_report(Y_test, pred1))

st.write("#### Gaussian Naive Bayes")
st.text(classification_report(Y_test, pred2))

st.write("#### SVM")
st.text(classification_report(Y_test, pred3))

# Streamlit app for predictions
st.write("### Predict Class Based on Environmental Data")

# Input fields
temperature = st.number_input('Temperature (°C)', value=25)
humidity = st.number_input('Humidity (%)', value=60)
ph = st.number_input('pH Level', value=7)
rainfall = st.number_input('Rainfall (mm)', value=100)

# Prediction
if st.button('Predict Class'):
    input_data = np.array([[temperature, humidity, ph, rainfall]])
    input_data = sc.transform(input_data)

    pred1 = model1.predict(input_data)
    pred2 = model2.predict(input_data)
    pred3 = model3.predict(input_data)

    st.write(f"Logistic Regression Prediction: {pred1[0]}")
    st.write(f"Gaussian Naive Bayes Prediction: {pred2[0]}")
    st.write(f"SVM Prediction: {pred3[0]}")