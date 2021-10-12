# Importing the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier

# Load the DataFrame
csv_file = 'penguin.csv'
df = pd.read_csv(csv_file)

# Display the first five rows of the DataFrame
df.head()

# Drop the NAN values
df = df.dropna()

# Add numeric column 'label' to resemble non numeric column 'species'
df['label'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo':2})


# Convert the non-numeric column 'sex' to numeric in the DataFrame
df['sex'] = df['sex'].map({'Male':0,'Female':1})

# Convert the non-numeric column 'island' to numeric in the DataFrame
df['island'] = df['island'].map({'Biscoe': 0, 'Dream': 1, 'Torgersen':2})


# Create X and y variables
X = df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


# Build a SVC model using the 'sklearn' module.
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
svc_score = svc_model.score(X_train, y_train)

# Build a LogisticRegression model using the 'sklearn' module.
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_score = log_reg.score(X_train, y_train)

# Build a RandomForestClassifier model using the 'sklearn' module.
rf_clf = RandomForestClassifier(n_jobs = -1)
rf_clf.fit(X_train, y_train)
rf_clf_score = rf_clf.score(X_train, y_train)

def prediction(model,island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex):
	prediction = model(island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex)
	score = model.score(island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex)
	return prediction,score

st.sidebar.title('Penguin Race Prediction')
b_l_m = st.sidebar.slider('Bill Length(in mm)',float(df['bill_length_mm'].min()),float(df['bill_length_mm'].max()))
b_d_m = st.sidebar.slider('Bill Depth(in mm',float(df['bill_depth_mm'].min()),float(df['bill_depth_mm'].max()))
f_l_m = st.sidebar.slider('Flipper Length(in mm)',float(df['flipper_length_mm'].min()),float(df['flipper_length_mm'].max()))
b_m_g = st.sidebar.slider('Body Mass(in gram',float(df['body_mass_g'].min()),float(df['body_mass_g'].max()))
sex = st.sidebar.selectbox('Sex',['Male','Female'])
island = st.sidebar.selectbox('Island',['Torgersen','Biscoe','Dream'])
model = st.sidebar.selectbox('Select Model',[svc_model,log_reg,rf_clf])

if st.sidebar.button('Predict') == True:
	prediction,score = prediction(model,island,b_l_m,b_d_m,f_l_m,b_m_g,sex)
	st.write('Penguin race is ',prediction)
	st.write('Score is ',score)
