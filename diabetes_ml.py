#%%
#Import dependencies
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# %%
#Data Collection & Analysis
diabetes_dataset = pd.read_csv('diabetes.csv')
#printing file
diabetes_dataset.head()

# %%
#printing shape
diabetes_dataset.shape

#getting the statistical measures
diabetes_dataset.describe()

#counter of people
#0 --> Non-Diabetic
#1 --> Diabetic
diabetes_dataset['Outcome'].value_counts()
# %%
diabetes_dataset.groupby('Outcome').median()
# %%
#separating the data and labels
x = diabetes_dataset.drop(columns='Outcome', axis=1)
y = diabetes_dataset['Outcome']
print(x)
# %%
#data standarization
scaler  = StandardScaler()
scaler.fit(x)
standardized_data = scaler.transform(x)
X = standardized_data
Y = diabetes_dataset['Outcome'] 
# %%
#Train test Split
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size= 0.2, stratify=Y, random_state=2)
# %%
#Training the model
classifier = svm.SVC(kernel='linear')
#Training the support vector machine Classifier
classifier.fit(x_train, y_train)
# %%
#Model Evaluation 
#accuracy score on the training data
x_train_prediction = classifier.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
print('Accuracy score of the training data : ' ,training_data_accuracy)
# %%
#accuracy score on the training data
x_test_prediction = classifier.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)
print('Accuracy score of the test data : ' ,test_data_accuracy)

# %%
#Making a predictive system
input_data = (8,183,64,0,0,23.3,0.672,32)

#changing the input_data to numpyarray
input_data_as_numpy_array = np.asarray(input_data)

#reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

#standardize de input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)
# %%
