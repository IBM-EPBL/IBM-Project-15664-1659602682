import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import  warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

data=pd.read_csv('./Data/CrudeOilPricesDaily.csv')
print(data)
print(data.head())

data['year'] = pd.DatetimeIndex(data['Date']).year
data['month'] = pd.DatetimeIndex(data['Date']).month
data['day'] = pd.DatetimeIndex(data['Date']).day
data.drop('Date',axis=1,inplace=True)

data['Price'].fillna(data['Price'].median(),inplace=True)

X=data.drop('Price',axis=1)
Y=data['Price']


from sklearn.model_selection import train_test_split


X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
X_train.shape,X_test.shape,y_train.shape,y_test.shape




from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor (n_estimators =1000, max_depth = 10, random_state = 34)


regressor.fit (X_train, np.ravel(y_train, order = 'C'))




# Creating a pickle file for the classifier
filename = 'Model/prediction-rfc-model.pkl'
pickle.dump(regressor, open(filename, 'wb'))




filename = 'Model/prediction-rfc-model.pkl'
pickle.dump(regressor, open(filename, 'wb'))

filename = 'Model/prediction-rfc-model.pkl'
classifier = pickle.load(open(filename, 'rb'))


data = np.array([[1986,3,17]])
my_prediction = classifier.predict(data)
warnings.filterwarnings("ignore", category=DeprecationWarning)
print(my_prediction[0])