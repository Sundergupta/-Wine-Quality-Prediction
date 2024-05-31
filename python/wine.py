import numpy as np # import numpy as np: Import the NumPy library for numerical operations.
import pandas as pd #import pandas as pd: Import the pandas library for data manipulation.
import matplotlib.pyplot as plt#import matplotlib.pyplot as plt: Import Matplotlib for plotting.
import seaborn as sb#import seaborn as sb: Import Seaborn for advanced statistical plotting.

from sklearn.model_selection import train_test_split#from sklearn.model_selection import train_test_split: Import the function to split data into training and testing sets.
from sklearn.preprocessing import MinMaxScaler#from sklearn.preprocessing import MinMaxScaler: Import the MinMaxScaler for feature scaling.
from sklearn import metrics#from sklearn import metrics: Import metrics for model evaluation.
from sklearn.svm import SVC#from sklearn.svm import SVC: Import the Support Vector Classifier.
from xgboost import XGBClassifier#from xgboost import XGBClassifier: Import the XGBoost Classifier.
from sklearn.linear_model import LogisticRegression#from sklearn.linear_model import LogisticRegression: Import Logistic Regression model

import warnings#import warnings: Import the warnings module to manage warnings.
warnings.filterwarnings('ignore')#warnings.filterwarnings('ignore'): Ignore warnings to keep the output clean.


df = pd.read_csv('winequalityN.csv')#df = pd.read_csv('winequalityN.csv'): Load the dataset from a CSV file into a DataFrame.
print(df.head())#print(df.head()): Display the first five rows of the DataFrame.

df.info()#df.info(): Print a concise summary of the DataFrame, including the data types and non-null values.


df.describe().T#df.describe().T: Generate descriptive statistics of the DataFrame and transpose it for better readability.

#
df.isnull().sum()#df.isnull().sum(): Count the number of missing values in each column.


for col in df.columns:#Iterate through each column to check for missing values.
    if df[col].isnull().sum() > 0:#If a column has missing values, fill them with the column's mean value.
	    df[col] = df[col].fillna(df[col].mean())#

df.isnull().sum().sum()#df.isnull().sum().sum(): Verify that all missing values have been filled.

df.hist(bins=20, figsize=(10, 10))#df.hist(bins=20, figsize=(10, 10)): Plot histograms for each feature.
plt.show()#


plt.bar(df['quality'], df['alcohol'])#plt.bar(df['quality'], df['alcohol']): Create a bar plot of alcohol content against wine quality.
plt.xlabel('quality')#plt.xlabel('quality') and plt.ylabel('alcohol'): Label the axes of the bar plot.
plt.ylabel('alcohol')#
plt.show()#


plt.figure(figsize=(12, 12))#plt.figure(figsize=(12, 12)): Set the figure size for the heatmap.
sb.heatmap(df.corr() > 0.7, annot=True, cbar=False)#sb.heatmap(df.corr() > 0.7, annot=True, cbar=False): Plot a heatmap of features with a correlation greater than 0.7.
plt.show()#


df = df.drop('total sulfur dioxide', axis=1)#Drop the 'total sulfur dioxide' column.


df = df.drop('total sulfur dioxide', axis=1)#Replace 'white' with 1 and 'red' with 0 in the DataFrame.


df.replace({'white': 1, 'red': 0}, inplace=True)#


features = df.drop(['quality', 'best quality'], axis=1)#Separate features (independent variables) from the target (dependent variable).
target = df['best quality']#

xtrain, xtest, ytrain, ytest = train_test_split(#
	features, target, test_size=0.2, random_state=40)#Split the data into training and testing sets with a test size of 20%.

xtrain.shape, xtest.shape#


norm = MinMaxScaler()#Initialize the MinMaxScaler.
xtrain = norm.fit_transform(xtrain)#Fit the scaler on the training data and transform it.
xtest = norm.transform(xtest)#Transform the test data.
#
models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf')]#

for i in range(3):#
	models[i].fit(xtrain, ytrain)#Initialize three models: Logistic Regression, XGBoost Classifier, and Support Vector Classifier.

	print(f'{models[i]} : ')#
	print('Training Accuracy : ', metrics.roc_auc_score(ytrain, models[i].predict(xtrain)))#
	print('Validation Accuracy : ', metrics.roc_auc_score(#Iterate over the models to train them and print training and validation accuracy using the ROC AUC score.
		ytest, models[i].predict(xtest)))#
	print()#


metrics.plot_confusion_matrix(models[1], xtest, ytest)#
plt.show()#Plot the confusion matrix for the XGBoost model.

print(metrics.classification_report(ytest,#Print the classification report for the XGBoost model.
									models[1].predict(xtest)))#
