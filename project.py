#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
df = pd.read_csv('Fraud.csv')
df.head()
df.shape

#### 1. Data cleaning (missing values, outliers and multi-collinearity)
#%%
#checking missing values
df.isnull().sum()
#%%
df.columns
#%%
df[['oldbalanceOrg', 'newbalanceOrig']]
#%%
sns.set_style('darkgrid')
#%%
plt.figure()
sns.displot(df['step'], kde=True);
plt.show()
#%%
# distribution of type transaction
df.type.value_counts() / df.shape[0]
#%%
#checking multi-collinearity
plt.figure(figsize = (10, 6))
sns.heatmap(df.corr(), cmap='cool_r', linewidth=0.4, annot=True);
plt.show()

#We see on graph, that 'oldabalanceOrg' have correlation equal to 1 with 'newbalanceOrig' columns
#Also, we see that 'oldbalanceDest' with 'newbalanceDest' have same story too.
#So we can drop on 1 column for each pair
#%%
print("Total Unique Values in nameOrig", df['nameOrig'].nunique())
#%%
print("Total Unique Values in nameDest", df['nameDest'].nunique())

#All shape of data approx ~ 6.3 mln and we see that total unique names from nameOrig almost same. Also we see that nameDest has same situation.
#Its up to you delete or not.
#%%
df.nameOrig
#%%
df.nameDest
#%%
df = df.drop(columns = ['oldbalanceOrg', 'newbalanceDest']) #, 'nameOrig', 'nameDest'])
### you can think about drop columns 'nameOrig', 'nameDest'
df.shape
#%%
df.amount
#%%
plt.boxplot(df['amount']);
plt.show()

#There are a lot of outliers in this feature but we cannot remove much of the data because many of the outliers are associated
#with fraudulent transactions which are already a minority for predictive modelling
#%%
df = df.drop_duplicates()
#%%
df.shape
#%%
df.isFraud.value_counts() / df.shape[0]
#Imbalanced classification
#I suggest use downsampling of dataset. Its means to use not full dataset, because its hard to compute for model.
#You can use like 10% or more. But you need dont reduce distribution of fraud transactions


### Selection of variables to be included in the model.
#%%
# Choose numerical data
X = df.drop(['type', 'nameOrig', 'nameDest', 'isFraud', 'isFlaggedFraud'], axis=1).values
y = df['isFraud'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100, test_size=0.3)
print(X_train.shape)

#%%
#Remove features with low variance - zero variance is chosen on default

sel_variance_threshold = VarianceThreshold()
X_train_remove_variance = sel_variance_threshold.fit_transform(X_train)
print(X_train_remove_variance.shape)

'''
if the shape doesn't change, then all the features have variance higher than threshold (zero in our case)
'''

#%%
#Univariate feature selection - checking if there is a statistically significant relationship between feature and target
'''Chi-Square Test'''

sel_chi2 = SelectKBest(chi2, k=4)    # select 4 features (as we have 4 features left)
X_train_chi2 = sel_chi2.fit_transform(X_train, y_train)
print(sel_chi2.get_support())

'''f test'''
sel_f = SelectKBest(f_classif, k=4)
X_train_f = sel_f.fit_transform(X_train, y_train)
print(sel_f.get_support())

'''mutual_info_classif test'''
sel_mutual = SelectKBest(mutual_info_classif, k=4)
X_train_mutual = sel_mutual.fit_transform(X_train, y_train)
print(sel_mutual.get_support())

'''
if we get True for all the features, then all of them are statistically significant
in our case all the features are numerically significant and no of the columns with the numeric data can be deleted
'''

'''
Elaboration of the model

The main problems of the dataset are Outliers and Imbalance.
Possible solutions:
- work with outliers (algorithm of choice)> balancing (let it be undersampling in our case)> applying any model

or

- work with outliers (algorithm of choice) > use classifiers which need no balancing
'''

#%%
# Identifying Outliers with Interquartile Range (IQR)

outliers = pd.DataFrame()

col = 'amount'

Q1 = df[col].quantile(q=0.25)
Q3 = df[col].quantile(q=0.75)
IQR = Q3-Q1

# rows in dataframe that have values outside 1.5*IQR of Q1 and Q3
if not (df.loc[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]).empty:
    outliers = outliers.append(df.loc[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))])

#%%
###################
# OUTLIERS
###################

'''
It is impossible to delete outliers as fraudulent transactions associated with outliers are 47% 
from the total number of fraudulent transactions
'''

outliers.query('isFraud ==1').shape[0] / df.query('isFraud ==1').shape[0]

#%%
# Replacing Outliers with Median Values
'''
In this technique, we replace the extreme values with median values. 
It is advised to not use mean values as they are affected by outliers. 
The first line of code below prints the 50th percentile value, or the median, which comes out to be 140. 
The second line prints the 95th percentile value, which comes out to be around 326. The third line of code below 
replaces all those values in the 'Loan_amount' variable, which are greater than the 95th percentile, 
with the median value. 
Finally, the fourth line prints summary statistics after all these techniques have been employed for outlier treatment.


print(df['Loan_amount'].quantile(0.50))
print(df['Loan_amount'].quantile(0.95))
df['Loan_amount'] = np.where(df['Loan_amount'] > 325, 140, df['Loan_amount'])
df.describe()
'''

#%%
#Log Transformation
'''

Transformation of the skewed variables may also help correct the distribution of the variables. 
These could be logarithmic, square root, or square transformations. 
The most common is the logarithmic transformation, which is done on the 'Loan_amount' variable 
in the first line of code below. 
The second and third lines of code print the skewness value before and after the transformation.

Skewness value explains the extent to which the data is normally distributed. 
Ideally, the skewness value should be between -1 and +1, and any major deviation from this range 
indicates the presence of extreme values.

df["Log_Loanamt"] = df["Loan_amount"].map(lambda i: np.log(i) if i > 0 else 0) 
print(df['Loan_amount'].skew())
print(df['Log_Loanamt'].skew())
'''

#%%
#RobustScaler & QuantileTransformer
'''

https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer

X_scaled = RobustScaler(quantile_range=(25, 75)).fit_transform(X)
# Data after quantile transformation (uniform pdf)
X_scaled = QuantileTransformer(output_distribution="uniform").fit_transform(X),
#Data after quantile transformation (gaussian pdf)",
X_scaled = QuantileTransformer(output_distribution="normal").fit_transform(X),

'''

#%%
####################
# IMBALANCE DATA
###################
'''
https://www.analyticsvidhya.com/blog/2021/06/5-techniques-to-handle-imbalanced-data-for-a-classification-problem/
'''

df.isFraud.value_counts() / df.shape[0]

# fraudulent transactions are less than 1% from the total number of transactions (0,13 %)

#%%
#Resampling (Oversampling and Undersampling)
'''
Sklearn.utils resample can be used for both undersamplings the majority class and oversample minority class instances.

from sklearn.utils import resample
#create two different dataframe of majority and minority class 
df_majority = df_train[(df_train['Is_Lead']==0)] 
df_minority = df_train[(df_train['Is_Lead']==1)] 
# upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,    # sample with replacement
                                 n_samples= 131177, # to match majority class
                                 random_state=42)  # reproducible results
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_minority_upsampled, df_majority])

'''
# Solution with downsampling and
#Resampling (Oversampling and Undersampling)
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, test_size=0.2)

from sklearn.utils import resample
#create two different dataframe of majority and minority class
df_majority = df_train[(df_train['isFraud'] == 0)]
df_minority = df_train[(df_train['isFraud'] == 1)]

# upsample minority class
df_majority_downsampled = resample(df_majority,
                                 replace=True,    # sample with replacement
                                 n_samples=len(df_minority), # to match majority class
                                 random_state=42)  # reproducible results
# Combine majority class with upsampled minority class
df_downsampled = pd.concat([df_majority_downsampled, df_minority])

print(df_downsampled)
#%%
# SMOTE (Synthetic Minority Oversampling Technique)

'''
Simply adding duplicate records of minority class often don’t add any new information to the model. 
In SMOTE new instances are synthesized from the existing data. 
If we explain it in simple words, SMOTE looks into minority class instances and use k nearest neighbor
to select a random nearest neighbor, and a synthetic instance is created randomly in feature space.

from imblearn.over_sampling import SMOTE
# Resampling the minority class. The strategy can be changed as required.
sm = SMOTE(sampling_strategy='minority', random_state=42)
# Fit the model to generate the data.
oversampled_X, oversampled_Y = sm.fit_sample(df_train.drop('Is_Lead', axis=1), df_train['Is_Lead'])
oversampled = pd.concat([pd.DataFrame(oversampled_Y), pd.DataFrame(oversampled_X)], axis=1)
'''


#%%
#Threshold moving
'''
In the case of our classifiers, many times classifiers actually predict the probability of class membership. 
We assign those prediction’s probabilities to a certain class based on a threshold which is usually 0.5, 
i.e. if the probabilities < 0.5 it belongs to a certain class, and if not it belongs to the other class.

For imbalanced class problems, this default threshold may not work properly. 
We need to change the threshold to the optimum value so that it can efficiently separate two classes.

from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier()
rf_model.fit(X_train,y_train)   
rf_model.predict_proba(X_test) #probability of the class label

After getting the probability we can check for the optimum value.
'''

# Choose data for the model
X = df.drop(['isFraud', 'isFlaggedFraud'], axis=1).values
y = df['isFraud'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100, test_size=0.3)
print(X_train.shape)

#%%
# BalancedBaggingClassifier
'''
A BalancedBaggingClassifier is the same as a sklearn classifier but with additional balancing. 
It includes an additional step to balance the training set at the time of fit for a given sampler. 
This classifier takes two special parameters “sampling_strategy” and “replacement”. 
The sampling_strategy decides the type of resampling required (e.g. ‘majority’ – resample only the majority class, 
‘all’ – resample all classes, etc) and replacement decides whether it is going to be a sample with replacement or not.
'''
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier
#Create an instance
classifier = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                sampling_strategy='not majority',
                                replacement=False,
                                random_state=42)
classifier.fit(X_train, y_train)
preds = classifier.predict(X_test)


#%%
'''
Checking how well the model performs
'''
cm = confusion_matrix(y_test, preds)
print("Confusion Matrix: \n", cm)
seaborn.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False)
matplotlib.pyplot.show()
total1 = sum(sum(cm))
accuracy1 = (cm[0, 0] + cm[1, 1]) / total1
print("\nCalculated from Confusion Matrix: ")
print('Accuracy : ', accuracy1)

sensitivity1 = cm[0, 0] / (cm[0, 0] + cm[0, 1])
print('Sensitivity : ', sensitivity1)

specificity1 = cm[1, 1] / (cm[1, 0] + cm[1, 1])
print('Specificity : ', specificity1)

precision1 = cm[0, 0] / (cm[0, 0] + cm[1, 0])
print('Precision : ', precision1)
#%%
print("\nCalculated with functions: \n")
print("accuracy: ", accuracy_score(y_test, preds))
print("Balanced accuracy: ", balanced_accuracy_score(y_test, preds))
print("precision (TP/(TP+FP)): ", precision_score(y_test, preds))
print("Recall (sensitivity, TP/(TP+FN)): ", recall_score(y_test, preds))
print("F1 Score: ", precision_score(y_test, preds))

print("Classification report: \n", classification_report(y_test, preds))


