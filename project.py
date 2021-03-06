#%%
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import seaborn as sns
import imblearn
import sklearn
from matplotlib import pyplot
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, chi2, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, precision_score, recall_score, \
    classification_report, roc_auc_score, f1_score
import xgboost as xgb
from sklearn.preprocessing import RobustScaler, QuantileTransformer, OrdinalEncoder
import lightgbm as lgbm
import catboost as catb

#%%

df = pd.read_csv('Fraud.csv')
df.head()
df.shape

#### 1. Data cleaning (missing values, outliers and multi-collinearity)
#%%
'''
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
'''
#%%
df = df.drop(columns = ['oldbalanceOrg', 'newbalanceDest']) #, 'nameOrig', 'nameDest'])
### you can think about drop columns 'nameOrig', 'nameDest'
'''
#%%
df = df.drop(columns = ['oldbalanceOrg', 'newbalanceDest', 'nameOrig', 'nameDest'])
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

'''
### Selection of variables to be included in the model.
#%%
# Choose numerical data
#%% make dummies of payment type
df = pd.get_dummies(df, columns=['type'], drop_first=True)
#%%
#drop categorical data
X = df.drop(['nameOrig', 'nameDest', 'isFraud', 'isFlaggedFraud'], axis=1).values

#%%
y = df['isFraud'].values

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=100, test_size=0.3)
print(X_train.shape)

'''
#%%
#Remove features with low variance - zero variance is chosen on default

sel_variance_threshold = VarianceThreshold()
X_train_remove_variance = sel_variance_threshold.fit_transform(X_train)
print(X_train_remove_variance.shape)

'''
'''
if the shape doesn't change, then all the features have variance higher than threshold (zero in our case)
'''
'''
#%%
#Univariate feature selection - checking if there is a statistically significant relationship between feature and target
''''''Chi-Square Test''''''

sel_chi2 = SelectKBest(chi2, k=8)    # select 8 features (as we have 5 features left)
X_train_chi2 = sel_chi2.fit_transform(X_train, y_train)
print(sel_chi2.get_support())
#%%
''''''f test''''''
sel_f = SelectKBest(f_classif, k=8)
X_train_f = sel_f.fit_transform(X_train, y_train)
print(sel_f.get_support())
#%%
''''''mutual_info_classif test''''''
sel_mutual = SelectKBest(mutual_info_classif, k=8)
X_train_mutual = sel_mutual.fit_transform(X_train, y_train)
print(sel_mutual.get_support())

'''
'''
if we get True for all the features, then all of them are statistically significant
in our case all the features are numerically significant and no of the columns with the numeric data
 and payment type can be deleted
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
'''
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
'''
It is impossible to delete outliers as fraudulent transactions associated with outliers are 47% 
from the total number of fraudulent transactions
'''
'''
outliers.query('isFraud ==1').shape[0] / df.query('isFraud ==1').shape[0]
'''
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
'''

print(df['amount'].quantile(0.50))
print(df['amount'].quantile(0.95))
df['amount'] = np.where(df['amount'] > (df['amount'].quantile(0.95) -1), df['amount'].quantile(0.50), df['amount'])
df.describe()

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
'''
'''
df["Log_amount"] = df["amount"].map(lambda i: np.log(i) if i > 0 else 0)
print(df['amount'].skew())
print(df['Log_amount'].skew())
'''


#RobustScaler & QuantileTransformer
'''

https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
'''
#%%
X = RobustScaler(quantile_range=(25, 75)).fit_transform(X)
# Data after quantile transformation (uniform pdf)
'''
#%%
X = QuantileTransformer(output_distribution="uniform").fit_transform(X),
#Data after quantile transformation (gaussian pdf)",
#%%
X = QuantileTransformer(output_distribution="normal").fit_transform(X),
'''

#%%
####################
# IMBALANCE DATA
###################
'''
https://www.analyticsvidhya.com/blog/2021/06/5-techniques-to-handle-imbalanced-data-for-a-classification-problem/
'''
'''
df.isFraud.value_counts() / df.shape[0]
'''
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
Simply adding duplicate records of minority class often don???t add any new information to the model. 
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
We assign those prediction???s probabilities to a certain class based on a threshold which is usually 0.5, 
i.e. if the probabilities < 0.5 it belongs to a certain class, and if not it belongs to the other class.

For imbalanced class problems, this default threshold may not work properly. 
We need to change the threshold to the optimum value so that it can efficiently separate two classes.
'''
'''
df['nameDest_startswith_C'] = df['nameDest'].str.startswith('C')

df[['nameDest_startswith_C']] = OrdinalEncoder(categories=[[False, True]]).fit_transform(df[['nameDest_startswith_C']])
#%%
df = pd.get_dummies(df, columns=['type'], drop_first=True)
#%%
df.drop(['nameOrig', 'nameDest'], axis=1, inplace=True)

df.head()

#%%

y = df['isFraud']
X = df.loc[:, df.columns != 'isFraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

rf_model = LogisticRegression()
rf_model.fit(X_train, y_train)
predicted_proba = rf_model.predict_proba(X_test)

#%%

step_factor = 0.001
threshold_value = 0.001
roc_score = 0
while threshold_value <= 0.4:  # continue to check best threshold upto probability 0.8
  temp_thresh = threshold_value
  predicted = (predicted_proba[:, 1] >= temp_thresh).astype('int')  # change the class boundary for prediction
  #print('Threshold', temp_thresh, '--', roc_auc_score(y_test, predicted))
  if roc_score < roc_auc_score(y_test, predicted):  # store the threshold for best classification
    roc_score = roc_auc_score(y_test, predicted)
    thrsh_score = threshold_value
  threshold_value = threshold_value + step_factor
print('---Optimum Threshold ---', thrsh_score, '--ROC--', roc_score)

preds = [1 if x >= thrsh_score else 0 for x in predicted_proba[:, 1]]

#%%
# Choose data for the model
df = df_downsampled
#%%
X = df.drop(['type', 'nameOrig', 'nameDest', 'isFraud', 'isFlaggedFraud'], axis=1).values
#%%
X = df.drop(['nameOrig', 'nameDest', 'isFraud', 'isFlaggedFraud'], axis=1).values
#%%
y = df['isFraud'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100, test_size=0.3)
print(X_train.shape)
'''
#%%
# BalancedBaggingClassifier
'''
A BalancedBaggingClassifier is the same as a sklearn classifier but with additional balancing. 
It includes an additional step to balance the training set at the time of fit for a given sampler. 
This classifier takes two special parameters ???sampling_strategy??? and ???replacement???. 
The sampling_strategy decides the type of resampling required (e.g. ???majority??? ??? resample only the majority class, 
???all??? ??? resample all classes, etc) and replacement decides whether it is going to be a sample with replacement or not.
'''
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier
#Create an instance
classifier = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                sampling_strategy='not majority',
                                replacement=False,
                                random_state=42)
#classifier = BaggingClassifier(DecisionTreeClassifier(), random_state=42)
classifier.fit(X_train, y_train)
preds = classifier.predict(X_test)

'''

Performance after downsampling:
Confusion Matrix: 
 [[1740  194]
 [ 311 1716]]
Calculated from Confusion Matrix: 
Accuracy :  0.8725069426912396
Sensitivity :  0.8996897621509824
Specificity :  0.8465712876171683
Precision :  0.8483666504144319

+ transaction type

Confusion Matrix: 
 [[1807  113]
 [ 189 1812]]
Accuracy:  0.9229788319306299
Balanced accuracy:  0.92334652986007
Precision (TP/(TP+FP)):  0.9412987012987013
Recall (sensitivity, TP/(TP+FN)):  0.9055472263868066
Specificity (TN/(TN+FP)) 0.9411458333333333
F1 Score:  0.9230769230769231

Performance without downsampling:
Confusion Matrix: 
 [[1906079     229]
 [   1353    1125]]
Calculated from Confusion Matrix: 
Accuracy :  0.9991712009622871
Sensitivity :  0.9998798725074857
Specificity :  0.4539951573849879
Precision :  0.9992906693397196


Performance with Robust Scaler and downsampling:
Confusion Matrix: 
 [[1692  225]
 [ 269 1738]]
Calculated from Confusion Matrix: 
Accuracy :  0.8741080530071356
Sensitivity :  0.8826291079812206
Specificity :  0.8659691081215745
Precision :  0.8628250892401835

+ transaction type
Confusion Matrix: 
 [[1831  120]
 [ 211 1805]]
Accuracy:  0.9165616334761785
Balanced accuracy:  0.9169151910294273
Precision (TP/(TP+FP)):  0.9376623376623376
Recall (sensitivity, TP/(TP+FN)):  0.8953373015873016
Specificity (TN/(TN+FP)) 0.938493080471553
F1 Score:  0.9160111646790154

Performance with Robust Scaler without downsampling:
Confusion Matrix: 
 [[1906066     242]
 [   1358    1120]]
Calculated from Confusion Matrix: 
Accuracy :  0.9991617708847403
Sensitivity :  0.9998730530428451
Specificity :  0.4519774011299435
Precision :  0.9992880450282685

+ transaction type
Confusion Matrix: 
 [[1906055     253]
 [   1267    1211]]

Accuracy:  0.9992036823405034
Balanced accuracy:  0.7442839238491812
Precision (TP/(TP+FP)):  0.8271857923497268
Recall (sensitivity, TP/(TP+FN)):  0.4887005649717514
Specificity (TN/(TN+FP)) 0.9998672827266109
F1 Score:  0.6144089294774226

Performance with Outliers Replaced with Median Values with downsampling:
Confusion Matrix: 
 [[1759  185]
 [ 282 1745]]
Calculated from Confusion Matrix: 
Accuracy :  0.8823973810123394
Sensitivity :  0.904835390946502
Specificity :  0.8608781450419339
Precision :  0.8618324350808427

+ transaction type
Confusion Matrix: 
 [[1806  132]
 [ 191 1806]]
Accuracy:  0.9179161372299873
Balanced accuracy:  0.9181225398469222
Precision (TP/(TP+FP)):  0.9318885448916409
Recall (sensitivity, TP/(TP+FN)):  0.9043565348022033
Specificity (TN/(TN+FP)) 0.9318885448916409
F1 Score:  0.9179161372299873
 

Performance with Outliers Replaced with Median Values without downsampling:
Confusion Matrix: 
 [[1906095     213]
 [   1396    1082]]
Calculated from Confusion Matrix: 
Accuracy :  0.999157055845967
Sensitivity :  0.9998882656947355
Specificity :  0.43664245359160614
Precision :  0.9992681485784205

+ transaction type
Confusion Matrix: 
 [[1906026     282]
 [   1393    1085]]
Accuracy:  0.9991224788949625
Balanced accuracy:  0.7188525887096766
Precision (TP/(TP+FP)):  0.7937088514996342
Recall (sensitivity, TP/(TP+FN)):  0.4378531073446328
Specificity (TN/(TN+FP)) 0.9998520700747203
F1 Score:  0.5643693107932379

with transactions type
Confusion Matrix: 
 [[1906064     244]
 [   1263    1215]]
Calculated with functions: 
Accuracy:  0.9992104929520648
Balanced accuracy:  0.7450933869351128
Precision (TP/(TP+FP)):  0.8327621658670322
Recall (sensitivity, TP/(TP+FN)):  0.49031476997578693
Specificity (TN/(TN+FP)) 0.9998720038944389
F1 Score:  0.6172212344424689

'''
'''
Alternative model XGBoost - doesn't need feature scaling and not influenced by outliers
'''
#%%
'''
classifier = xgb.XGBClassifier(max_depth=4)
'''

'''
Performance without downsampling
Confusion Matrix: 
 [[1906237      71]
 [   1963     515]]
Calculated from Confusion Matrix: 
Accuracy :  0.9989344012372262
Sensitivity :  0.9999627552315785
Specificity :  0.20782889426957224
Precision :  0.9989712818362855


Performance with downsampling:
Confusion Matrix: 
 [[1767  171]
 [ 206 1805]]
Calculated from Confusion Matrix: 
Accuracy :  0.9045327931121803
Sensitivity :  0.9117647058823529
Specificity :  0.8975634012928891
Precision :  0.895590471363406


with payment data
Confusion Matrix: 
 [[1906162     146]
 [   1546     932]]
 Accuracy:  0.999113572710613
Balanced accuracy:  0.6880165890533095
Precision (TP/(TP+FP)):  0.8645640074211502
Recall (sensitivity, TP/(TP+FN)):  0.3761097659402744
Specificity (TN/(TN+FP)) 0.9999234121663446
F1 Score:  0.5241844769403824
 
'''
#%%
'''
LightGBM is a gradient boosting framework based on decision trees to increases the efficiency of 
the model and reduces memory usage. It uses two novel techniques: Gradient-based One Side Sampling 
and Exclusive Feature Bundling (EFB)
'''
'''
classifier = lgbm.LGBMClassifier()
'''
'''
Performance evaluation
Without downsampling:
Confusion Matrix: 
 [[1904492    1816]
 [   2120     358]]
Calculated from Confusion Matrix: 
Accuracy :  0.9979379563764613
Sensitivity :  0.9990473732471353
Specificity :  0.14447134786117838
Precision :  0.9988880800078883

F1 Score:  0.16467341306347746

With downsampling:
Confusion Matrix: 
 [[1766  179]
 [ 239 1761]]
Calculated from Confusion Matrix: 
Accuracy :  0.89404309252218
Sensitivity :  0.9079691516709512
Specificity :  0.8805
Precision :  0.8807980049875311

With payment type
Confusion Matrix: 
 [[1905362     946]
 [   1863     615]]
Calculated with functions: 
Accuracy:  0.9985283840095223
Balanced accuracy:  0.6238438860871546
Precision (TP/(TP+FP)):  0.3939782190903267
Recall (sensitivity, TP/(TP+FN)):  0.24818401937046006
Specificity (TN/(TN+FP)) 0.9995037528038491
F1 Score:  0.30453082446150037
'''

#%%
'''
CatBoost - Catboost Model is a powerful, scalable, and robust machine learning model that enables us to 
have escalated performance based on the gradient boosting system and the decision trees altogether. 
Moreover, it is available both for categorical and continuous data values. 
'''
'''
classifier = catb.CatBoostClassifier(verbose=0)
'''
'''
Performance evaluation
Without downsampling:
Confusion Matrix: 
 [[1906167     141]
 [   1887     591]]
Calculated from Confusion Matrix: 
Accuracy :  0.9989375445964084
Sensitivity :  0.9999260350373602
Specificity :  0.23849878934624696
Precision :  0.999011034278904

F1 Score:  0.8073770491803278

With downsampling:
Confusion Matrix: 
 [[1725  196]
 [ 227 1788]]
Calculated from Confusion Matrix: 
Accuracy :  0.8925304878048781
Sensitivity :  0.8979698073919833
Specificity :  0.8873449131513648
Precision :  0.8837090163934426

F1 Score:  0.9012096774193549



With transactions type
Confusion Matrix: 
 [[1906061     247]
 [   1492     986]]
Calculated with functions: 
Accuracy:  0.9990889497303521
Balanced accuracy:  0.6988859818332916
Precision (TP/(TP+FP)):  0.7996755879967559
Recall (sensitivity, TP/(TP+FN)):  0.3979015334947538
Specificity (TN/(TN+FP)) 0.9998704301718295
F1 Score:  0.5313931554836971
'''

#%%
'''
classifier.fit(X_train, y_train)
preds = classifier.predict(X_test)
'''
#%%
'''
Checking how well the model performs
'''
cm = confusion_matrix(y_test, preds)
print("Confusion Matrix: \n", cm)
seaborn.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False)
matplotlib.pyplot.show()
#%%
'''
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
'''

#%%

#For an imbalanced class dataset F1 score is a more appropriate metric.
print("\nCalculated with functions: \n")
print("Accuracy: ", accuracy_score(y_test, preds))
print("Balanced accuracy: ", balanced_accuracy_score(y_test, preds))
print("Precision (TP/(TP+FP)): ", precision_score(y_test, preds))
print("Recall (sensitivity, TP/(TP+FN)): ", recall_score(y_test, preds))
print("Specificity (TN/(TN+FP))", recall_score(y_test, preds, pos_label=0))
print("F1 Score: ", f1_score(y_test, preds))

print("Classification report: \n", classification_report(y_test, preds))

#%%
# Feature importance (doesn't work for LogReg and BaggingClassifier)
#Plotting the feature importance for Top 10 most important columns
# get importance
'''
importance = classifier.feature_importances_
# summarize feature importance
for i, v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i, v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()
'''
'''
For CatBoost

Feature: 0, Score: 12.42587
Feature: 1, Score: 10.49467
Feature: 2, Score: 62.96744
Feature: 3, Score: 14.11203

with downsampling
Feature: 0, Score: 29.96031
Feature: 1, Score: 20.50994
Feature: 2, Score: 39.85208
Feature: 3, Score: 9.67768

LGBM Feature Importance

Feature: 0, Score: 971.00000
Feature: 1, Score: 799.00000
Feature: 2, Score: 314.00000
Feature: 3, Score: 701.00000
Feature: 4, Score: 67.00000
Feature: 5, Score: 32.00000
Feature: 6, Score: 42.00000
Feature: 7, Score: 74.00000

XGB Feature Importance
Feature: 0, Score: 0.05156
Feature: 1, Score: 0.03314
Feature: 2, Score: 0.18451
Feature: 3, Score: 0.16597
Feature: 4, Score: 0.15608
Feature: 5, Score: 0.00170
Feature: 6, Score: 0.10157
Feature: 7, Score: 0.30546

CatBoost feature importance
Feature: 0, Score: 8.76754
Feature: 1, Score: 9.07470
Feature: 2, Score: 40.09675
Feature: 3, Score: 24.66942
Feature: 4, Score: 8.67844
Feature: 5, Score: 0.00014
Feature: 6, Score: 1.25331
Feature: 7, Score: 7.45969

'''
#%%
# For Log Reg
'''
import math

w = lr_model.coef_[0]
features = X.columns[1:]
feature_importance = pd.DataFrame(X.columns, columns=["feature"])
feature_importance["importance"] = pow(math.e, w)
feature_importance = feature_importance.sort_values(by=["importance"], ascending=False)

from sklearn.linear_model import LogisticRegression

ax = feature_importance.plot.barh(x='feature', y='importance')
plt.savefig('feature_importance_lr.png', bbox_inches='tight')
plt.show()

#%%
# For bagging classifier
classifier.fit(X,y)
importance = np.mean([
    tree.feature_importances_ for tree in classifier.estimators_
], axis=0)
#%%
feature_names = X_train.feature_names
feature_importances = np.mean([tree.feature_importances_ for tree in classifier.estimators_], axis=0)

output = {fn:fi for fn,fi in zip(feature_names, feature_importances)}
print(output)
#%%
# summarize feature importance
for i, v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i, v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()

#%%
importances = classifier.feature_importances_
'''