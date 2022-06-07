#%%
import pandas as pd
import matplotlib.pyplot as plt

#%%
df = pd.read_csv('Fraud.csv')
df.head()
#%%
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
import seaborn as sns
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








