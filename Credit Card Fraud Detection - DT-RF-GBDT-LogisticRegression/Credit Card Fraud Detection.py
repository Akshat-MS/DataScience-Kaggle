#!/usr/bin/env python
# coding: utf-8

# ## Credit Card Fraud Detection
# In this project you will predict fraudulent credit card transactions with the help of Machine learning models. Please import the following libraries to get started

# In[1]:


get_ipython().system('pip install xgboost')


# ##### Import dependent packages

# In[2]:


#Importing packages
import numpy as np
import pandas as pd
import math
from collections import Counter

# Importing matplotlib and seaborn for graphs
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")

import scipy.stats as stats
from scipy.stats import skew
from scipy.stats import kurtosis

#Import packages for remove warnings
import warnings
warnings.filterwarnings('ignore')

# Importing packages for Rescaling
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.preprocessing import PowerTransformer

# Load the library for splitting the data
from sklearn.model_selection import train_test_split

# Import imbalace technique algorithims
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.over_sampling import BorderlineSMOTE

# Importing RFE and LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

#Importing Xgboost
import xgboost as xgb
from xgboost import XGBClassifier, plot_importance

# Importing RandomForestClassifier and DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.tree import DecisionTreeClassifier

# GridSearch, cross_val_score & KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold

#Importing K nearest neighbour
from sklearn.neighbors import KNeighborsClassifier

# Statsmodel
import statsmodels.api as sm

# Libraries for Model Evaluation
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, recall_score
import zipfile


# ##### Loading Data

# In[3]:


zf = zipfile.ZipFile('310_23498_bundle_archive.zip')
credit_card_data = pd.read_csv(zf.open('creditcard.csv'))
credit_card_data.head()


# In[4]:


credit_card_data_org = credit_card_data.copy()


# ##### Dividing Data - Fraud vs Genuine

# In[5]:


#Divided the data into two sets Fraud vs Genuine based upon TARGET
genuine = credit_card_data.loc[credit_card_data['Class'] != 1]
fraud = credit_card_data.loc[credit_card_data['Class'] == 1]

genuine.head()
fraud.head()


# ## Understadning Data

# In[6]:


credit_card_data.shape


# In[7]:


credit_card_data.columns.values


# In[8]:


credit_card_data.info()


# In[9]:


credit_card_data.describe()


# #### Inference
# 
# - All features are numerical variables and there are no null values.

# ### Data distribution - Imbalanced or Balanced

# In[10]:


fig, ax = plt.subplots()

labels = ['Genuine','Fraud']
explode=(0.1,0)

cc_fraud = credit_card_data["Class"].value_counts()

df = pd.DataFrame({'labels': cc_fraud.index,
                   'values': cc_fraud.values
                  })
ax.pie(cc_fraud.values, explode=explode, labels=labels,  
       colors=['g','r'], autopct='%1.4f%%', 
       shadow=True, startangle=-20,   
       pctdistance=1.3,labeldistance=1.6)
ax.axis('equal')
ax.set_title("Genuine vs Fraud")
ax.legend(frameon=False, bbox_to_anchor=(1.2,0.8))


# #### Inferences
# - Imbalanced class distribution.
# - The number of observations belonging to one class is significantly lower than those belonging to the other class.
# - In this situation, the predictive model developed using conventional machine learning algorithms could be biased and inaccurate.
# 
# Challenges
# - Standard classifier algorithms like Decision Tree and Logistic Regression have a bias towards classes which have number of instances. They tend to only predict the majority class data. The features of the minority class are treated as noise and are often ignored. Thus, there is a high probability of misclassification of the minority class as compared to the majority class.

# ## EDA - Exploratory data analysis
# 

# ### Closer look - Time & Amount features
# - Amount
# - Time
# 
# 

# #### Amount  & Time

# In[11]:


credit_card_data[['Time','Amount']].describe()


# #### Inferences
# 
# - On an average, credit card transaction is happening at **every 94813.86 seconds**.
# - Average transaction amount is 88.35 with a standard deviation of 250, with a minimum amount of 0.0 and the maximum amount 25,691.16. 

# ### Scatter Plot (using pair ploting)
# ##### Check the skewness of Time & Amount variable (using pair plot)

# In[12]:


sns.set_style("darkgrid", {"axes.facecolor": ".9"})
num_cols = credit_card_data[[ 'Amount','Time','Class']]
sns.pairplot(num_cols,hue="Class",palette=["b","r"],size=5)
plt.show()


# #### Inferences
# 
# Time
# - **Credit Card Transactions (Fraud vs Genuine) with Time (Graph 2,2)** - Fraudulent transactions have a **distribution more even** than valid transactions - are **equaly distributed in time**, including the low real transaction times, during night in Europe timezone
# - The time elapsed for a fraud activity seems to be at the end of the dataset. However, this **Time variable does not seem of signiificance** as there does not seem to be a connect between fraud and time elapsed. **We can have more closer look later.**
# 
# Amount
# - **Graph (1,2)**, it is clearly visible that there are frauds only on the transactions which have transaction amount approximately less than 3000. Transactions which have transaction amount approximately above 3000 have no fraud.
# - Most the transaction amount falls between 0 and about 3000 and we have some outliers for really big amount transactions and it may actually make sense to **drop those outliers in our analysis** if they are just a few points that are very extreme.
# - Most daily transactions are not extremely expensive, but it’s likely where most fraudulent transactions are occurring as well.
# 

# ### Detailed Data Distibution Analysis

# In[13]:


print("Maximum amount of Fraud transaction - ",credit_card_data[(credit_card_data['Class'] == 1)]['Amount'].max())
print("Maximum amount of Genuine transaction - ",credit_card_data[(credit_card_data['Class'] == 0)]['Amount'].max())


# In[14]:


print("Fraud Transaction distribution : \n",credit_card_data[(credit_card_data['Class'] == 1)]['Amount'].value_counts().head())
print("\nGenuine Transaction distribution : \n",credit_card_data[(credit_card_data['Class'] == 0)]['Amount'].value_counts().head())


# #### Inferences
# - Maximum fraud transaction amount was 2125.87 and lowest was just 0.00.
# - Genuine high value transaction are very less

# ### Amount

# In[15]:


plt.figure(figsize=(14,6))

plt.subplot(1,2,1)
sns.boxplot(x = 'Class', y = 'Amount', data = credit_card_data, palette=("g",'b'),showfliers=True)
plt.title('Amount Distribution for Fraud and Genuine transactions with Outliers',fontsize=12,family = "Comic Sans MS")
plt.xlabel('Class', fontsize=12,family = "Comic Sans MS")

plt.subplot(1,2,2)
sns.boxplot(x = 'Class', y = 'Amount', data = credit_card_data, palette=("b",'r'),showfliers=False)
plt.title('Amount Distribution for Fraud and Genuine transactions without Outliers',fontsize=12,family = "Comic Sans MS")
plt.xlabel('Class', fontsize=12,family = "Comic Sans MS")
plt.show()


# #### Inferences
# - We have some outliers for really big amount transactions and it may actually make sense to drop those outliers in our analysis if they are just a few points that are very extreme.
# - **We should be conscious about that these outliers should not be the fraudulent transaction**

# ### Time 

# In[16]:


# Converting time from second to hour
credit_card_data['Time_hr'] = credit_card_data['Time'].apply(lambda sec : (sec/3600))


# In[17]:


# Calculating hour of the day
credit_card_data['Hour'] = credit_card_data['Time_hr']%24   # 2 days of data
credit_card_data['Hour'] = credit_card_data['Hour'].apply(lambda x : math.floor(x))


# In[18]:


# Calculating First and Second day
credit_card_data['Day'] = credit_card_data['Time_hr']/24   # 2 days of data
credit_card_data['Day'] = credit_card_data['Day'].apply(lambda x : 1 if(x==0) else math.ceil(x))


# In[19]:


credit_card_data[['Time','Time_hr','Hour','Day','Amount','Class']].head(10)


# In[20]:


# calculating fraud transaction daywise
dayFrdTran = credit_card_data[(credit_card_data['Class'] == 1)]['Day'].value_counts()

# calculating genuine transaction daywise
dayGenuTran = credit_card_data[(credit_card_data['Class'] == 0)]['Day'].value_counts()

# calculating total transaction daywise
dayTran = credit_card_data['Day'].value_counts()

print("No of Transaction Day wise:")
print(dayTran)

print("\nNo of Fraud transaction Day wise:")
print(dayFrdTran)

print("\nNo of Genuine transactions Day wise:")
print(dayGenuTran)

print("\nPercentage of fraud transactions Day wise:")
print((dayFrdTran/dayTran)*100)


# #### Inferences
# 
# - Fraud transcation percentage is 0.19% on Day 1.
# 
# - Fraud transcation percentage is 0.15% on Day 2.
# 
# - Fraud transaction are more on Day 1 as compare to Day 2.
# 
# Let's see the above numbers in the graph.

# ### Day wise transcation distribution

# In[21]:


fig = plt.figure(figsize=(20, 6))
fig.set_facecolor("lightgrey")

plt.subplot(1,3,1)
sns.countplot(credit_card_data['Day'])
plt.title("Distribution of Total Transactions",fontsize=12,family = "Comic Sans MS")
plt.ylabel('Count', fontsize=14,family = "Comic Sans MS")
plt.xlabel('Day', fontsize=14,family = "Comic Sans MS")

plt.subplot(1,3,2)
sns.countplot(credit_card_data[(credit_card_data['Class'] == 1)]['Day'])
plt.title("Distribution of Fraud Transactions",fontsize=12,family = "Comic Sans MS")
plt.ylabel('Count', fontsize=14,family = "Comic Sans MS")
plt.xlabel('Day', fontsize=14,family = "Comic Sans MS")

plt.subplot(1,3,3)
sns.countplot(credit_card_data[(credit_card_data['Class'] == 0)]['Day'])
plt.title("Distribution of Genuine Transactions",fontsize=12,family = "Comic Sans MS")
plt.ylabel('Count', fontsize=14,family = "Comic Sans MS")
plt.xlabel('Day', fontsize=14,family = "Comic Sans MS")


# In[22]:


# Let's see if we find any particular pattern between time ( in hours ) and Fraud vs Genuine Transactions

plt.figure(figsize=(10,8))

sns.distplot(credit_card_data[credit_card_data['Class'] == 0]["Hour"], color='green') # Genuine - green
sns.distplot(credit_card_data[credit_card_data['Class'] == 1]["Hour"], color='red') # Fraudulent - Red

plt.title('Fraud vs Genuine Transactions by Hours', fontsize=15)
plt.xlim([0,25])
plt.show()


# #### Inferences
# - Fraudulent transactions have a distribution even - are equaly distributed in time

# ### Corelation between V1-V14, Amount & Time features

# In[23]:


corr_one = credit_card_data[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 
                             'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'Amount', 'Time']]
plt.figure(figsize = (12,12))
plt.title('Credit Card Transactions features correlation plot - I (Pearson)')
corr = corr_one.corr()
sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,linewidths=.1,cmap="Blues")
plt.show()


# #### Inferences
# 
# - **No correlation between features V1-V14** as they are the principal components obtained with PCA.
# - Certainly there is some correlations between features & Time (**inverse correlation with V3**)
# - Strong correlation between features & Amount (**direct correlation with V7 , inverse correlation with V1 and V5**).
# 

# ### Corelation between V15-V28, Amount & Time features

# In[24]:


corr_one = credit_card_data[['V15','V16','V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 
                             'V26', 'V27','V28', 'Amount', 'Time']]
plt.figure(figsize = (12,12))
plt.title('Credit Card Transactions features correlation plot - II (Pearson)')
corr = corr_one.corr()
sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,linewidths=.1,cmap="Greens")
plt.show()


# #### Inferences
# 
# - **No correlation between features V15-V28** as they are the principal components obtained with PCA.
# - Strong correlation between features & Amount (**direct correlation with V22**).
# 
# Lets have a closure look on co-related variables.

# ### Closer look to correlated Variables V7 & V22

# In[25]:


fig = plt.figure(figsize=(14,6))
sns.set_style("whitegrid")
fig.set_facecolor("lightgrey")

sns.lmplot(x='V7', y='Amount',data=credit_card_data, hue='Class', fit_reg=True,markers=["o", "x"])
plt.ylabel("V7", fontsize=18,family = "Comic Sans MS")
plt.xlabel('Amount', fontsize=18,family = "Comic Sans MS")
plt.title("V7 w.r.t. Amount", fontsize=18,family = "Comic Sans MS")

plt.show()


# In[26]:


fig = plt.figure(figsize=(20,8))
sns.set_style("whitegrid")
fig.set_facecolor("lightgrey")

sns.lmplot(x='V20', y='Amount',data=credit_card_data, hue='Class',markers=["o", "x"])
plt.ylabel("V20", fontsize=18,family = "Comic Sans MS")
plt.xlabel('Amount', fontsize=18,family = "Comic Sans MS")
plt.title("V20 w.r.t. Amount", fontsize=18,family = "Comic Sans MS")
plt.show()


# #### Inferences
# 
# - V22 and V7 features are correlated (the regression lines for Class = 0 have a positive slope, whilst the regression line for Class = 1 have a smaller positive slope).
# 

# ### Closer look to correlated Variables V5 & V2

# In[27]:


fig = plt.figure(figsize=(20,8))
sns.set_style("whitegrid")
fig.set_facecolor("lightgrey")

sns.lmplot(x='V2', y='Amount',data=credit_card_data, hue='Class',markers=["o", "x"])
plt.ylabel("V2", fontsize=18,family = "Comic Sans MS")
plt.xlabel('Amount', fontsize=18,family = "Comic Sans MS")
plt.title("V2 w.r.t. Amount", fontsize=18,family = "Comic Sans MS")
plt.show()


# In[28]:


fig = plt.figure(figsize=(20,8))
sns.set_style("whitegrid")
fig.set_facecolor("lightgrey")

sns.lmplot(x='V5', y='Amount',data=credit_card_data, hue='Class',markers=["o", "x"])
plt.ylabel("V5", fontsize=18,family = "Comic Sans MS")
plt.xlabel('Amount', fontsize=18,family = "Comic Sans MS")
plt.title("V5 w.r.t. Amount", fontsize=18,family = "Comic Sans MS")
plt.show()


# #### Inferences
# 
# - V2 & V5 features are inverse correlated (**the regression lines for Class = 0 have a negative slope while the regression lines for Class = 1 have a very small negative slope**).

# ### Distribution of features based on Genuine or Fraud

# In[29]:


def dist_Genuine_Vs_Fraud(colname):
    i = 0
    sns.set_style('whitegrid')
    fig = plt.figure(figsize=(22, 75))

    for feature in colname:
        i += 1
        plt.subplot(10,3,i)
        #Used divided sets for ploting the graphs.
        sns.kdeplot(genuine[feature], bw=0.5,label="Genuine")
        sns.kdeplot(fraud[feature], bw=0.5,label="Fraud")
        #Setting the y can x labels
        plt.ylabel('Density plot', fontsize=12)
        plt.xlabel(feature, fontsize=12)
        locs, labels = plt.xticks()
        plt.tick_params(axis='both', which='major', labelsize=12)
    plt.show();


# In[30]:


# plotting kde plot from the dataset to see the skewness
col_name_num = ['V1', 'V2', 'V3','V4', 'V5', 'V6', 'V7', 'V8', 'V9',
       'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18',
       'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27',
       'V28','Amount','Time']


# In[31]:


dist_Genuine_Vs_Fraud(col_name_num)


# #### Inferences
# 
# - Features - V4, V11 have clearly separated distributions for Class values 0 and 1, 
# - Features - V12, V14, V18 are partially separated distribution.
# - Features - V25, V26, V28 have similar profiles for the two values of Class.
# - There are few exceptions (Time and **Amount**), which need to be **mitigate**.
# - The features distribution for genuine transactions (values of Class = 0) is centered around 0
# - The feture distribution for fraudulent transactions (values of Class = 1) have a skewed (asymmetric) distribution.
# 

# ### Scale Amount feature 
# - Scale the data so that the column(feature) with lesser significance might not end up dominating the objective function due to its larger range. 
# - In addition, features having different unit should also be scaled thus providing each feature equal initial weightage.
# 
# This will result in a better prediction model.

# In[32]:


# Recaling the Amount variable using Log, MinMax & Scalar and comparing them.

# Scale amount by log
credit_card_data['Amt_Log'] = np.log(credit_card_data.Amount + 0.01)

# object of the class StandardScaler ()
ss = StandardScaler() 
credit_card_data['Amt_Scaler'] = ss.fit_transform(credit_card_data['Amount'].values.reshape(-1,1))

# object of the class MinMaxScaler ()
mms = MinMaxScaler() 
credit_card_data['Amt_MinMax'] = mms.fit_transform(credit_card_data['Amount'].values.reshape(-1,1))

# object of the class RobustScaler ()
rs = RobustScaler() 
credit_card_data['Amt_Robust'] = rs.fit_transform(credit_card_data['Amount'].values.reshape(-1,1))


# In[33]:


plt.figure(figsize=(22,6))

plt.subplot(1,3,1)
sns.boxplot(x = 'Class', y = 'Amount', data = credit_card_data, palette=("g",'b'),showfliers=True)
plt.title('Boxplot Class vs Amt',fontsize=12,family = "Comic Sans MS")
plt.xlabel('Class', fontsize=12,family = "Comic Sans MS")

plt.subplot(1,3,2)
sns.boxplot(x = 'Class', y = 'Amt_Log', data = credit_card_data, palette=("g",'r'),showfliers=True)
plt.title('Boxplot Class vs Amt_Log',fontsize=12,family = "Comic Sans MS")
plt.xlabel('Class', fontsize=12,family = "Comic Sans MS")

plt.subplot(1,3,3)
sns.boxplot(x = 'Class', y = 'Amt_Scaler', data = credit_card_data, palette=("g",'r'),showfliers=True)
plt.title('Boxplot Class vs Amt_Scaler',fontsize=12,family = "Comic Sans MS")
plt.xlabel('Class', fontsize=12,family = "Comic Sans MS")
plt.show()

plt.figure(figsize=(17,5))

plt.subplot(1,3,1)
sns.boxplot(x = 'Class', y = 'Amt_MinMax', data = credit_card_data, palette=("g",'r'),showfliers=True)
plt.title('Boxplot Class vs Amt_MinMax',fontsize=12,family = "Comic Sans MS")
plt.xlabel('Class', fontsize=12,family = "Comic Sans MS")

plt.subplot(1,3,2)
sns.boxplot(x = 'Class', y = 'Amt_Robust', data = credit_card_data, palette=("g",'r'),showfliers=True)
plt.title('Boxplot Class vs Amt_Robust',fontsize=12,family = "Comic Sans MS")
plt.xlabel('Class', fontsize=12,family = "Comic Sans MS")

plt.show()


# #### Inferences
# 
# - Slight difference in the log amount of our two Classes.
# - The IQR of fraudulent transactions are higher than normal transactions, but normal transactions have the highest values.
# - By seeing the above graphs, Scaling the amount by log will best suit for our model.

# ### Copy of data set

# In[34]:


credit_card_data_bkup = credit_card_data.copy()


# In[35]:


credit_card_data.columns


# ## Train & Test Split

# ### Separate Target Variable and Predictor Variables
# - Removed unwanted features or columns

# In[36]:


X = credit_card_data.drop(['Time','Class','Time_hr', 'Hour', 'Day','Amount', 'Amt_Scaler',
       'Amt_MinMax', 'Amt_Robust'],axis=1)
y = credit_card_data['Class']


# In[37]:


X.rename(columns = {'Amt_Log':'Amount'}, inplace = True) 


# In[38]:


X.columns


# ### Stratified split 
# - It will ensure that test dataset has at least 100 records corresponding to the minority class

# In[39]:


# Split the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, shuffle=True, random_state=101)
print(" Train Data Set \n\t",X_train.shape, "\n\t", y_train.shape)
print(" Test Data Set \n\t",X_test.shape, "\n\t", y_test.shape)


# In[40]:


print(" \t Train Data set with Fraud and Genuine distribution ")
print(y_train.value_counts())
print("\n \t Test Data set with Fraud and Genuine distribution ")
print(y_test.value_counts())
print("\n \t Total Minority class data is \n ",np.sum(y))


# #### Inferences 
# - **Minority** class has **344 rows** in Train date set.
# - **Minority** class has **148 rows** in Test date set.
# - Total count of **Minority** class in Train and Test data set is **492**.
# ##### Preserve X_test & y_test to evaluate on the test data once you build the model

# ### Skewness

# In[41]:


def distplot(df,colnames) :
    rows = math.ceil(len(colnames) / 3)
    fig = plt.figure(figsize=(30, 75))
    fig.set_facecolor("lightgrey")
    for i in range(0,len(colnames)):
        plt.subplot(rows,3,i+1)
        sns.distplot(df[str(colnames[i])],color='r')
        #str2 = colnames[i] + " :- Distribution plot for skewness"
        str2 = colnames[i] + " Skew: " + str(np.round(skew(df[colnames[i]]),2)) + "\n" + colnames[i] + " Kurtosis: " + str(np.round(kurtosis(df[colnames[i]]),2))
        #plt.title(cols[i] + 
        plt.title(str2, fontsize=12,family = "Comic Sans MS")
        plt.xlabel(colnames[i], fontsize=14,family = "Comic Sans MS")


# In[42]:


# plotting histograms from the dataset to see the skewness
col_name_skewness = X_train.columns
distplot(X_train,col_name_skewness)


# #### Inferences
# * Most of the variables are highly skewed.
# * Applying PowerTransformer. We will use Yeo Johnson method instead of Box Cox Transformation in order to take care of negative values.

# ### Power Transfor using yeo-johnson Method

# In[43]:


# - Apply : preprocessing.PowerTransformer(copy=False) to fit & transform the train & test data
pt = PowerTransformer(method='yeo-johnson', standardize=True, copy=False)
X_train[col_name_skewness] = pt.fit_transform(X_train[col_name_skewness])


# In[44]:


distplot(X_train,col_name_skewness)


# ### Utility variables for Model metrics (train and Test) & ROC curve 

# In[45]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(6, 6))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return fpr, tpr, thresholds


# In[46]:


# evaluate predictions on train subset
def modelMetrics_train(model, X_train, y_train):
    
    # use predict_proba for ROC_AUC
    y_pred = model.predict_proba(X_train)[:,1]
    
    roc = metrics.roc_auc_score(y_train, y_pred)
    print("Train AUC: %.2f%%" % (roc * 100.0))

    fpr, tpr, thresholds = metrics.roc_curve(y_train, y_pred)
    threshold = thresholds[np.argmax(tpr-fpr)]
    print("Train ROC Curve Threshold: %.3f" % threshold)
    draw_roc(y_train, y_pred)
    
    print("Computing Predictions based on threshold value of ", threshold, " and then computing below metrics\n")

    y_pred_final = y_pred > threshold
    y_pred_final = y_pred_final.astype(int)

    print("Train Accuracy: ",metrics.accuracy_score(y_train, y_pred_final))
    print("Train Classification Report: \n", classification_report(y_train, y_pred_final))
    cm=confusion_matrix(y_train, y_pred_final)
    print("Train Confusion Matrix:\n",cm)
    return threshold


# In[47]:


# evaluate predictions
def modelMetrics_test(model, threshold_train_val):
    # use predict_proba for ROC_AUC
    y_pred = model.predict_proba(X_test)[:,1]

    y_pred = y_pred > threshold_train_val
    y_pred = y_pred.astype(int)

    roc = metrics.roc_auc_score(y_test, y_pred)
    print("Test AUC: %.2f%%" % (roc * 100.0))

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    draw_roc(y_test, y_pred)

    print("Accuracy on Test Dataset: ",metrics.accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    cm=confusion_matrix(y_test, y_pred)
    print("Confusion Matrix on Test Dataset:\n",cm)


# In[48]:


def OptimalCutoff(y_pred_final,x1,y1):
    # Let's create columns with different probability cutoffs 
    numbers = [float(x)/10 for x in range(10)]
    for i in numbers:
        y_pred_final[i]= y_pred_final.Fraud_Prob.map(lambda x: 1 if x > i else 0)

    # Calculating accuracy sensitivity and specificity for various probability cutoffs.
    cutoff_df = pd.DataFrame( columns = ['Probability','Accuracy','Sensitivity','Specificity'])

    num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for i in num:
        cm1 = metrics.confusion_matrix(y_pred_final.Fraud, y_pred_final[i] )
        total1=sum(sum(cm1))
        accuracy = (cm1[0,0]+cm1[1,1])/total1
        specificity = cm1[0,0]/(cm1[0,0]+cm1[0,1])
        sensitivity = cm1[1,1]/(cm1[1,0]+cm1[1,1])
        cutoff_df.loc[i] =[ i ,accuracy,sensitivity,specificity]
    print(cutoff_df)
    
    # Let's plot accuracy sensitivity and specificity for various probabilities.
    cutoff_df.plot.line(x='Probability', y=['Accuracy','Sensitivity','Specificity'])
    plt.vlines(x=x1,ymax=0.9,ymin=0.0,color="g",linestyles="--")
    plt.hlines(y=y1,xmax=0.9,xmin=0.0,color="b",linestyles="--")
    plt.show()


# ### Methods to mitigate class Imbalanced dataset treatment
# 
# - Undersampling
# - Oversampling
# - SMOTE
# - Borderline SMOTE
# - ADASYN

# In[49]:


# Define the resampling method
ranundersam = RandomUnderSampler(random_state=0)
ranoversam = RandomOverSampler(random_state=0)
smote = SMOTE(random_state=0)
borderlinesmote = BorderlineSMOTE(kind='borderline-2',random_state=0)
adasyn = ADASYN(random_state=42)


# In[50]:


# resample the training data
X_train_rus, y_train_rus = ranundersam.fit_sample(X_train,y_train)
X_train_ros, y_train_ros = ranoversam.fit_sample(X_train,y_train)
X_train_sm, y_train_sm = smote.fit_sample(X_train,y_train)
X_train_blsm, y_train_blsm = borderlinesmote.fit_sample(X_train,y_train)
X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)


# In[51]:


print('Resampled dataset shape using Random Under sample -  %s' % Counter(y_train_rus))
print('Resampled dataset shape using Random over sample - %s' % Counter(y_train_ros))
print('Resampled dataset shape using SMOTE - %s' % Counter(y_train_sm))
print('Resampled dataset shape using Border line SMOTE - %s' % Counter(y_train_blsm))
print('Resampled dataset shape using Adasyn - %s' % Counter(y_train_adasyn))


# ## Model Building & Hyperparameter tuning
# 

# ##### For the prediction of Fraud customers we will be fitting variety of models with imblanced and balanced data set and select one which is the best predictor of Fraud. 
# 
# - Different models on the **imbalanced** & **balanced** dataset and see the result
#   - Logistic Regression
#   - Random Forest
#   - Decision Tree
#   - XGBoost

# ### Logistic on Imbalanced Data set

# In[ ]:


lr_imbal =LogisticRegression(random_state=42)


# In[ ]:


lr_imbal.fit(X_train, y_train)


# In[ ]:


# Evaluation of model performance on Train subset
threshold_train_val = modelMetrics_train(lr_imbal, X_train, y_train)


# In[ ]:


# Evaluation of model performance on Test subset
modelMetrics_test(lr_imbal, threshold_train_val)


# In[ ]:


# Create the parameter grid based
param_grid = {'C': [0.01, 0.1, 1, 10, 100],
             'penalty': ['l1', 'l2']}
# Create a based model
lr_imbal_hp = LogisticRegression(random_state=42)
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = lr_imbal_hp, 
                           param_grid = param_grid,
                           cv=StratifiedKFold(5).split(X_train, y_train), 
                           scoring="roc_auc",
                           return_train_score=True, 
                           n_jobs = -1,
                           verbose = 1)


# In[ ]:


grid_search.fit(X_train,y_train)


# In[ ]:


# printing the optimal score and hyperparameters
print('Best score ',grid_search.best_score_,'using',grid_search.best_params_)


# In[ ]:


# model with the best hyperparameters
lr_imbal_hp = LogisticRegression(C=0.01, penalty='l2',solver='liblinear', random_state=42)
lr_imbal_hp.fit(X_train, y_train)


# In[ ]:


# Evaluation of model performance on Train subset
threshold_train_val = modelMetrics_train(lr_imbal_hp, X_train, y_train)


# In[ ]:


# Evaluation of model performance on Test subset
modelMetrics_test(lr_imbal_hp, threshold_train_val)


# In[ ]:





# ### Logistic on balanced data set using Random Oversampling

# In[ ]:


# Create the parameter grid based
param_grid = {'C': [0.01, 0.1, 1, 10, 100],
             'penalty': ['l1', 'l2']}
# Create a based model
lr_bal_hp = LogisticRegression(random_state=42)
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = lr_bal_hp, 
                           param_grid = param_grid,
                           cv=StratifiedKFold(5).split(X_train_ros, y_train_ros), 
                           scoring="roc_auc",
                           return_train_score=True, 
                           n_jobs = -1,
                           verbose = 1)


# In[ ]:


grid_search.fit(X_train_ros, y_train_ros)


# In[ ]:


# printing the optimal score and hyperparameters
print('Best score ',grid_search.best_score_,'using',grid_search.best_params_)


# In[ ]:


# model with the best hyperparameters
lr_bal_hp_ros = LogisticRegression(C=100, penalty='l2', solver='liblinear',random_state=42)
lr_bal_hp_ros.fit(X_train_ros, y_train_ros)


# In[ ]:


# Evaluation of model performance on Train subset
threshold_train_val = modelMetrics_train(lr_bal_hp_ros, X_train_ros, y_train_ros)


# In[ ]:


# Evaluation of model performance on Test subset
modelMetrics_test(lr_bal_hp_ros, threshold_train_val)


# ### Logistic on balanced data set using SMOTE

# In[ ]:


# Create the parameter grid based
param_grid = {'C': [0.01, 0.1, 1, 10, 100],
             'penalty': ['l1', 'l2']}
# Create a based model
lr_bal_hp = LogisticRegression(random_state=42)
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = lr_bal_hp, 
                           param_grid = param_grid,
                           cv=StratifiedKFold(5).split(X_train_sm, y_train_sm), 
                           scoring="roc_auc",
                           return_train_score=True, 
                           n_jobs = -1,
                           verbose = 1)


# In[ ]:


grid_search.fit(X_train_sm, y_train_sm)


# In[ ]:


# printing the optimal score and hyperparameters
print('Best score ',grid_search.best_score_,'using',grid_search.best_params_)


# In[ ]:


# model with the best hyperparameters
lr_bal_sm = LogisticRegression(C=100, penalty='l2',solver='liblinear', random_state=42)
lr_bal_sm.fit(X_train_sm, y_train_sm)


# In[ ]:


# Evaluation of model performance on Train subset
threshold_train_val = modelMetrics_train(lr_bal_sm, X_train_sm, y_train_sm)


# In[ ]:


# Evaluation of model performance on Test subset
modelMetrics_test(lr_bal_sm, threshold_train_val)


# In[ ]:





# ### Logistic on balanced data set using Border line SMOTE

# In[ ]:


# Create the parameter grid based
param_grid = {'C': [0.01, 0.1, 1, 10, 100],
             'penalty': ['l1', 'l2']}
# Create a based model
lr_bal_hp = LogisticRegression(random_state=42)
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = lr_bal_hp, 
                           param_grid = param_grid,
                           cv=StratifiedKFold(5).split(X_train_blsm, y_train_blsm), 
                           scoring="roc_auc",
                           return_train_score=True, 
                           n_jobs = -1,
                           verbose = 1)


# In[ ]:


grid_search.fit(X_train_blsm, y_train_blsm)


# In[ ]:


# printing the optimal score and hyperparameters
print('Best score ',grid_search.best_score_,'using',grid_search.best_params_)


# In[ ]:


# model with the best hyperparameters
lr_bal_blsm = LogisticRegression(C=10, penalty='l2',solver='liblinear', random_state=42)
lr_bal_blsm.fit(X_train_blsm, y_train_blsm)


# In[ ]:


# Evaluation of model performance on Train subset
threshold_train_val = modelMetrics_train(lr_bal_blsm, X_train_blsm, y_train_blsm)


# In[ ]:


# Evaluation of model performance on Test subset
modelMetrics_test(lr_bal_blsm, threshold_train_val)


# In[ ]:





# ### Logistic on balanced data set using ADASYN

# In[ ]:


# Create the parameter grid based
param_grid = {'C': [0.01, 0.1, 1, 10, 100],
             'penalty': ['l1', 'l2']}
# Create a based model
lr_bal_hp = LogisticRegression(random_state=42)
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = lr_bal_hp, 
                           param_grid = param_grid,
                           cv=StratifiedKFold(5).split(X_train_adasyn, y_train_adasyn), 
                           scoring="roc_auc",
                           return_train_score=True, 
                           n_jobs = -1,
                           verbose = 1)


# In[ ]:


grid_search.fit(X_train_adasyn, y_train_adasyn)


# In[ ]:


# printing the optimal score and hyperparameters
print('Best score ',grid_search.best_score_,'using',grid_search.best_params_)


# In[ ]:


# model with the best hyperparameters
lr_bal_adasyn = LogisticRegression(C=100, penalty='l2',solver='liblinear', random_state=42)
lr_bal_adasyn.fit(X_train_adasyn, y_train_adasyn)


# In[ ]:


# Evaluation of model performance on Train subset
threshold_train_val = modelMetrics_train(lr_bal_adasyn, X_train_adasyn, y_train_adasyn)


# In[ ]:


# Evaluation of model performance on Test subset
modelMetrics_test(lr_bal_adasyn, threshold_train_val)


# In[ ]:





# #### Inferences 
# 
# | Model                | Train Dataset       | Hyperparameters              | Train AUC   | Threshold | Test AUC |
# | :-------------       | :-------------      | :----------:                 | -----------: | -----------: | -----------: |
# |  Logistic Regression |  Imbalanced         | Default                      | 97.89%   | 0.003 | 93.78% |
# |  Logistic Regression |  Imbalanced         | {'C': 0.01, 'penalty': 'l2'} | 98.12%   | 0.007 | 93.87% |
# |  Logistic Regression |  Random Oversampler | {'C': 100, 'penalty': 'l2'}  | 98.54%   | 0.471 | 93.98% |
# |  Logistic Regression |  SMOTE              | {'C': 100, 'penalty': 'l2'}  | 98.83%   | 0.454 | 93.86% |
# |  Logistic Regression |  Border line SMOTE  | {'C': 10, 'penalty': 'l2'}  | 99.90%   | 0.417 | 94.12% |
# |  Logistic Regression |  ADASYN             | {'C': 100, 'penalty': 'l2'}  | 96.23%   | 0.325 | 92.65% |

# ### K-Nearest Neighbours

# In[ ]:


knn_dt=KNeighborsClassifier()
knn_dt.fit(X_train,y_train)


# In[ ]:


# Evaluation of model performance on Train subset
threshold_train_val = modelMetrics_train(knn_dt, X_train, y_train)


# In[ ]:


# Evaluation of model performance on Test subset
modelMetrics_test(knn_dt, threshold_train_val)


# In[ ]:


k_range = list(range(1,8,2))
score1=[]
score2=[]
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    score1.append(knn.score(X_train,y_train))
    score2.append(knn.score(X_test,y_test))
    
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(k_range,score1,label= 'Accuracy on training set')
plt.plot(k_range,score2,label= 'Accuracy on testing set')
plt.xlabel('Value of K in KNN')
plt.ylabel('Accuracy')
plt.legend()


# In[ ]:


knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)


# In[ ]:


# Evaluation of model performance on Train subset
threshold_train_val = evaluation_metrics_train(knn, X_train, y_train)


# In[ ]:


# Evaluation of model performance on Test subset
evaluation_metrics_test(knn, threshold_train_val)


# In[ ]:





# In[ ]:





# ### Utility functions used for Random Forest

# In[53]:


def rf_tuning_max_depth(X_train, y_train):
    # Tuning max_depth parameters to build the model on
    parameters = {'max_depth': range(2, 20, 5)}

    # instantiate the model
    rf = RandomForestClassifier()

    # fit tree on training data
    rf = GridSearchCV(rf, parameters, 
                      cv=StratifiedKFold(5).split(X_train, y_train), 
                      scoring="roc_auc",
                      return_train_score=True)
    rf.fit(X_train, y_train)

    scores = rf.cv_results_
    
    fig = plt.figure(figsize=(25,20))
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})

    # plotting accuracies with max_depth
    plt.figure()
    plt.plot(scores["param_max_depth"],scores["mean_train_score"], label="Training roc_auc")
    plt.plot(scores["param_max_depth"],scores["mean_test_score"], label="Test roc_auc")
    plt.xlabel("max_depth",fontsize=12,family = "Comic Sans MS")
    plt.ylabel("roc_auc", fontsize=12,family = "Comic Sans MS")
    plt.legend()
    plt.show()


# In[54]:


def rf_tuning_n_estimators(X_train, y_train):
    #Tuning n_estimators parameters to build the model on
    parameters = {'n_estimators': range(100, 300, 100)}

    # instantiate the model
    rf = RandomForestClassifier(max_depth=4)

    # fit tree on training data
    rf = GridSearchCV(rf, parameters, 
                      cv=StratifiedKFold(5).split(X_train, y_train), 
                      scoring="roc_auc",
                      return_train_score=True)
    rf.fit(X_train, y_train)

    # scores of GridSearch CV
    scores = rf.cv_results_

    fig = plt.figure(figsize=(25,20))
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})

    # plotting accuracies with n_estimators
    plt.figure()
    plt.plot(scores["param_n_estimators"],scores["mean_train_score"], label="Training roc_auc")
    plt.plot(scores["param_n_estimators"],scores["mean_test_score"], label="Test roc_auc")
    plt.xlabel("n_estimators",fontsize=12,family = "Comic Sans MS")
    plt.ylabel("roc_auc", fontsize=12,family = "Comic Sans MS")
    plt.legend()
    plt.show()


# In[55]:


def rf_tuning_max_features(X_train, y_train):
    #Tuning max_features parameters to build the model on
    parameters = {'max_features': [2, 4, 8, 10]}

    # instantiate the model
    rf = RandomForestClassifier(max_depth=4)

    # fit tree on training data
    rf = GridSearchCV(rf, parameters, 
                      cv=StratifiedKFold(5).split(X_train, y_train), 
                      scoring="roc_auc",
                      return_train_score=True)
    rf.fit(X_train, y_train)

    # scores of GridSearch CV
    scores = rf.cv_results_
    
    fig = plt.figure(figsize=(25,20))
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})

    # plotting roc_auc with max_features
    plt.figure()
    plt.plot(scores["param_max_features"],scores["mean_train_score"], label="Training roc_auc")
    plt.plot(scores["param_max_features"],scores["mean_test_score"], label="Test roc_auc")
    plt.xlabel("max_features",fontsize=12,family = "Comic Sans MS")
    plt.ylabel("roc_auc", fontsize=12,family = "Comic Sans MS")
    plt.legend()
    plt.show()


# In[56]:


def rf_tuning_min_samples_leaf(X_train, y_train):
    # Tuning min_samples_leaf parameters to build the model on
    parameters = {'min_samples_leaf': range(100, 400, 50)}

    # instantiate the model
    rf = RandomForestClassifier(max_depth=4)

    # fit tree on training data
    rf = GridSearchCV(rf, parameters, 
                      cv=StratifiedKFold(5).split(X_train, y_train), 
                      scoring="roc_auc",
                      return_train_score=True)
    rf.fit(X_train, y_train)

    # scores of GridSearch CV
    scores = rf.cv_results_

    fig = plt.figure(figsize=(25,20))
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})

    # plotting roc_auc with min_samples_leaf
    plt.figure()
    plt.plot(scores["param_min_samples_leaf"],scores["mean_train_score"], label="Training roc_auc")
    plt.plot(scores["param_min_samples_leaf"],scores["mean_test_score"], label="Test roc_auc")
    plt.xlabel("min_samples_leaf",fontsize=12,family = "Comic Sans MS")
    plt.ylabel("roc_auc", fontsize=12,family = "Comic Sans MS")
    plt.legend()
    plt.show()


# In[57]:


def rf_tuning_min_samples_split(X_train, y_train):
    # Tuning min_samples_split parameters to build the model on
    parameters = {'min_samples_split': range(200, 500, 50)}

    # instantiate the model
    rf = RandomForestClassifier(max_depth=4)

    # fit tree on training data
    rf = GridSearchCV(rf, parameters, 
                      cv=StratifiedKFold(5).split(X_train, y_train), 
                      scoring="roc_auc",
                      return_train_score=True)
    rf.fit(X_train, y_train)

    # scores of GridSearch CV
    scores = rf.cv_results_

    fig = plt.figure(figsize=(25,20))
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})

    # plotting roc_auc with min_samples_split
    plt.figure()
    plt.plot(scores["param_min_samples_split"],scores["mean_train_score"], label="Training roc_auc")
    plt.plot(scores["param_min_samples_split"],scores["mean_test_score"], label="Test roc_auc")
    plt.xlabel("min_samples_split",fontsize=12,family = "Comic Sans MS")
    plt.ylabel("roc_auc", fontsize=12,family = "Comic Sans MS")
    plt.legend()
    plt.show()


# ### Random Forest Classifier on imbalanced data set

# In[ ]:


# Running the random forest with default parameters.
rfc_imbal = RandomForestClassifier(random_state=42)
rfc_imbal.fit(X_train,y_train)


# In[ ]:


# Evaluation of model performance on Train subset
threshold_train_val = modelMetrics_train(rfc_imbal, X_train, y_train)


# In[ ]:


# Evaluation of model performance on Test subset
modelMetrics_test(rfc_imbal, threshold_train_val)


# In[ ]:


rf_tuning_max_depth(X_train, y_train)


# In[ ]:


rf_tuning_n_estimators(X_train, y_train)


# In[ ]:


rf_tuning_max_features(X_train, y_train)


# In[ ]:


rf_tuning_min_samples_leaf(X_train, y_train)


# In[ ]:


rf_tuning_min_samples_split(X_train, y_train)


# ##### Grid Search to Find Optimal Hyperparameters

# In[ ]:


# Create the parameter grid based on the results of random search 
param_grid = {
    'max_depth': [7],
    'min_samples_leaf': [150,200],
    'min_samples_split': [250,350],
    'n_estimators': [150, 160], 
    'max_features': [4,8]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv=StratifiedKFold(5).split(X_train, y_train), 
                  scoring="roc_auc",
                  return_train_score=True, n_jobs = -1,verbose = 1)


# In[ ]:


# Fit the grid search to the data
grid_search.fit(X_train, y_train)


# In[ ]:


# printing the optimal  score and hyperparameters
print('Best score ',grid_search.best_score_,'using',grid_search.best_params_)


# In[ ]:


# model with the best hyperparameters
rfc = RandomForestClassifier(bootstrap=True,
                             max_depth=7,
                             min_samples_leaf=150, 
                             min_samples_split=350,
                             max_features=8,
                             n_estimators=150)


# In[ ]:


# fit
rfc.fit(X_train,y_train)


# In[ ]:


# Evaluation of model performance on Train subset
threshold_train_val = modelMetrics_train(rfc, X_train, y_train)


# In[ ]:


# Evaluation of model performance on Test subset
modelMetrics_test(rfc, threshold_train_val)


# ### Random Forest on balanced data set using Random Oversampling

# In[ ]:


# Running the random forest with default parameters.
rfc_bal_ros = RandomForestClassifier(random_state=42)
rfc_bal_ros.fit(X_train_ros,y_train_ros)


# In[ ]:


# Evaluation of model performance on Train subset
threshold_train_val = modelMetrics_train(rfc_bal_ros, X_train_ros, y_train_ros)


# In[ ]:


# Evaluation of model performance on Test subset
modelMetrics_test(rfc_bal_ros, threshold_train_val)


# In[ ]:


rf_tuning_max_depth(X_train_ros, y_train_ros)


# In[ ]:


rf_tuning_n_estimators(X_train_ros, y_train_ros)


# In[ ]:


rf_tuning_max_features(X_train_ros, y_train_ros)


# In[ ]:


rf_tuning_min_samples_leaf(X_train_ros, y_train_ros)


# In[ ]:


rf_tuning_min_samples_split(X_train_ros, y_train_ros)


# In[ ]:


# Create the parameter grid based on the results of random search 
param_grid_ros = {
    'max_depth': [12],
    'min_samples_leaf': [300],
    'min_samples_split': [250,400],
    'n_estimators': [200], 
    'max_features': [8]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid_ros, 
                          cv=StratifiedKFold(5).split(X_train_ros, y_train_ros), 
                  scoring="roc_auc",
                  return_train_score=True, n_jobs = -1,verbose = 1)


# In[ ]:


# Fit the grid search to the data
grid_search.fit(X_train_ros, y_train_ros)


# In[ ]:


# printing the optimal  score and hyperparameters
print('Best Score of',grid_search.best_score_,'using',grid_search.best_params_)


# In[58]:


# model with the best hyperparameters
rfc = RandomForestClassifier(bootstrap=True,
                             max_depth=12,
                             min_samples_leaf=300, 
                             min_samples_split=400,
                             max_features=8,
                             n_estimators=200)


# In[59]:



# fit
rfc.fit(X_train_ros,y_train_ros)


# In[60]:


# Evaluation of model performance on Train subset
threshold_train_val = modelMetrics_train(rfc, X_train_ros, y_train_ros)


# In[61]:


# Evaluation of model performance on Test subset
modelMetrics_test(rfc, threshold_train_val)


# In[ ]:





# ### Random Forest on balanced data set using SMOTE
# 

# In[ ]:


# Running the random forest with default parameters.
rfc_bal_sm = RandomForestClassifier(random_state=42)
rfc_bal_sm.fit(X_train_sm,y_train_sm)


# In[ ]:


# Evaluation of model performance on Train subset
threshold_train_val = modelMetrics_train(rfc_bal_sm, X_train_sm, y_train_sm)


# In[ ]:


# Evaluation of model performance on Test subset
modelMetrics_test(rfc_bal_sm, threshold_train_val)


# In[ ]:


rf_tuning_max_depth(X_train_sm, y_train_sm)


# In[ ]:


rf_tuning_n_estimators(X_train_sm, y_train_sm)


# In[ ]:


rf_tuning_max_features(X_train_sm, y_train_sm)


# In[ ]:


rf_tuning_min_samples_leaf(X_train_sm, y_train_sm)


# In[72]:


rf_tuning_min_samples_split(X_train_sm, y_train_sm)


# In[75]:


# Create the parameter grid based on the results of random search 
param_grid_ros = {
    'max_depth': [12],
    'min_samples_leaf': [300],
    'min_samples_split': [250,400],
    'n_estimators': [200], 
    'max_features': [8]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid_ros, 
                          cv=StratifiedKFold(5).split(X_train_sm, y_train_sm), 
                  scoring="roc_auc",
                  return_train_score=True, n_jobs = -1,verbose = 1)


# In[ ]:


# Fit the grid search to the data
grid_search.fit(X_train_sm, y_train_sm)


# In[ ]:


# printing the optimal  score and hyperparameters
print('Best Score ',grid_search.best_score_,'using',grid_search.best_params_)


# In[96]:


# model with the best hyperparameters
rfc_sm = RandomForestClassifier(bootstrap=True,
                             max_depth=7,
                             min_samples_leaf=100, 
                             min_samples_split=200,
                             max_features=8,
                             n_estimators=200)


# In[97]:


# fit
rfc_sm.fit(X_train_sm,y_train_sm)


# In[98]:


# Evaluation of model performance on Train subset
threshold_train_val = modelMetrics_train(rfc_sm, X_train_sm, y_train_sm)


# In[99]:


# Evaluation of model performance on Test subset
modelMetrics_test(rfc_sm, threshold_train_val)


# ### Random Forest on balanced data set using Border line SMOTE

# In[ ]:


# Running the random forest with default parameters.
rfc_bal_blsm = RandomForestClassifier(random_state=42)
rfc_bal_blsm.fit(X_train_blsm,y_train_blsm)


# In[ ]:


# Evaluation of model performance on Train subset
threshold_train_val = modelMetrics_train(rfc_bal_blsm, X_train_blsm, y_train_blsm)


# In[ ]:


# Evaluation of model performance on Test subset
modelMetrics_test(rfc_bal_blsm, threshold_train_val)


# In[ ]:


rf_tuning_max_depth(X_train_blsm, y_train_blsm)


# In[ ]:


rf_tuning_n_estimators(X_train_blsm, y_train_blsm)


# In[ ]:


rf_tuning_max_features(X_train_blsm, y_train_blsm)


# In[ ]:


rf_tuning_min_samples_leaf(X_train_blsm, y_train_blsm)


# In[ ]:


rf_tuning_min_samples_split(X_train_blsm, y_train_blsm)


# In[ ]:


# Create the parameter grid based on the results of random search 
param_grid_ros = {
    'max_depth': [7],
    'min_samples_leaf': [150,200],
    'min_samples_split': [200,300],
    'n_estimators': [200], 
    'max_features': [8]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid_ros, 
                          cv=StratifiedKFold(5).split(X_train_blsm, y_train_blsm), 
                  scoring="roc_auc",
                  return_train_score=True, n_jobs = -1,verbose = 1)


# In[ ]:


# Fit the grid search to the data
grid_search.fit(X_train_blsm, y_train_blsm)


# In[ ]:


# printing the optimal  score and hyperparameters
print('Best score ',grid_search.best_score_,'using',grid_search.best_params_)


# In[62]:


# model with the best hyperparameters
rfc = RandomForestClassifier(bootstrap=True,
                             max_depth=7,
                             min_samples_leaf=150, 
                             min_samples_split=200,
                             max_features=8,
                             n_estimators=200)


# In[63]:


# fit
rfc.fit(X_train_blsm,y_train_blsm)


# In[64]:


# Evaluation of model performance on Train subset
threshold_train_val = modelMetrics_train(rfc, X_train_blsm, y_train_blsm)


# In[65]:


# Evaluation of model performance on Test subset
modelMetrics_test(rfc, threshold_train_val)


# In[ ]:





# ### Random Forest on balanced data set using ADASYN

# In[ ]:


# Running the random forest with default parameters.
rfc_bal_adasyn = RandomForestClassifier(random_state=42)
rfc_bal_adasyn.fit(X_train_adasyn,y_train_adasyn)


# In[ ]:


# Evaluation of model performance on Train subset
threshold_train_val = modelMetrics_train(rfc_bal_adasyn, X_train_adasyn, y_train_adasyn)


# In[ ]:


# Evaluation of model performance on Test subset
modelMetrics_test(rfc_bal_adasyn, threshold_train_val)


# In[ ]:


rf_tuning_max_depth(X_train_adasyn, y_train_adasyn)


# In[ ]:


rf_tuning_n_estimators(X_train_adasyn, y_train_adasyn)


# In[ ]:


rf_tuning_max_features(X_train_adasyn, y_train_adasyn)


# In[ ]:


rf_tuning_min_samples_leaf(X_train_adasyn, y_train_adasyn)


# In[ ]:


rf_tuning_min_samples_split(X_train_adasyn, y_train_adasyn)


# In[74]:


param_grid = {
    'max_depth': [7],
    'min_samples_leaf': [100,150],
    'min_samples_split': [350],
    'n_estimators': [200], 
    'max_features': [4]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv=StratifiedKFold(5).split(X_train_adasyn, y_train_adasyn), 
                  scoring="roc_auc",
                  return_train_score=True, n_jobs = -1,verbose = 1)


# In[ ]:


grid_search.fit(X_train_adasyn, y_train_adasyn)


# In[ ]:


print('Best score ',grid_search.best_score_,'using',grid_search.best_params_)


# In[78]:


# model with the best hyperparameters
rfc_adasyn = RandomForestClassifier(bootstrap=True,
                             max_depth=7,
                             min_samples_leaf=150, 
                             min_samples_split=300,
                             max_features=8,
                             n_estimators=200)


# In[79]:


# fit
rfc_adasyn.fit(X_train_adasyn,y_train_adasyn)


# In[80]:


# Evaluation of model performance on Train subset
threshold_train_val = modelMetrics_train(rfc_adasyn, X_train_adasyn, y_train_adasyn)


# In[81]:


# Evaluation of model performance on Test subset
modelMetrics_test(rfc_adasyn, threshold_train_val)


# In[ ]:





# ### Inferences
# 
# | Model                | Train Dataset       | Hyperparameters              | Train AUC   | Threshold | Test AUC
# | :-------------       | :-------------      | :----------:                 | -----------: |-----------: |-----------: |
# |  Random Forest |  Imbalanced         | Default                      | 100%   | 0.530 | 90.87%   | 
# |  Random Forest |  Imbalanced         | {'max_depth': 7, 'max_features': 8, 'min_samples_leaf': 150, 'min_samples_split': 350, 'n_estimators': 150}  | 98.89%   | 0.002 | 91.19% |
# |  Random Forest |  Random Oversampler | {'max_depth': 7, 'max_features': 8, 'min_samples_leaf': 300, 'min_samples_split': 400, 'n_estimators': 200}  | 99.99%   | 0.612 | 93.52% |
# |  Random Forest |  SMOTE              | {'max_depth': 7, 'max_features': 8, 'min_samples_leaf': 100, 'min_samples_split': 200, 'n_estimators': 200}  | 99.67%   | 0.374 | 93.4% |
# |  Random Forest | Border line SMOTE   | {'max_depth': 7, 'max_features': 8, 'min_samples_leaf': 200, 'min_samples_split': 300, 'n_estimators': 100}  | 99.98%   | 0.543 | 93,93% |
# |  Random Forest |  ADASYN             | {'max_depth': 7, 'max_features': 8, 'min_samples_leaf': 150, 'min_samples_split': 300, 'n_estimators': 200} | 99.59%   | 0.462 | 94.86% |

# ### Utility functions for Decision Tree

# In[78]:


def dt_max_depth(X_train, y_train):
    # parameters to build the model on
    parameters = {'max_depth': range(1, 10, 2)}

    # instantiate the model
    dtree = DecisionTreeClassifier(criterion = "gini", 
                                   random_state = 100)

    # fit tree on training data
    tree = GridSearchCV(dtree, parameters, 
                        cv=StratifiedKFold(5).split(X_train, y_train), 
                       scoring="roc_auc", return_train_score=True)
    tree.fit(X_train, y_train)
    
    scores = tree.cv_results_
  
    fig = plt.figure(figsize=(25,20))
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})

    # plotting roc_auc with min_samples_split
    plt.figure()
    plt.plot(scores["param_max_depth"],scores["mean_train_score"], label="Training roc_auc")
    plt.plot(scores["param_max_depth"],scores["mean_test_score"], label="Test roc_auc")
    plt.xlabel("max_depth",fontsize=12,family = "Comic Sans MS")
    plt.ylabel("roc_auc", fontsize=12,family = "Comic Sans MS")
    plt.legend()
    plt.show()


# In[ ]:


def dt_min_samples_leaf(X_train, y_train):
    # parameters to build the model on
    parameters = {'min_samples_leaf': range(5, 200, 40)}

    # instantiate the model
    dtree = DecisionTreeClassifier(criterion = "gini", 
                                   random_state = 100)

    # fit tree on training data
    tree = GridSearchCV(dtree, parameters, 
                        cv=StratifiedKFold(5).split(X_train, y_train), 
                       scoring="roc_auc", return_train_score=True)
    tree.fit(X_train, y_train)
    
    scores = tree.cv_results_
    
    fig = plt.figure(figsize=(25,20))
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})

    # plotting roc_auc with min_samples_split
    plt.figure()
    plt.plot(scores["param_min_samples_leaf"],scores["mean_train_score"], label="Training roc_auc")
    plt.plot(scores["param_min_samples_leaf"],scores["mean_test_score"], label="Test roc_auc")
    plt.xlabel("min_samples_leaf",fontsize=12,family = "Comic Sans MS")
    plt.ylabel("roc_auc", fontsize=12,family = "Comic Sans MS")
    plt.legend()
    plt.show()


# In[ ]:


def dt_min_samples_split(X_train, y_train):
    # parameters to build the model on
    parameters = {'min_samples_split': range(5, 200, 40)}

    # instantiate the model
    dtree = DecisionTreeClassifier(criterion = "gini", 
                                   random_state = 100)

    # fit tree on training data
    tree = GridSearchCV(dtree, parameters, 
                        cv=StratifiedKFold(5).split(X_train, y_train), 
                       scoring="roc_auc", return_train_score=True)
    tree.fit(X_train, y_train)
    
    scores = tree.cv_results_
    
    fig = plt.figure(figsize=(25,20))
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})

    # plotting roc_auc with min_samples_split
    plt.figure()
    plt.plot(scores["param_min_samples_split"],scores["mean_train_score"], label="Training roc_auc")
    plt.plot(scores["param_min_samples_split"],scores["mean_test_score"], label="Test roc_auc")
    plt.xlabel("min_samples_split",fontsize=12,family = "Comic Sans MS")
    plt.ylabel("roc_auc", fontsize=12,family = "Comic Sans MS")
    plt.legend()
    plt.show()


# ### Decision Tree on imbalanced data set

# In[ ]:


dt_max_depth(X_train, y_train)


# In[ ]:


dt_min_samples_leaf(X_train, y_train)


# In[ ]:


dt_min_samples_split(X_train, y_train)


# In[ ]:


# Create the parameter grid 
param_grid = {
    'max_depth': [3,6],
    'min_samples_leaf': range(80, 100),
    'min_samples_split': range(50, 120),
    'criterion': ["entropy", "gini"]
}

# Instantiate the grid search model
dtree = DecisionTreeClassifier()
grid_search = GridSearchCV(estimator = dtree, param_grid = param_grid, 
                          cv=StratifiedKFold(5).split(X_train, y_train), 
                       scoring="roc_auc", return_train_score=True, verbose = 1)

# Fit the grid search to the data
grid_search.fit(X_train,y_train)


# In[ ]:


# printing the optimal score and hyperparameters
print("best score", grid_search.best_score_)
print(grid_search.best_estimator_)
print(grid_search.best_params_)


# In[55]:


# model with optimal hyperparameters
dt_imbal = DecisionTreeClassifier(criterion = "entropy", 
                                  random_state = 100,
                                  max_depth=3, 
                                  min_samples_leaf=80,
                                  min_samples_split=50)
dt_imbal.fit(X_train, y_train)


# In[56]:


# Evaluation of model performance on Train subset
threshold_train_val_dt = modelMetrics_train(dt_imbal, X_train, y_train)


# In[57]:


# Evaluation of model performance on Test subset
modelMetrics_test(dt_imbal, threshold_train_val_dt)


# ### Decision Tree on balanced data set using Random Oversampling

# In[58]:


# Running the Decision Tree with default parameters.
dt_bal_ros = DecisionTreeClassifier(max_depth=5)
dt_bal_ros.fit(X_train_ros,y_train_ros)


# In[59]:


threshold_train_val = modelMetrics_train(dt_bal_ros, X_train_ros, y_train_ros)


# In[60]:


# Evaluation of model performance on Test subset
modelMetrics_test(dt_bal_ros, threshold_train_val)


# In[79]:


dt_max_depth(X_train_ros, y_train_ros)


# In[ ]:


dt_min_samples_leaf(X_train_ros, y_train_ros)


# In[ ]:


dt_min_samples_split(X_train_ros, y_train_ros)


# In[ ]:


# Create the parameter grid 
param_grid = {
    'max_depth': [3,6],
    'min_samples_leaf': range(40, 120, 40),
    'min_samples_split': range(50, 100, 50),
    'criterion': ["entropy", "gini"]
}

# Instantiate the grid search model
dtree = DecisionTreeClassifier()
grid_search = GridSearchCV(estimator = dtree, param_grid = param_grid, 
                          cv=StratifiedKFold(5).split(X_train_ros, y_train_ros), 
                       scoring="roc_auc", return_train_score=True, verbose = 1)

# Fit the grid search to the data
grid_search.fit(X_train_ros,y_train_ros)


# In[ ]:


# printing the optimal score and hyperparameters
print("Best score", grid_search.best_score_)
print(grid_search.best_estimator_)
print(grid_search.best_params_)


# In[ ]:


# model with optimal hyperparameters
clf_gini_ros = DecisionTreeClassifier(criterion = "entropy", 
                                  random_state = 100,
                                  max_depth=6, 
                                  min_samples_leaf=40,
                                  min_samples_split=50)
clf_gini_ros.fit(X_train_ros, y_train_ros)


# In[ ]:


threshold_train_val = modelMetrics_train(clf_gini_ros, X_train_ros, y_train_ros)


# In[ ]:


# Evaluation of model performance on Test subset
modelMetrics_test(clf_gini_ros, threshold_train_val)


# In[ ]:





# In[ ]:





# ### Decision Tree on balanced data set using SMOTE

# In[ ]:


# Running the Decision Tree with default parameters.
dt_bal_sm = DecisionTreeClassifier(max_depth=5)
dt_bal_sm.fit(X_train_sm,y_train_sm)


# In[ ]:


# Evaluation of model performance on Train subset
threshold_train_val = modelMetrics_train(dt_bal_sm, X_train_sm, y_train_sm)


# In[ ]:


# Evaluation of model performance on Test subset
modelMetrics_test(dt_bal_sm, threshold_train_val)


# In[ ]:


dt_max_depth(X_train_sm, y_train_sm)


# In[ ]:


dt_min_samples_leaf(X_train_sm, y_train_sm)


# In[ ]:


dt_min_samples_split(X_train_sm, y_train_sm)


# In[ ]:


# Create the parameter grid 
param_grid = {
    'max_depth': [5,7],
    'min_samples_leaf': range(40, 100, 20),
    'min_samples_split': range(50, 120, 40),
    'criterion': ["entropy", "gini"]
}
# Instantiate the grid search model
dtree = DecisionTreeClassifier()
grid_search = GridSearchCV(estimator = dtree, param_grid = param_grid, 
                          cv=StratifiedKFold(5).split(X_train_sm, y_train_sm), 
                       scoring="roc_auc", return_train_score=True, verbose = 1)

# Fit the grid search to the data
grid_search.fit(X_train_sm,y_train_sm)


# In[ ]:


# printing the optimal score and hyperparameters
print("Best score", grid_search.best_score_)
print(grid_search.best_estimator_)
print(grid_search.best_params_)


# In[ ]:


# model with optimal hyperparameters
clf_gini_smote = DecisionTreeClassifier(criterion = "entropy", 
                                  random_state = 100,
                                  max_depth=7, 
                                  min_samples_leaf=40,
                                  min_samples_split=90)
clf_gini_smote.fit(X_train_sm, y_train_sm)


# In[ ]:


# Evaluation of model performance on Train subset
threshold_train_val = modelMetrics_train(clf_gini_smote, X_train_sm, y_train_sm)


# In[ ]:


# Evaluation of model performance on Test subset
modelMetrics_test(clf_gini_smote, threshold_train_val)


# In[ ]:





# In[ ]:





# ### Decision Tree on balanced data set using ADASYN 

# In[ ]:


dt_max_depth(X_train_adasyn, y_train_adasyn)


# In[ ]:


dt_min_samples_leaf(X_train_adasyn, y_train_adasyn)


# In[ ]:


dt_min_samples_split(X_train_adasyn, y_train_adasyn)


# In[ ]:


# Create the parameter grid 
param_grid = {
    'max_depth': [3,6],
    'min_samples_leaf': range(40, 120, 40),
    'min_samples_split': range(50, 100, 50),
    'criterion': ["entropy", "gini"]
}

# Instantiate the grid search model
dtree = DecisionTreeClassifier()
grid_search = GridSearchCV(estimator = dtree, param_grid = param_grid, 
                          cv=StratifiedKFold(5).split(X_train_adasyn, y_train_adasyn), 
                       scoring="roc_auc", return_train_score=True, verbose = 1)

# Fit the grid search to the data
grid_search.fit(X_train_adasyn,y_train_adasyn)


# In[ ]:


# printing the optimal score and hyperparameters
print("best score", grid_search.best_score_)
print(grid_search.best_estimator_)
print(grid_search.best_params_)


# In[ ]:


# model with optimal hyperparameters
clf_gini_adasyn = DecisionTreeClassifier(criterion = "entropy", 
                                  random_state = 100,
                                  max_depth=6, 
                                  min_samples_leaf= 40,
                                  min_samples_split=50)
clf_gini_adasyn.fit(X_train_adasyn, y_train_adasyn)


# In[ ]:


# Evaluation of model performance on Train subset
threshold_train_val = modelMetrics_train(clf_gini_adasyn, X_train_adasyn, y_train_adasyn)


# In[ ]:


# Evaluation of model performance on Test subset
modelMetrics_test(clf_gini_adasyn, threshold_train_val)


# In[ ]:





# ### Decision Tree on balanced data set using Borderline SMOTE

# In[ ]:


dt_max_depth(X_train_blsm, y_train_blsm)


# In[ ]:


dt_min_samples_leaf(X_train_blsm, y_train_blsm)


# In[ ]:


dt_min_samples_split(X_train_blsm, y_train_blsm)


# In[ ]:


# printing the optimal score and hyperparameters
print("best score", grid_search.best_score_)
print(grid_search.best_estimator_)
print(grid_search.best_params_)


# In[ ]:


# model with optimal hyperparameters
clf_gini_blsm = DecisionTreeClassifier(criterion = "entropy", 
                                  random_state = 100,
                                  max_depth=6, 
                                  min_samples_leaf=70,
                                  min_samples_split=100)
clf_gini_blsm.fit(X_train_blsm, y_train_blsm)


# In[ ]:


threshold_train_val_blsm = modelMetrics_train(clf_gini_blsm, X_train_blsm, y_train_blsm)


# In[ ]:


# Evaluation of model performance on Test subset
modelMetrics_test(clf_gini_blsm, threshold_train_val_blsm)


# In[ ]:





# In[ ]:





# | Model                | Train Dataset       | Hyperparameters              | Train AUC   | Threshold | Test AUC | 
# | :-------------       | :-------------      | :----------:                 | -----------: |-----------: |-----------: |
# |  Decision Tree |  Imbalanced         | {'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 80, 'min_samples_split': 50} | 94.89%  | 0.002 | 93.13%   |
# |  Decision Tree |  Random Oversampler | {'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 40, 'min_samples_split': 50}  | 99.74%   | 0.564 | 87.91%   |
# |  Decision Tree |  SMOTE              | {'criterion': 'entropy', 'max_depth': 7, 'min_samples_leaf': 40, 'min_samples_split': 90}  | 99.55%   | 0.547 | 92.06%   |
# |  Decision Tree |  Border line SMOTE  | {'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 70, 'min_samples_split': 100}  | 99.88%   | 0.500 | 93.34%   |
# |  Decision Tree |  ADASYN             | {'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 40, 'min_samples_split': 50}| 97.77%   | 0.560 | 92.55%   |

# In[ ]:





# ### XG Boost Model on imbalanced data set with tuned Hyperparameter Paramaters

# In[52]:


# fit model on training data with default hyperparameters
xgb_dt = XGBClassifier(random_state=0)
xgb_dt.fit(X_train,y_train)


# In[53]:


# Evaluation of model performance on Train subset
threshold_train_val_xgb = modelMetrics_train(xgb_dt, X_train, y_train)


# In[54]:


# Evaluation of model performance on Test subset
modelMetrics_test(xgb_dt, threshold_train_val_xgb)


# XG Boost Model - Hyperparameter Tuning

# Determining n_estimators

# In[119]:


tree_range = range(2, 75, 5)
score1=[]
score2=[]
for tree in tree_range:
    xgb=XGBClassifier(n_estimators=tree)
    xgb.fit(X_train,y_train)
    score1.append(xgb.score(X_train,y_train))
    score2.append(xgb.score(X_test,y_test))
    
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(8,4))
plt.plot(tree_range,score1,label= 'Accuracy on training set')
plt.plot(tree_range,score2,label= 'Accuracy on testing set')
plt.xlabel('Value of number of trees in XGboost')
plt.ylabel('Accuracy')
plt.legend()


# In[55]:


# hyperparameter tuning with XGBoost
# creating a KFold object 
# folds = 3
kfolds = StratifiedKFold(5)

# specify range of hyperparameters
param_grid = {'learning_rate': [0.2, 0.6], 
             'subsample': [0.3, 0.6, 0.9]}          

# specify model
xgb_model = XGBClassifier(max_depth=3, n_estimators=27)

# set up GridSearchCV()
model_cv = GridSearchCV(estimator = xgb_model, 
                        param_grid = param_grid, 
                        scoring= 'roc_auc', 
                        cv = kfolds.split(X_train, y_train),
                        verbose = 1,
                        return_train_score=True)


# In[56]:


# fit the model
model_cv.fit(X_train, y_train)       


# In[122]:


# printing the optimal score and hyperparameters
print('Best score ',model_cv.best_score_,'using',model_cv.best_params_)


# In[57]:


# cv results
cv_results = pd.DataFrame(model_cv.cv_results_)


# In[58]:


# convert parameters to int for plotting on x-axis
cv_results['param_learning_rate'] = cv_results['param_learning_rate'].astype('float')
cv_results['param_subsample'] = cv_results['param_subsample'].astype('float')


# In[ ]:


# # plotting
plt.figure(figsize=(16,6))

for n, subsample in enumerate(param_grid['subsample']):
    # subplot 1/n
    plt.subplot(1,len(param_grid['subsample']), n+1)
    df = cv_results[cv_results['param_subsample']==subsample]

    plt.plot(df["param_learning_rate"], df["mean_test_score"])
    plt.plot(df["param_learning_rate"], df["mean_train_score"])
    plt.xlabel('learning_rate')
    plt.ylabel('AUC')
    plt.title("subsample={0}".format(subsample))
    plt.ylim([0.60, 1])
    plt.legend(['test score', 'train score'], loc='upper left')
    plt.xscale('log')


# In[59]:


# chosen hyperparameters
params = {'learning_rate': 0.6,
          'max_depth': 3, 
          'n_estimators':27,
          'subsample':0.9,
         'objective':'binary:logistic'}

# fit model on training data
xgb = XGBClassifier(params = params)
xgb.fit(X_train, y_train)


# In[60]:


# Evaluation of model performance on Train subset
threshold_train_val_xgb_h = modelMetrics_train(xgb, X_train, y_train)


# In[61]:


# Evaluation of model performance on Test subset
modelMetrics_test(xgb, threshold_train_val_xgb_h)


# In[62]:


# feature importance
importance = dict(zip(X_train.columns, xgb.feature_importances_))


# In[63]:


# plot
plt.bar(range(len(xgb.feature_importances_)), xgb.feature_importances_)
plt.show()


# In[ ]:





# ### Utility functions

# In[92]:


def xgb_n_estimators(X_train, y_train, X_test, y_test):
    tree_range = range(2, 150, 15)
    score1=[]
    score2=[]
    for tree in tree_range:
        xgb=XGBClassifier(n_estimators=tree)
        xgb.fit(X_train,y_train)
        score1.append(xgb.score(X_train,y_train))
        score2.append(xgb.score(X_test,y_test))

    get_ipython().run_line_magic('matplotlib', 'inline')
    plt.figure(figsize=(8,4))
    plt.plot(tree_range,score1,label= 'Accuracy on training set')
    plt.plot(tree_range,score2,label= 'Accuracy on testing set')
    plt.xlabel('Value of number of trees in XGboost')
    plt.ylabel('Accuracy')
    plt.legend()


# ### XGboost Model on balanced data set using Borderline Random Oversampling
# 

# In[52]:


xgb_n_estimators(X_train_ros, y_train_ros, X_test, y_test)


# In[64]:


# hyperparameter tuning with XGBoost
# creating a KFold object 
# folds = 3
kfolds = StratifiedKFold(5)

# specify range of hyperparameters
param_grid = {'learning_rate': [0.2, 0.6], 
             'subsample': [0.3, 0.6, 0.9]}          

# specify model
xgb_model = XGBClassifier(max_depth=3, n_estimators=24)

# set up GridSearchCV()
model_cv = GridSearchCV(estimator = xgb_model, 
                        param_grid = param_grid, 
                        scoring= 'roc_auc', 
                        cv = kfolds.split(X_train_ros, y_train_ros),
                        verbose = 1,
                        return_train_score=True)


# In[65]:


# fit the model
model_cv.fit(X_train_ros, y_train_ros) 


# In[66]:


# printing the optimal  score and hyperparameters
print('Best score ',model_cv.best_score_,'using',model_cv.best_params_)


# In[67]:


# cv results
cv_results = pd.DataFrame(model_cv.cv_results_)


# In[68]:


# convert parameters to int for plotting on x-axis
cv_results['param_learning_rate'] = cv_results['param_learning_rate'].astype('float')
cv_results['param_subsample'] = cv_results['param_subsample'].astype('float')


# In[69]:


# # plotting
plt.figure(figsize=(16,6))

for n, subsample in enumerate(param_grid['subsample']):
    # subplot 1/n
    plt.subplot(1,len(param_grid['subsample']), n+1)
    df = cv_results[cv_results['param_subsample']==subsample]

    plt.plot(df["param_learning_rate"], df["mean_test_score"])
    plt.plot(df["param_learning_rate"], df["mean_train_score"])
    plt.xlabel('learning_rate')
    plt.ylabel('AUC')
    plt.title("subsample={0}".format(subsample))
    plt.ylim([0.60, 1])
    plt.legend(['test score', 'train score'], loc='upper left')
    plt.xscale('log')


# In[70]:


# chosen hyperparameters
# 'objective':'binary:logistic' outputs probability rather than label, which we need for auc
params = {'learning_rate': 0.6,
          'max_depth': 3, 
          'n_estimators':27,
          'subsample':0.6,
         'objective':'binary:logistic'}

# fit model on training data
xgb_model_ros = XGBClassifier(params = params)
xgb_model_ros.fit(X_train_ros, y_train_ros)


# In[71]:


# Evaluation of model performance on Train subset
threshold_train_val = modelMetrics_train(xgb_model_ros, X_train_ros, y_train_ros)


# In[72]:


# Evaluation of model performance on Test subset
modelMetrics_test(xgb_model_ros, threshold_train_val)


# In[ ]:





# ### XGboost on balanced data set using SMOTE

# In[73]:


# fit model on training data with default hyperparameters
xgb_smote=XGBClassifier(random_state=0)
xgb_smote.fit(X_train_sm,y_train_sm)


# In[74]:


# Evaluation of model performance on Train subset
threshold_train_val = modelMetrics_train(xgb_smote,X_train_sm, y_train_sm)


# In[ ]:


xgb_n_estimators(X_train_sm, y_train_sm, X_test, y_test)


# In[75]:


# hyperparameter tuning with XGBoost
# creating a KFold object 
# folds = 3
kfolds = StratifiedKFold(5)

# specify range of hyperparameters
param_grid = {'learning_rate': [0.2, 0.6], 
             'subsample': [0.3, 0.6, 0.9]}          

# specify model
xgb_model = XGBClassifier(max_depth=3, n_estimators=27)

# set up GridSearchCV()
model_cv = GridSearchCV(estimator = xgb_model, 
                        param_grid = param_grid, 
                        scoring= 'roc_auc', 
                        cv = kfolds.split(X_train_sm, y_train_sm),
                        verbose = 1,
                        return_train_score=True)


# In[76]:


# fit the model
model_cv.fit(X_train_sm, y_train_sm) 


# In[77]:


# printing the optimal  score and hyperparameters
print('Best score ',model_cv.best_score_,'using',model_cv.best_params_)


# In[78]:


# cv results
cv_results = pd.DataFrame(model_cv.cv_results_)


# In[79]:


# convert parameters to int for plotting on x-axis
cv_results['param_learning_rate'] = cv_results['param_learning_rate'].astype('float')
cv_results['param_subsample'] = cv_results['param_subsample'].astype('float')


# In[80]:


# # plotting
plt.figure(figsize=(16,6))

for n, subsample in enumerate(param_grid['subsample']):
    # subplot 1/n
    plt.subplot(1,len(param_grid['subsample']), n+1)
    df = cv_results[cv_results['param_subsample']==subsample]

    plt.plot(df["param_learning_rate"], df["mean_test_score"])
    plt.plot(df["param_learning_rate"], df["mean_train_score"])
    plt.xlabel('learning_rate')
    plt.ylabel('AUC')
    plt.title("subsample={0}".format(subsample))
    plt.ylim([0.60, 1])
    plt.legend(['test score', 'train score'], loc='upper left')
    plt.xscale('log')


# In[81]:


# chosen hyperparameters
# 'objective':'binary:logistic' outputs probability rather than label, which we need for auc
params = {'learning_rate': 0.6,
          'max_depth': 3, 
          'n_estimators':27,
          'subsample':0.9,
         'objective':'binary:logistic'}

# fit model on training data
xgb_model_smote = XGBClassifier(params = params)
xgb_model_smote.fit(X_train_sm, y_train_sm)


# In[82]:


# Evaluation of model performance on Train subset
threshold_train_val_xgb = modelMetrics_train(xgb_model_smote, X_train_sm, y_train_sm)


# In[83]:


# Evaluation of model performance on Test subset
modelMetrics_test(xgb_model_smote, threshold_train_val_xgb)


# In[ ]:





# In[ ]:





# ### Xgboost on balanced data set using Borderline SMOTE

# In[84]:


# hyperparameter tuning with XGBoost
# creating a KFold object 
# folds = 3
kfolds = StratifiedKFold(5)

# specify range of hyperparameters
param_grid = {'learning_rate': [0.2, 0.6], 
             'subsample': [0.3, 0.6, 0.9]}          

# specify model
xgb_model = XGBClassifier(max_depth=3, n_estimators=27)

# set up GridSearchCV()
model_cv = GridSearchCV(estimator = xgb_model, 
                        param_grid = param_grid, 
                        scoring= 'roc_auc', 
                        cv = kfolds.split(X_train_blsm, y_train_blsm),
                        verbose = 1,
                        return_train_score=True)


# In[85]:


# fit the model
model_cv.fit(X_train_blsm, y_train_blsm) 


# In[86]:


# printing the optimal score and hyperparameters
print('Best score ',model_cv.best_score_,'using',model_cv.best_params_)


# In[87]:


# cv results
cv_results = pd.DataFrame(model_cv.cv_results_)


# In[88]:


# convert parameters to int for plotting on x-axis
cv_results['param_learning_rate'] = cv_results['param_learning_rate'].astype('float')
cv_results['param_subsample'] = cv_results['param_subsample'].astype('float')


# In[ ]:


# # plotting
plt.figure(figsize=(16,6))

for n, subsample in enumerate(param_grid['subsample']):
    # subplot 1/n
    plt.subplot(1,len(param_grid['subsample']), n+1)
    df = cv_results[cv_results['param_subsample']==subsample]

    plt.plot(df["param_learning_rate"], df["mean_test_score"])
    plt.plot(df["param_learning_rate"], df["mean_train_score"])
    plt.xlabel('learning_rate')
    plt.ylabel('AUC')
    plt.title("subsample={0}".format(subsample))
    plt.ylim([0.60, 1])
    plt.legend(['test score', 'train score'], loc='upper left')
    plt.xscale('log')


# In[89]:


# chosen hyperparameters
# 'objective':'binary:logistic' outputs probability rather than label, which we need for auc
params = {'learning_rate': 0.6,
          'max_depth': 3, 
          'n_estimators':27,
          'subsample':0.6,
         'objective':'binary:logistic'}

# fit model on training data
xgb = XGBClassifier(params = params)
xgb.fit(X_train_blsm, y_train_blsm)


# In[90]:


# Evaluation of model performance on Train subset
threshold_train_val = modelMetrics_train(xgb, X_train_blsm, y_train_blsm)


# In[91]:


# Evaluation of model performance on Test subset
modelMetrics_test(xgb, threshold_train_val)


# In[ ]:





# ### XGboost Model on balanced data set using Borderline ADASYN 

# In[ ]:


# hyperparameter tuning with XGBoost
# creating a KFold object 
kfolds = StratifiedKFold(5)

# specify range of hyperparameters
param_grid = {'learning_rate': [0.2, 0.6], 
             'subsample': [0.3, 0.6, 0.9]}          

# specify model
xgb_model = XGBClassifier(max_depth=3, n_estimators=27)

# set up GridSearchCV()
model_cv = GridSearchCV(estimator = xgb_model, 
                        param_grid = param_grid, 
                        scoring= 'roc_auc', 
                        cv = kfolds.split(X_train_adasyn, y_train_adasyn),
                        verbose = 1,
                        return_train_score=True)


# In[ ]:


# fit the model
model_cv.fit(X_train_adasyn, y_train_adasyn)


# In[ ]:


# printing the optimal  score and hyperparameters
print('Best score ',model_cv.best_score_,'using',model_cv.best_params_)


# In[ ]:


# cv results
cv_results = pd.DataFrame(model_cv.cv_results_)


# In[ ]:


# convert parameters to int for plotting on x-axis
cv_results['param_learning_rate'] = cv_results['param_learning_rate'].astype('float')
cv_results['param_subsample'] = cv_results['param_subsample'].astype('float')


# In[ ]:


# # plotting
plt.figure(figsize=(16,6))

for n, subsample in enumerate(param_grid['subsample']):
    # subplot 1/n
    plt.subplot(1,len(param_grid['subsample']), n+1)
    df = cv_results[cv_results['param_subsample']==subsample]

    plt.plot(df["param_learning_rate"], df["mean_test_score"])
    plt.plot(df["param_learning_rate"], df["mean_train_score"])
    plt.xlabel('learning_rate')
    plt.ylabel('AUC')
    plt.title("subsample={0}".format(subsample))
    plt.ylim([0.60, 1])
    plt.legend(['test score', 'train score'], loc='upper left')
    plt.xscale('log')


# In[ ]:


# chosen hyperparameters
# 'objective':'binary:logistic' outputs probability rather than label, which we need for auc
params = {'learning_rate': 0.6,
          'max_depth': 3, 
          'n_estimators':27,
          'subsample':0.9,
         'objective':'binary:logistic'}

# fit model on training data
xgb_model_adasyn = XGBClassifier(params = params)
xgb_model_adasyn.fit(X_train_adasyn, y_train_adasyn)


# In[ ]:


# Evaluation of model performance on Train subset
threshold_train_val = modelMetrics_train(xgb_model_adasyn, X_train_adasyn, y_train_adasyn)


# In[ ]:


# Evaluation of model performance on Test subset
modelMetrics_test(xgb_model_adasyn, threshold_train_val)


# In[ ]:


X_train_blsm, y_train_blsm = borderlinesmote.fit_sample(X_train,y_train)


# In[ ]:





# In[ ]:





# | Model                | Train Dataset       | Hyperparameters              | Train AUC  | Threshold | Test AUC
# | :-------------       | :-------------      | :----------:                 | -----------: | -----------: | -----------: |
# |  XG Boost |  Imbalanced         | {'max_depth':3,'n_estimators':27, 'learning_rate': 0.6, 'subsample': 0.9}  | 100.00%   | 0.872 |89.85%   |
# |  XG Boost |  Random Oversampler | {'max_depth':3,'n_estimators':27, 'learning_rate': 0.6, 'subsample': 0.6}  | 100.00%   | 1.000 |88.84%   |
# |  XG Boost |  SMOTE              | {'max_depth':3,'n_estimators':27, 'learning_rate': 0.6, 'subsample': 0.9}   | 100.00%   | 0.875 |90.19%  |
# |  XG Boost |  Border line SMOTE  | {'max_depth':3,'n_estimators':27, 'learning_rate': 0.6, 'subsample': 0.6}   | 100.00%   | 0.758 |91.87%  |
# |  XG Boost |  ADASYN             | {'max_depth':3,'n_estimators':27, 'learning_rate': 0.6, 'subsample': 0.3} | 100.00%   | 0.910 |90.19%   |

# In[ ]:





# In[ ]:





# ### Summary

# | Model                | Train Dataset       | Hyperparameters              | Train AUC   | Threshold | Test AUC |
# | :-------------       | :-------------      | :----------:                 | -----------: | -----------: | -----------: |
# |  Logistic Regression |  Imbalanced         | Default                      | 97.89%   | 0.003 | 93.78% |
# |  Logistic Regression |  Imbalanced         | {'C': 0.01, 'penalty': 'l2'} | 98.12%   | 0.007 | 93.87% |
# |  Logistic Regression |  Random Oversampler | {'C': 100, 'penalty': 'l2'}  | 98.54%   | 0.471 | 93.98% |
# |  Logistic Regression |  SMOTE              | {'C': 100, 'penalty': 'l2'}  | 98.83%   | 0.454 | 93.86% |
# |  Logistic Regression |  Border line SMOTE  | {'C': 10, 'penalty': 'l2'}  | 99.90%   | 0.417 | 94.12% |
# |  Logistic Regression |  ADASYN             | {'C': 100, 'penalty': 'l2'}  | 96.23%   | 0.325 | 92.65% |
# | K-Nearest Neighbours |  Imbalanced         | Default  | 99.9%   | 0.200 | 88.84% |
# |  Decision Tree |  Imbalanced         | {'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 80, 'min_samples_split': 50} | 94.89%  | 0.002 | 93.13%   |
# |  Decision Tree |  Random Oversampler | {'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 40, 'min_samples_split': 50}  | 99.74%   | 0.564 | 87.91%   |
# |  Decision Tree |  SMOTE              | {'criterion': 'entropy', 'max_depth': 7, 'min_samples_leaf': 40, 'min_samples_split': 90}  | 99.55%   | 0.547 | 92.06%   |
# |  Decision Tree |  Border line SMOTE  | {'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 70, 'min_samples_split': 100}  | 99.88%   | 0.500 | 93.34%   |
# |  Decision Tree |  ADASYN             | {'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 40, 'min_samples_split': 50}| 97.77%   | 0.560 | 92.55%   |
# |  Random Forest |  Imbalanced         | Default                      | 100%   | 0.530 | 90.87%   | 
# |  Random Forest |  Imbalanced         | {'max_depth': 7, 'max_features': 8, 'min_samples_leaf': 150, 'min_samples_split': 350, 'n_estimators': 150}  | 98.89%   | 0.002 | 91.19% |
# |  Random Forest |  Random Oversampler | {'max_depth': 7, 'max_features': 8, 'min_samples_leaf': 300, 'min_samples_split': 400, 'n_estimators': 200}  | 99.99%   | 0.612 | 93.52% |
# |  Random Forest |  SMOTE              | {'max_depth': 7, 'max_features': 8, 'min_samples_leaf': 100, 'min_samples_split': 200, 'n_estimators': 200}  | 99.67%   | 0.374 | 93.96% |
# |  Random Forest | Border line SMOTE   | {'max_depth': 7, 'max_features': 8, 'min_samples_leaf': 200, 'min_samples_split': 300, 'n_estimators': 100}  | 99.98%   | 0.543 | 93,93% |
# |  Random Forest |  ADASYN             | {'max_depth': 7, 'max_features': 8, 'min_samples_leaf': 150, 'min_samples_split': 300, 'n_estimators': 200} | 99.59%   | 0.462 | 94.86% |
# |  XG Boost |  Imbalanced         | {'max_depth':3,'n_estimators':27, 'learning_rate': 0.6, 'subsample': 0.9}  | 100.00%   | 0.872 |89.85%   |
# |  XG Boost |  Random Oversampler | {'max_depth':3,'n_estimators':27, 'learning_rate': 0.6, 'subsample': 0.6}  | 100.00%   | 1.000 |88.84%   |
# |  XG Boost |  SMOTE              | {'max_depth':3,'n_estimators':27, 'learning_rate': 0.6, 'subsample': 0.9}   | 100.00%   | 0.875 |90.19%  |
# |  XG Boost |  Border line SMOTE  | {'max_depth':3,'n_estimators':27, 'learning_rate': 0.6, 'subsample': 0.6}   | 100.00%   | 0.758 |91.87%  |
# |  XG Boost |  ADASYN             | {'max_depth':3,'n_estimators':27, 'learning_rate': 0.6, 'subsample': 0.3} | 100.00%   | 0.910 |90.19%   |

# In[ ]:





# In[ ]:





# ### Important features of the best model to understand the dataset

# In[ ]:


var_imp = []
for i in rfc_adasyn.feature_importances_:
    var_imp.append(i)
print('Top var =', var_imp.index(np.sort(rfc_adasyn.feature_importances_)[-1])+1)
print('2nd Top var =', var_imp.index(np.sort(rfc_adasyn.feature_importances_)[-2])+1)
print('3rd Top var =', var_imp.index(np.sort(rfc_adasyn.feature_importances_)[-3])+1)

top_var_index = var_imp.index(np.sort(rfc_adasyn.feature_importances_)[-1])
second_top_var_index = var_imp.index(np.sort(rfc_adasyn.feature_importances_)[-2])

X_train_1 = X_train.to_numpy()[np.where(y_train==1.0)]
X_train_0 = X_train.to_numpy()[np.where(y_train==0.0)]

np.random.shuffle(X_train_0)

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = [16, 16]

plt.scatter(X_train_1[:, top_var_index], X_train_1[:, second_top_var_index], label='Actual Class-1 Examples')
plt.scatter(X_train_0[:X_train_1.shape[0], top_var_index], X_train_0[:X_train_1.shape[0], second_top_var_index],
            label='Actual Class-0 Examples')
plt.legend()


# In[ ]:





# ### Cost Benefit Analysis

# In[85]:


# predictions on test subset
y_pred = rfc_adasyn.predict(X_test)


# In[86]:


# computing confusion matrix
cm=confusion_matrix(y_test, y_pred)
TP = cm[1,1] # true positive 
TN = cm[0,0] # true negatives
FP = cm[0,1] # false positives
FN = cm[1,0] # false negatives
print(cm)


# In[87]:


# create predictions dataframe. Column 1 is Class --> Actual
test_predictions = pd.DataFrame(y_test)
test_predictions.rename(columns={'Class':'Actual'}, inplace=True)
test_predictions.head()


# In[88]:


# add predicted columns with prediction values
test_predictions['Predicted'] = y_pred
test_predictions.head()


# In[89]:


# merge with transaction amount from the original dataframe by merging based on index values
test_predictions = test_predictions.merge(pd.DataFrame(credit_card_data_org['Amount']), left_index=True, right_index=True)
test_predictions.head()


# In[90]:


# Computing transaction cost for all True Positive cases
trnx_cost_TP = test_predictions.query('Actual==1 & Predicted==1')['Amount'].sum()
trnx_cost_TP


# In[91]:


# Computing transaction cost for all False Negative cases
trnx_cost_FN = test_predictions.query('Actual==1 & Predicted==0')['Amount'].sum()
trnx_cost_FN


# In[94]:


# Assuming 1 units as the cost of each verification call
call_verification_cost = (TP + FP) * 1
call_verification_cost


# In[95]:


total_savings = trnx_cost_TP - (trnx_cost_FN + call_verification_cost)
print("Total Savings (Cost Benefit Analysis) is ", round(total_savings,2), "amount units on Test Subset")


# In[ ]:




