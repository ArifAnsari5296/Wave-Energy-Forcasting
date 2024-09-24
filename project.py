#!/usr/bin/env python
# coding: utf-8

# # Wave Energy Forecasting: Machine Learning for Optimal Renewable Energy Utilization.
# 
# 
# 

# ### The project aims to leverage machine learning techniques for predicting wave energy output, contributing to the optimization of renewable energy Utilization.

# #### Include the header file

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import os
import datetime
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Read the csv file

# In[2]:


data = pd.read_csv('Coastal Data System.csv')


# In[3]:


df =data
df


# ### How to analyze the data
# 
# 
# 

# 
# 
# #### .head()
# 
# * it shows the first N rows in the data (by default , N = 5)

# In[4]:


df.head()


#  
#  
#  #### .shape
#  
#  * It shows the total no.of rows and columns of the dataframe

# In[5]:


df.shape


# 
# 
# #### .index
# 
# * The attribute provides the index of the dataframe
# 

# In[6]:


df.index


# 
# 
# #### .columns
# 
# * It shows the name of each column

# In[7]:


df.columns


# 
# 
# #### .dtype 
# 
# * It shows the datatype of each column.

# In[8]:


df.dtypes


# 
# 
# #### .unique()
# 
# * In a column .It shows all the unique values .It can be applied on a single column only , not on the whole dataframe

# In[9]:


df['SST'].unique()


# In[10]:


df['Tp'].unique()


# 
# 
# 
# #### nunique()
# 
# * Its shows the total no.of unique values in each column .It can be applied on a single column as well as on whole dataframe

# In[11]:


df.nunique()


# 
# 
# #### .count 
# 
# * It shows the total no .of non -null in each column .It can be applied on a single column as well as in whole columns

# In[12]:


df.count


# 
# 
# #### .value_counts 
# 
# * In a column ,It shows all the unique valus with their count . It can be applied on single column only

# In[13]:


df['Tp'].value_counts()


# In[14]:


df = df.drop(df.index[[0, 1]])
df = df.reset_index(drop=True)


# 
# 
# #### .info()
# 
# * Provides basic information about the dataframe

# In[15]:


df.info()


# 
# 
# #### .describe ()
# 
# * Returns description of the data in the Dataframe

# In[16]:


df


# 
# 
# #### Find the maximum   Power 
# 
# * Make the separation to make singular 

# #### Make a copy of database
# 

# In[17]:


df.describe()


# In[18]:


df = df.set_index('Date/Time')


# In[19]:


df.index = pd.to_datetime(df.index)


# In[20]:


color_pal = sns.color_palette()


# In[21]:


df.plot(style='.', figsize=(15, 5), color=color_pal[0])
plt.show()


# In[22]:


#Train Test Split
train = df.loc[df.index <'01-01-2019']
test = df.loc[df.index >= '01-01-2019']


# In[23]:


fig, ax = plt.subplots(figsize=(15, 5))
train.plot(ax=ax, label='Training set')
test.plot(ax=ax, label='Test set')
plt.show()


# In[24]:


df


# In[25]:


df.plot(subplots = True ,figsize=(12,12));


# In[26]:


### lets Assume that TP is Linear Related with peak Direction


# In[27]:


sns.heatmap(df.corr())


# In[28]:


output_col = 'Tp'
df.corr()[output_col].sort_values(ascending = False)[1:].plot(kind = 'bar',figsize =(12,4),grid = True)


# In[29]:


#output_col1 = 'Peak Direction'
#df.corr()[output_col1].sort_values(ascending = False)[1:].plot(kind = 'bar',figsize =(12,4),grid = True)


# In[30]:


corr_df = pd.DataFrame(df.corr()[output_col].sort_values(ascending =False)[1:10])
corr_df.columns = ['Corr']


# In[31]:


corr_df


# In[32]:


input_columns = corr_df[(abs(corr_df['Corr'])>0.2)&(corr_df['Corr']<0.90)].index


# In[33]:


input_columns


# In[34]:


X = df[input_columns]; y = df [output_col]


# In[35]:


### y = f(X) + e Assuming linear
df.info()


# In[36]:


df.describe()


# In[37]:


train_length = 39000 #days
X_train , y_train = X.iloc[:train_length,:] , y[:train_length]
X_test , y_test = X.iloc[train_length:,:] , y[train_length:]


# In[38]:


from sklearn.preprocessing import MinMaxScaler , StandardScaler


# In[39]:


# scaler = MinMaxScaler()  # Use this if you want to use MinMaxScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[40]:


from sklearn.linear_model import LinearRegression

#from sklearn.ensemble import RandomForestRegression

lm = LinearRegression()


# In[41]:


lm.fit(X_train_scaled , y_train)


# In[42]:


yp_train = lm.predict(X_train_scaled)


# In[43]:


plt.figure(figsize=(12, 4))
plt.scatter(df.index[:train_length], y_train, color='orange', label='Scatter plot')
plt.plot(df.index[:train_length], y_train, label='Line plot')

plt.grid()
plt.legend()
plt.show()


# In[44]:


yp_test = lm.predict(X_test_scaled)


# In[45]:


plt.figure(figsize = (12,4))

plt.scatter(df.index[:train_length], y_train, color='orange', label='Scatter plot')
plt.plot(df.index[:train_length] , yp_train)

plt.scatter(df.index[train_length:] , y_test , color = 'green')
plt.plot(df.index[train_length:], yp_test , color ='black')

plt.axvline(df.index[train_length] , color = 'red')

plt.grid()
plt.legend()
plt.show()


# In[46]:


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()

rf.fit(X_train, y_train)

yp_train = rf.predict(X_train)
yp_test = rf.predict(X_test)


# In[47]:


plt.figure(figsize = (12,4))

plt.scatter(df.index[:train_length], y_train, color='orange', label='Scatter plot')
plt.plot(df.index[:train_length] , yp_train)

plt.scatter(df.index[train_length:] , y_test , color = 'green')
plt.plot(df.index[train_length:], yp_test , color ='black')

plt.axvline(df.index[train_length] , color = 'red')

plt.grid()
plt.legend()
plt.show()


# In[48]:


import matplotlib.pyplot as plt

# Assuming df.index is a datetime index
plt.figure(figsize=(12, 4))

# Scatter plot for training data
plt.scatter(df.index[:train_length], y_train, color='orange', label='Training Data')

# Scatter plot for testing data
plt.scatter(df.index[train_length:], y_test, color='green', label='Testing Data')

# Plot predictions
plt.plot(df.index[:train_length], yp_train, color='blue', label='Predictions (Training)')
plt.plot(df.index[train_length:], yp_test, color='black', label='Predictions (Testing)')

# Add a vertical line to indicate the split between training and testing data
plt.axvline(df.index[train_length], color='red', linestyle='--', label='Train-Test Split')

plt.grid()
plt.legend()
plt.show()


# ### What R-squared does : Compares your models predictions to the mean of the targets .Values can range from negative infinity (a very poor model) to 1 . For example ,if all your model does is predict the mean of the mean of the targets , It's R^2 value would be  0.And if your model prefectly a range of numbers it's R^2 value be 1

# In[49]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df['Hs'], df['Hmax'], df['Tz'])
ax.set_xlabel('Hs')
ax.set_ylabel('Hmax')
ax.set_zlabel('Tz')
ax.set_title('3D Scatter Plot of Hs, Hmax, and Tz')
plt.show()


# In[50]:


from sklearn.metrics import r2_score

# Assuming you have already trained your RandomForestRegressor model (rf)
y_pred = rf.predict(X_test)

# Calculate R^2 score
r2 = r2_score(y_test, yp_test)

print("R^2 score:", r2)


# In[51]:


from sklearn.metrics import mean_absolute_error

# Assuming you have already trained your RandomForestRegressor model (rf)
yp_test = rf.predict(X_test)

# Calculate Mean Absolute Error
mae = mean_absolute_error(y_test, yp_test)

print("Mean Absolute Error:", mae)


# In[52]:


y_test.mean()


# In[53]:


yp_test.mean()


# In[54]:


y_test_mean = np.full(len(y_test),y_test.mean())


# In[55]:


r2_score(y_test,y_test_mean)


# In[56]:


r2_score(y_test , yp_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




