#!/usr/bin/env python
# coding: utf-8

# # -*- coding: utf-8 -*-
# """
# Created on sun sept 5 08:58:38 2020
# 
# @author: ashutosh yadav
# SCH NO.. 181112022
# """
# # Data Science Assignment : Using Formula
# #importing libraries
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn import linear_model
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# import seaborn as sns
# from sklearn.metrics import r2_score

# In[6]:


#import data sets
from sklearn.datasets import load_iris 

iris = load_iris()
x = iris.data
y = iris.target
print(iris['target_names'])


# In[16]:


#split dataset into training set and test set
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.2 , random_state = 101)


# In[17]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[21]:


reg = linear_model.LinearRegression()
reg.fit(x_train,y_train)
print(reg.coef_) #it is b1 of eq b0+b1*x
print (reg.intercept_)


# In[23]:


pred_y = reg.predict(x_test)


# In[27]:


acc = r2_score(y_test , pred_y)
print(acc)


# In[28]:


#MAE measures average mag. of errors in set predictions without considering their dir..
print (mean_absolute_error(y_test,pred_y))


# In[30]:


#PMSE is average squared difference b/w estimated and actual val
print(mean_squared_error(y_test,pred_y))


# In[ ]:





# In[ ]:




