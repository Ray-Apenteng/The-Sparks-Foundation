#!/usr/bin/env python
# coding: utf-8

# ## **Predicting Student Performance Based on Study Hours: A Simple Linear Regression Approach**

# ### **Introduction**
# Understanding the factors influencing student performance is essential for enhancing academic outcomes. This project explores the relationship between study hours and student scores using simple linear regression, a statistical method that models the relationship between a dependent variable and an independent variable. Using a dataset of study hours and corresponding scores, we will develop a predictive model to estimate student performance based on study habits, providing insights for educators and students to improve academic results.

# ![craiyon_012909_The_image_showcases_a_student_studying_in_a_serene_environment__surrounded_by_books_a.png](attachment:craiyon_012909_The_image_showcases_a_student_studying_in_a_serene_environment__surrounded_by_books_a.png)

# ### ***Setup and Import Libraries***
# Start by importing the necessary libraries.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# ### ***Load the Data***

# In[6]:


#read the csv data from the data source which is a url
url = "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
df = pd.read_csv(url)
df.head()


# ### ***Exploratory Data Analysis (EDA)***
# Examine the data to understand its structure and summary statistics.

# In[8]:


# Checking the structure of the dataset
df.info()


# In[10]:


# Summary statistics
df.describe()


# In[11]:


#Checking if the data is normally distributed
df.skew()


# In[12]:


#Checking for outliers
df.kurtosis()


# **The data is symmetric and there are no considerable outliers in the data**

# ##### ***Graphical Analysis of the data***

# In[14]:


print(plt.style.available)
plt.style.use('fivethirtyeight')


# In[17]:


#Visualizing the relationship between the study hours and scores
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='Hours', y='Scores')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()


# **From the data we can see that there is a positive linear relationship between the number of study hours and the percentage score. That is the more hours of study the higher the score will be**

# ### ***Data Preparation***
# We will separate the features (independent variable) and target (dependent variable).

# In[19]:


# Define features and target
X = df[['Hours']]  # Features
y = df['Scores']   # Target


# ### ***Data Split***
# Now we will split the data into training and testing sets. 80% of the data will be used to train the model while 20% will be used to test the model

# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


# ### ***Train the Model***
# Now we will train the simple linear regression model.

# In[21]:


# Create the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

print("The Linear Regression Model has been trained")


# ### ***Making  Predictions***
# Let's use the model to make predictions on the test set.

# In[23]:


print(X_test) # print test data
y_pred = model.predict(X_test) # Predicting the scores using test data


# ### ***Evaluate the Model***
# Evaluate the model's performance using metrics like Mean Absolute Error (MAE) and R-squared.

# In[24]:


mae = mean_absolute_error(y_test, y_pred) # Mean Absolute Error (MAE)
r2 = r2_score(y_test, y_pred) # R-squared

print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')


# A Mean Absolute Error (MAE) of 4.877 indicates that, on average, the model's predictions are approximately 4.88 units away from the actual values. This suggests that the model is making relatively accurate predictions.
# 
# The R-squared value of 0.973 indicates that approximately 97.3% of the variance in the target variable can be explained by the linear relationship with the predictor variable(s).

# In[29]:


df_check = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df_check


# ### ***Visualization of Predictions***
# Visualize the regression line and the predictions.

# In[38]:


# Plotting the regression line
plt.figure(figsize=(8,6))
plt.scatter(X, y, label='Actual data')
plt.plot(X, model.predict(X), color='orange', label='Regression line')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.legend()
plt.show()

# Plotting predictions vs actual values
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.plot([0, 100], [0, 100], '--', color='orange')
plt.xlabel('Actual Scores')
plt.ylabel('Predicted Scores')
plt.title('Actual vs Predicted Scores')
plt.show()


# **Answering the question, What will be predicted score if a student studies for 9.25 hrs/day?**

# In[28]:


# Assign the variable you want to predict to hours and use the model to predict the score
hours = np.array([[9.25]])
my_pred = model.predict(hours)
print(f"According to the trained model, if a student studies for 9.25 hrs/day they are likely to achieve a score of {my_pred[0]:.2f} in the exam.")


# ### Conclusion
# 
# In this project, we performed a simple linear regression analysis to predict the percentage score of a student based on the number of hours studied. The model was trained using the training set and evaluated using the test set. The Mean Absolute Error (MAE) and R-squared value indicated that the model has a reasonable accuracy in predicting the scores.
# 
# Further improvements could be made by collecting more data and possibly exploring more complex models if needed.
