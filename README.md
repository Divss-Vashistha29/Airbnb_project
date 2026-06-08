# Airbnb_project

import pandas as pd # pyright: ignore[reportMissingModuleSource]
import numpy as np # pyright: ignore[reportMissingImports]
import os
from sklearn.model_selection import train_test_split # pyright: ignore[reportMissingModuleSource]
from sklearn.linear_model import LinearRegression # pyright: ignore[reportMissingModuleSource, reportMissingImports]
from sklearn.metrics import mean_squared_error, r2_score  # type: ignore
import seaborn as sns  # type: ignore
import matplotlib.pyplot as plt  # type: ignore


# DATA COLLECTION
print("\n loading the dataset...\n")
data= pd.read_csv(r'AB_NYC_2019.csv')

# Display the first few rows of the dataset
print("\n displaying the first 10 rows of the dataset...\n")
print(data.head(10))

# Display summary statistics of the dataset
print("\n displaying summary statistics...\n")
print(data.describe())

# display the datatypes of each column
print("\n displaying datatypes...\n")
print(data.dtypes)

# ensure completeness and accuracy
print("\n drop duplicate values..")
data=data.drop_duplicates(subset=['id'])
print(data.duplicated().sum())

# check the null values in the dataset
print("\n counting null values...\n")
print(data.isnull().sum())

# type conversion
print("\n type conversion")
data['last_review']=pd.to_datetime(data['last_review'],errors='coerce', dayfirst=True)
print(data['last_review'].dtypes)
data['reviews_per_month']=pd.to_numeric(data['reviews_per_month'],errors='coerce')
#replace non-numeric values with NaN and convert to numeric
data['price']= data['price'].astype(str).str.replace(r'[^0-9.]','',regex=True)
data['price']= pd.to_numeric(data['price'],errors='coerce')


# handle missing values
print("\n handling missing values...\n")
data['last_review']=data['last_review'].interpolate()
print('last_review',data['last_review'].head(10))
data['reviews_per_month']= data['reviews_per_month'].interpolate(method='linear')
data['name']= data['name'].fillna('No listing name')
data['host_name']=data['host_name'].fillna('unknown host')

# DATA EXPLORATION
# filtering the room types 
data=data[data['room_type'].isin(['Entire home/apt', 'Private room', 'Shared room'])]
#filtering the manhattan and brooklyn neighbour groups
data_mb=data[data['neighbourhood_group'].isin(['Manhattan', 'Brooklyn'])]
# grouping the average price acc. to room_type and neighbourhood
avg_price=data_mb.groupby(['neighbourhood_group','room_type'])['price'].mean().reset_index()
print("Average Price by Neighbourhood Group and Room Type:",avg_price)


#print(data[['neighbourhood_group','room_type','minimum_nights','number_of_reviews','reviews_per_month','availability_365','price']].isna().sum())

# LINEAR REGRESSION MODEL 
features=['neighbourhood_group','room_type', 'minimum_nights','number_of_reviews', 'reviews_per_month', 'availability_365']
x=data[features]
y= np.log1p(data['price'])
x= pd.get_dummies(x,drop_first=True)
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2, random_state=42)
model= LinearRegression()
model.fit(x_train,y_train)
# predictions
y_pred_log= model.predict(x_test)
# model evaluation
print("r2 score:", r2_score(y_test,y_pred_log))
print("RMSE:",np.sqrt(mean_squared_error(y_test,y_pred_log)))
# inspect the coefficients
coeff= pd.DataFrame({'feature': x.columns,'coefficient': model.coef_}).sort_values(by='coefficient',ascending=False)
print(coeff.head(10))

# VISUALIZATION
# boxplot
sns.boxplot(x='neighbourhood_group',y='price',data=data)
plt.xlabel('Neighbourhood Group')
plt.ylabel('Price')
plt.title('Price distribution by Neighbourhood Group')
plt.show()
# histogram plot
sns.histplot(x='number_of_reviews',y='price',data=data,color="red")
plt.xlabel('Number of Reviews')
plt.ylabel('Price')
plt.title('Price vs Number of Reviews')
plt.show()
