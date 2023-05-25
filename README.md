# Ex-07-Data-Visualization-

## AIM
To Perform Data Visualization on the given dataset and save the data to a file. 

# Explanation
Data visualization is the graphical representation of information and data. By using visual elements like charts, graphs, and maps, data visualization tools provide an accessible way to see and understand trends, outliers, and patterns in data.

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature generation and selection techniques to all the features of the data set
### STEP 4
Apply data visualization techniques to identify the patterns of the data.


# PROGRAM:
```py
# Data Pre-Processing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Semester 3/19AI403 _Intro to DS/Exp_7/Superstore.csv",encoding="latin-1")
df

df.head()
df.info()

df.drop('Row ID',axis=1,inplace=True)
df.drop('Order ID',axis=1,inplace=True)
df.drop('Customer ID',axis=1,inplace=True)
df.drop('Customer Name',axis=1,inplace=True)
df.drop('Country',axis=1,inplace=True)
df.drop('Postal Code',axis=1,inplace=True)
df.drop('Product ID',axis=1,inplace=True)
df.drop('Product Name',axis=1,inplace=True)
df.drop('Order Date',axis=1,inplace=True)
df.drop('Ship Date',axis=1,inplace=True)
print("Updated dataset")
df

df.isnull().sum()

# detecting and removing outliers in current numeric data
plt.figure(figsize=(8,8))
plt.title("Data with outliers")
df.boxplot()
plt.show()
plt.figure(figsize=(8,8))
cols = ['Sales','Quantity','Discount','Profit']
Q1 = df[cols].quantile(0.25)
Q3 = df[cols].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
plt.title("Dataset after removing outliers")
df.boxplot()
plt.show()

# Which Segment has Highest sales?
sns.lineplot(x="Segment",y="Sales",data=df,marker='o')
plt.title("Segment vs Sales")
plt.xticks(rotation = 90)
plt.show()
sns.barplot(x="Segment",y="Sales",data=df)
plt.xticks(rotation = 90)
plt.show()

# Which City has Highest profit?
df.shape
df1 = df[(df.Profit >= 60)]
df1.shape
plt.figure(figsize=(30,8))
states=df1.loc[:,["City","Profit"]]
states=states.groupby(by=["City"]).sum().sort_values(by="Profit")
sns.barplot(x=states.index,y="Profit",data=states)
plt.xticks(rotation = 90)
plt.xlabel=("City")
plt.ylabel=("Profit")
plt.show()


# Which ship mode is profitable?
sns.barplot(x="Ship Mode",y="Profit",data=df)
plt.show()
sns.lineplot(x="Ship Mode",y="Profit",data=df)
plt.show()
sns.violinplot(x="Profit",y="Ship Mode",data=df)
sns.pointplot(x=df["Profit"],y=df["Ship Mode"])

# Sales of the product based on region.
states=df.loc[:,["Region","Sales"]]
states=states.groupby(by=["Region"]).sum().sort_values(by="Sales")
sns.barplot(x=states.index,y="Sales",data=states)
plt.xticks(rotation = 90)
plt.xlabel=("Region")
plt.ylabel=("Sales")
plt.show()
df.groupby(['Region']).sum().plot(kind='pie', y='Sales',figsize=(6,9),pctdistance=1.7,labeldistance=1.2)

# Find the relation between sales and profit.
df["Sales"].corr(df["Profit"])
df_corr = df.copy()
df_corr = df_corr[["Sales","Profit"]]
df_corr.corr()
sns.pairplot(df_corr, kind="scatter")
plt.show()
# Heatmap
df4=df.copy()

#encoding
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder
le=LabelEncoder()
ohe=OneHotEncoder
oe=OrdinalEncoder()

df4["Ship Mode"]=oe.fit_transform(df[["Ship Mode"]])
df4["Segment"]=oe.fit_transform(df[["Segment"]])
df4["City"]=le.fit_transform(df[["City"]])
df4["State"]=le.fit_transform(df[["State"]])
df4['Region'] = oe.fit_transform(df[['Region']])
df4["Category"]=oe.fit_transform(df[["Category"]])
df4["Sub-Category"]=le.fit_transform(df[["Sub-Category"]])

#scaling
from sklearn.preprocessing import RobustScaler
sc=RobustScaler()
df5=pd.DataFrame(sc.fit_transform(df4),columns=['Ship Mode', 'Segment', 'City', 'State','Region',
                                               'Category','Sub-Category','Sales','Quantity','Discount','Profit'])

# Heatmap
plt.subplots(figsize=(12,7))
sns.heatmap(df5.corr(),cmap="PuBu",annot=True)
plt.show()
Find the relation between sales and profit based on the following category.

# Segment
df_corr = df5.copy()
df_corr = df_corr[["Sales","Profit","Segment"]]
df_corr.corr()

# City
df_corr = df5.copy()
df_corr = df_corr[["Sales","Profit","City"]]
df_corr.corr()

# States
df_corr = df5.copy()
df_corr = df_corr[["Sales","Profit","State"]]
df_corr.corr()

# Segment and Ship Mode
df_corr = df5.copy()
df_corr = df_corr[["Sales","Profit","Segment","Ship Mode"]]
df_corr.corr()

# Segment, Ship mode and Region
df_corr = df5.copy()
df_corr = df_corr[["Sales","Profit","Segment","Ship Mode","Region"]]
df_corr.corr()

# Loss 
loss_df=df[df['Profit'] < 0]
loss_df.groupby(by='City').sum().sort_values('Profit',ascending=True).head(10)
loss_df.sort_values(['Sales'],ascending=True).groupby(by='Category').mean()
plt.rcParams['figure.figsize']=(15,3)
plt.bar(loss_df['Sub-Category'],loss_df['Sales']);
plt.rcParams.update({'font.size':10});
plt.xlabel('SubCategory');
plt.ylabel('Sales');

plt.rcParams['figure.figsize']=(10,8)
plt.bar(df['Ship Mode'],df['Sales']);
plt.rcParams.update({'font.size':14});
plt.xlabel('Ship Mode');
plt.ylabel('Sales');

```

###  OUPUT:
### Dataset:
![s1](https://github.com/simbu07/Ex-08-Data-Visualization-/assets/94525786/0b286fa7-b42d-476d-81ed-403cee64a1a2)

### Data Preprocessing:
![s2](https://github.com/simbu07/Ex-08-Data-Visualization-/assets/94525786/6a80d512-c314-4819-810e-417f34c2f05b)

![s3](https://github.com/simbu07/Ex-08-Data-Visualization-/assets/94525786/378c4fda-61ea-4def-a3b2-e2b4f7823d81)

![s4](https://github.com/simbu07/Ex-08-Data-Visualization-/assets/94525786/6b26a683-dd35-4b39-9311-4f6fcf4c9425)

![s5](https://github.com/simbu07/Ex-08-Data-Visualization-/assets/94525786/f1377b9e-46c1-480f-8adb-072b2d221940)

![s6](https://github.com/simbu07/Ex-08-Data-Visualization-/assets/94525786/63e0d2c9-2784-45fd-9401-fa77003e159c)
### Segement has highest Sales?
#### Comparatively consumer Segment has the highest sales
![s7](https://github.com/simbu07/Ex-08-Data-Visualization-/assets/94525786/6ccdb6a5-15d5-4fae-861d-b8b750480b86)

![s8](https://github.com/simbu07/Ex-08-Data-Visualization-/assets/94525786/6f22637c-a7e1-4506-8e3f-3da9eb801ea6)
### Which City has Highest profit?
#### New York City has the Highest Profit
![s9](https://github.com/simbu07/Ex-08-Data-Visualization-/assets/94525786/9e8633dc-9244-42be-8c24-37fae92f6c9b)
### Which ship mode is profitable?
#### First Class Ship Mode is most profitable
![s10](https://github.com/simbu07/Ex-08-Data-Visualization-/assets/94525786/4e24e185-59c9-4931-bfe0-6bf16bfbdc50)

![s11](https://github.com/simbu07/Ex-08-Data-Visualization-/assets/94525786/104fd77b-3383-480f-9d6f-45d6177d03b4)

![s12](https://github.com/simbu07/Ex-08-Data-Visualization-/assets/94525786/e3d121f2-6555-4319-bacd-bc11fbb82bc1)

![s13](https://github.com/simbu07/Ex-08-Data-Visualization-/assets/94525786/8da5e6cc-4188-4fbb-bf73-5be16b1d9975)

![s14](https://github.com/simbu07/Ex-08-Data-Visualization-/assets/94525786/814fd16b-cb1b-43e7-96f8-c46a3c8df2b1)

![s15](https://github.com/simbu07/Ex-08-Data-Visualization-/assets/94525786/f8e2604c-f7d0-41ab-8960-33d6829bb48b)
### Find the relation between sales and profit?
#### Sales is not much related to profit
![s16](https://github.com/simbu07/Ex-08-Data-Visualization-/assets/94525786/47e50162-d783-4514-babb-46e8b374bdf7)

![s17](https://github.com/simbu07/Ex-08-Data-Visualization-/assets/94525786/2389d6d6-f959-4012-ae50-e54fab562206)
### Heatmap:
![s18](https://github.com/simbu07/Ex-08-Data-Visualization-/assets/94525786/7c5ed109-0db0-4856-b60e-b2b8072a986e)

### Find the relation between sales and profit based on the following category.
### Segment - Profit is much related to Segment than Sales

### City - Profit is much related to City than Sales

### States - Sales is much related to City than Profit

### Segment and Ship Mode - Ship mode is more related to Sales than Profit

### Segment, Ship mode and Region - Region is more related to Profit than Sales
![s19](https://github.com/simbu07/Ex-08-Data-Visualization-/assets/94525786/667d39d8-0cd1-46ca-826f-10cfb1c8bf37)

![s20](https://github.com/simbu07/Ex-08-Data-Visualization-/assets/94525786/3fe66a95-16f1-4e2c-b2ad-745c8ed43e58)



### RESULT:
Thus, Data Visualization is performed on the given dataset and save the data to a file.
