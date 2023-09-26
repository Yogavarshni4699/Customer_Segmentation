# Customer_Segmentation
Customer Segmentation Models in Python
<aside>
💡 DataSet:  [E-Commerce Dataset](https://www.kaggle.com/datasets/carrie1/ecommerce-data)

</aside>

Customer segmentation models are often used for dividing a company’s clients into different user groups. Customers in each group display shared characteristics that distinguish them from other users.

Here is a simple example of how companies use data segmentation to drive sales/marketing:

**H&M**. One of the easiest and most common ways to segment your customers is by their date of birth. It gives you a great opportunity — and an excuse — to send them a personalized email without appearing pushy. H&M is one of the brands that use this segmentation criterion.

After creating a customer segmentation model, we can notice a handful of customers like me always wait for a special offer before making purchases. They classify us into a segment called “thrifty shoppers.” Every time a new promotion is released, the company’s marketing team sends me and every other “thrifty shopper” a curated advertisement highlighting product affordability.

Whenever I get notified of a special discount, I rush to purchase all the items I require before the promotion ends, which increases the company’s sales.

Similarly, all the platform’s customers are grouped into different segments and sent targeted promotions based on their purchase behavior.

Customer segmentation models using unsupervised machine learning algorithms such as [K-Means clustering](https://365datascience.com/tutorials/python-tutorials/k-means-clustering/) or hierarchical clustering. These models can pick up on similarities between user groups that often go unnoticed by the human eye. RFM is used in marketing to analyze customer value and explore other metrics for evaluating the performance of a clustering algorithm.

### **Table of Contents:**

 1. ****Prerequisites for Building a Customer Segmentation Model****

1. ****Understand the Segmentation Data****
2. ****Preprocessing Data for Segmentation****
3. ****Building The Customer Segmentation Model****
4. ****Segmentation Model Interpretation and Visualization****

****Step 1: Prerequisites for Building a Customer Segmentation Model****

I will be using an [E-Commerce Dataset](https://www.kaggle.com/datasets/carrie1/ecommerce-data) from Kaggle that contains transaction information from around 4,000 customers. I used Google Colab to easily run the code provided and display visualizations at each step. Also, installed the following libraries  — Numpy, Pandas, Matplotlib, Seaborn, Scikit-Learn, Kneed, and Scipy.

****Step 2: Understand the Segmentation Data****

Getting to know the data.

```python
import pandas as pd
df = pd.read_csv('data.csv',encoding='unicode_escape')
df.head()
```

![image](https://github.com/Yogavarshni4699/Customer_Segmentation/assets/91062811/84f1f71c-efb4-4c13-abb4-01231bd6f184)


The dataframe consists of 8 variables:

1. InvoiceNo: The unique identifier of each customer invoice.
2. StockCode: The unique identifier of each item in stock.
3. Description: The item purchased by the customer.
4. Quantity: The number of each item purchased by a customer in a single invoice.
5. InvoiceDate: The purchase date.
6. UnitPrice: Price of one unit of each item.
7. CustomerID: A unique identifier assigned to each user.
8. Country: The country from where the purchase was made.

With the transaction data above, the task is to build different customer segments based on each user’s purchase behavior.

****Step 3: Preprocessing Data for Segmentation****

The raw data is complex and in a format that cannot be easily ingested by customer segmentation models. We need to do some preliminary data preparation to make this data interpretable.

The informative features in this dataset that tell us about customer buying behavior include “Quantity”, “InvoiceDate” and “UnitPrice.” These variation helps to derive a customer’s RFM profile - Recency, Frequency, Monetary Value.

RFM is commonly used in marketing to evaluate a client’s value based on their:

1. Recency: How recently have they made a purchase?
2. Frequency: How often have they bought something?
3. Monetary Value: How much money do they spend on average when making purchases?

With the variables in this e-commerce transaction dataset, we will calculate each customer’s recency, frequency, and monetary value. These RFM values will then be used to build the segmentation model.

### **Recency**

Let’s start by calculating recency. To identify a customer’s recency, we need to pinpoint when each user was last seen making a purchase:

```python
# convert the date column to DateTime format
df['Date']= pd.to_datetime(df['InvoiceDate'])
# keep only the most recent date of purchase
df['rank'] = df.sort_values(['CustomerID','Date']).groupby(['CustomerID'])['Date'].rank(method='min')
df_rec = df[df['rank']==1]
```

In the dataframe we just created, we only kept rows with the most recent date for each customer. We now need to rank every customer based on what time they last bought something and assign a recency score to them.

For example, if customer A was last seen acquiring an item 2 months ago and customer B did the same 2 days ago, customer B must be assigned a higher recency score.

To assign a recency score to each customerID, run the following lines of code:

```python
df_rec['recency'] = (df_rec['Date'] - pd.to_datetime(min(df_rec['Date']))).dt.days
```

The dataframe now has a new column called “recency” that tells us when each customer last bought something from the platform:


### **Frequency**

Now, let’s calculate frequency — how many times has each customer made a purchase on the platform:

```python
freq = df_rec.groupby('CustomerID')['Date'].count()
df_freq = pd.DataFrame(freq).reset_index()
df_freq.columns = ['CustomerID','frequency']
```

The new dataframe we created consists of two columns — “CustomerID” and “frequency.” Let’s merge this dataframe with the previous one:

```python
rec_freq = df_freq.merge(df_rec,on='CustomerID')
rec_freq.head()
```



****Monetary Value****

Finally, we can calculate each user’s monetary value to understand the total amount they have spent on the platform.

To achieve this, run the following lines of code:

```python
rec_freq['total'] = rec_freq['Quantity']*df['UnitPrice']
m = rec_freq.groupby('CustomerID')['total'].sum()
m = pd.DataFrame(m).reset_index()
m.columns = ['CustomerID','monetary_value']
rfm = m.merge(rec_freq,on='CustomerID')
```

Now, let’s select only the columns required to build the customer segmentation model:

```python
finaldf = rfm[['CustomerID','recency','frequency','monetary_value']]
```

### **Removing Outliers**

Successfully derived three meaningful variables from the raw, uninterpretable transaction data we started out with.

Before building the customer segmentation model, we first need to check the dataframe for outliers and remove them. To get a visual representation of outliers in the dataframe, let’s create a boxplot of each variable:

```python
import seaborn as sns
import matplotlib.pyplot as plt
list1 = ['recency','frequency','monetary_value']
for i in list1:
    print(str(i)+': ')
    ax = sns.boxplot(x=finaldf[str(i)])
    plt.show()
```

![image](https://github.com/Yogavarshni4699/Customer_Segmentation/assets/91062811/f07c6226-a78e-4edd-852f-9f11fd238d86)


Observe that “recency” is the only variable with no visible outliers. “Frequency” and “monetary_value”, on the other hand, have many outliers that must be removed before we proceed to build the model.

To identify outliers, we will compute a measurement called a Z-Score. Z-Scores tell us how far away from the mean a data point is.

```python
from scipy import stats
import numpy as np
# remove the customer id column
new_df = finaldf[['recency','frequency','monetary_value']]
# remove outliers
z_scores = stats.zscore(new_df)
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
new_df = new_df[filtered_entries]
```



### **Standardization**

The final pre-processing technique we will apply to the dataset is standardization.

Run the following lines of code to scale the dataset’s values so that they follow a normal distribution:

```python
from sklearn.preprocessing import StandardScaler
new_df = new_df.drop_duplicates()
col_names = ['recency', 'frequency', 'monetary_value']
features = new_df[col_names]
scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)
scaled_features = pd.DataFrame(features, columns = col_names)
```

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/5d11e880-cfbc-4ea6-912a-20575d117a48/eeeb9354-b384-445b-abc0-e31c0a40d727/Untitled.png)

****Step 4: Building The Customer Segmentation Model****

K-Means clustering algorithm to perform customer segmentation. The goal of a K-Means clustering model is to segment all the data available into non-overlapping sub-groups that are distinct from each other.

When building a clustering model, we need to decide how many segments we want to group the data into. We will create a loop and run the K-Means algorithm from 1 to 10 clusters. Then, we can plot model results for this range of values and select the elbow of the curve as the number of clusters to use.

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
SSE = []
for cluster in range(1,10):
    kmeans = KMeans(n_clusters = cluster, init='k-means++')
    kmeans.fit(scaled_features)
    SSE.append(kmeans.inertia_)
# converting the results into a dataframe and plotting them
frame = pd.DataFrame({'Cluster':range(1,10), 'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
```


The “elbow” of this graph is the point of inflection on the curve, and in this case is at the 4-cluster mark.

This means that the optimal number of clusters to use in this K-Means algorithm is 4. Let’s now build the model with 4 clusters:

```python
# First, build a model with 4 clusters
kmeans = KMeans( n_clusters = 4, init='k-means++')
kmeans.fit(scaled_features)
```

To evaluate the performance of this model, we will use a metric called the silhouette score. This is a coefficient value that ranges from -1 to +1. A higher silhouette score is indicative of a better model.

```python
print(silhouette_score(scaled_features, kmeans.labels_, metric='euclidean'))
```

The silhouette coefficient of this model is **0.44,** indicating reasonable cluster separation

## **Step 5: Segmentation Model Interpretation and Visualization**

Now that we have built our segmentation model, we need to assign clusters to each customer in the dataset:

```python
pred = kmeans.predict(scaled_features)
frame = pd.DataFrame(new_df)
frame['cluster'] = pred
```



Visualization:

```python
avg_df = frame.groupby(['cluster'], as_index=False).mean()
for i in list1:
    sns.barplot(x='cluster',y=str(i),data=avg_df)
    plt.show()
```

![image](https://github.com/Yogavarshni4699/Customer_Segmentation/assets/91062811/2ee07841-aae3-4220-9f90-12d076984955)


just by looking at the charts above, we can identify the following attributes of customers in each segment:
