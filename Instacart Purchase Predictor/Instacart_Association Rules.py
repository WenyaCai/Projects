# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 20:16:42 2023

@author: liyat
"""


# import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from matplotlib import pyplot
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.stats import f
from sklearn.metrics import calinski_harabasz_score
from collections import Counter
from itertools import combinations, groupby
from sklearn.cluster import DBSCAN


#import dataset
df_aisles=pd.read_csv("C:/Users/liyat/Desktop/INSY662/instacart/aisles.csv")
df_departments=pd.read_csv("C:/Users/liyat/Desktop/INSY662/instacart/departments.csv")
df_products=pd.read_csv("C:/Users/liyat/Desktop/INSY662/instacart/products.csv")
df_orders=pd.read_csv("C:/Users/liyat/Desktop/INSY662/instacart/orders.csv")
df_order_products_prior=pd.read_csv("C:/Users/liyat/Desktop/INSY662/instacart/order_products__prior.csv")
df_order_products_train=pd.read_csv("C:/Users/liyat/Desktop/INSY662/instacart/order_products__train.csv")

#EDA
#from orders df
#number of orders by each user(distribution)
max_values = df_orders.groupby('user_id')['order_number'].max().reset_index()
data = max_values["order_number"]
plt.show()
#day-of-week distribution
data = df_orders["order_dow"]
plt.hist(data, bins=30, density=True, alpha=0.75, color='b')
plt.show()
#hour of day distribution
data = df_orders["order_hour_of_day"]
plt.hist(data, bins=30, density=True, alpha=0.75, color='b')
plt.show()
#days since prior order distribution
data = df_orders["days_since_prior_order"]
plt.hist(data, bins=30, density=True, alpha=0.75, color='b')
plt.show()

#from order_product_train df
#combine with product df, aisles df, and departments df
merged_df = pd.merge(pd.merge(pd.merge(df_order_products_train, df_products, on='product_id', how='inner'), df_aisles, on='aisle_id', how='inner'), df_departments, on='department_id', how='inner')
#check the top10 most popular products
product_counts=merged_df["product_name"].value_counts()
print(product_counts.head(10))
#check the distribution of departments and aisles
department_counts=merged_df["department"].value_counts()
print(department_counts)
aisle_counts=merged_df["aisle"].value_counts()
print(aisle_counts.head(10))

###################################
####Predict Next purchased items###
###################################

### Customer Segmentation
combined_df = pd.merge(pd.merge(pd.merge(df_orders, df_order_products_prior, on='order_id', how='inner'), df_products, on='product_id', how='inner'), df_aisles, on='aisle_id', how='inner')
segment_df = combined_df[['user_id','aisle']]
grouped_df = segment_df.groupby('user_id')['aisle'].value_counts().reset_index(name='count')
pivot_df = grouped_df.pivot_table(index='user_id', columns='aisle', values='count', fill_value=0)

# standardize data with z-score standardization
scaler = StandardScaler()
df_std = scaler.fit_transform(pivot_df)

# visualization with PCA to choose clustering method
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_std)
df_pca = pd.DataFrame(df_pca, columns =['PC1', 'PC2']) 

pyplot.scatter(df_pca['PC1'], df_pca['PC2'])
pyplot.xlabel("PC 1")
pyplot.ylabel("PC 2")
pyplot.show()

# apply dbscan
dbscan = DBSCAN(eps=0.5, min_samples=200) 
labels_ = dbscan.fit_predict(df_std)
silhouette_score(df_std,labels_)

# tried min_sample=50,200,500,1000, all yield a negative silhouette score -> misclustering

# apply k-mean clustering
# measure model performance using elbow method
withinss = []
for i in range (2,8):    
    kmeans = KMeans(n_clusters=i, n_init='auto')
    model = kmeans.fit(df_std)
    withinss.append(model.inertia_)

pyplot.plot([2,3,4,5,6,7],withinss)
#K=2 or 5

# measure model performance using silhouette score
for i in range (2,8):    
    kmeans = KMeans(n_clusters=i, n_init='auto')
    model = kmeans.fit(df_std)
    labels = model.labels_
    print(i,':',silhouette_score(df_std,labels))

# measure model performance using f-score
for i in range (2,8):    
    df1=i-1
    df2=206209-i
    kmeans = KMeans(n_clusters=i, n_init='auto')
    model = kmeans.fit(df_std)
    labels = model.labels_
    score = calinski_harabasz_score(df_std, labels)
    print(i,'F-score:',score)
    print(i,'p-value:',1-f.cdf(score, df1, df2))
#k=2

# segment customers into 2 groups
kmeans = KMeans(n_clusters=2, n_init='auto')
cluster_assignments = kmeans.fit_predict(df_std)
divided_datasets = []
for cluster_idx in range(2):
    cluster_subset = pivot_df[cluster_assignments == cluster_idx]
    divided_datasets.append(cluster_subset)

user_segment1=pd.DataFrame(divided_datasets[0])
user_segment2=pd.DataFrame(divided_datasets[1])


# show the segment characteristics
seg1_des=pd.Series(user_segment1.mean())
seg2_des=pd.Series(user_segment2.mean())
description=pd.concat([seg1_des, seg2_des], axis=1).rename(columns={0: 'Segment1',1:'Segment2'})

user_segment1_df = user_segment1.reset_index().melt(id_vars='user_id', var_name='aisle', value_name='count').dropna()
user_segment2_df = user_segment2.reset_index().melt(id_vars='user_id', var_name='aisle', value_name='count').dropna()

user_segment1_ids=user_segment1_df['user_id'].unique()
user_segment2_ids=user_segment2_df['user_id'].unique()

order_segment1_ids=df_orders[df_orders['user_id'].isin(user_segment1_ids)]['order_id'].unique()
order_segment2_ids=df_orders[df_orders['user_id'].isin(user_segment2_ids)]['order_id'].unique()

user_segment1_orders=df_order_products_prior[df_order_products_prior['order_id'].isin(order_segment1_ids)].reset_index()
user_segment1_orders=user_segment1_orders.drop(columns=['index'])
user_segment2_orders=df_order_products_prior[df_order_products_prior['order_id'].isin(order_segment2_ids)].reset_index()
user_segment2_orders=user_segment2_orders.drop(columns=['index'])

segment1_orders=user_segment1_orders.set_index('order_id')['product_id'].rename('item_id')
segment2_orders=user_segment2_orders.set_index('order_id')['product_id'].rename('item_id')

### Association Rules
# Returns frequency counts for items and item pairs
def freq(iterable):
    if type(iterable) == pd.core.series.Series:
        return iterable.value_counts().rename("freq")
    else: 
        return pd.Series(Counter(iterable)).rename("freq")

    
# Returns number of unique orders
def order_count(order_item):
    return len(set(order_item.index))


# Returns generator that yields item pairs, one at a time
def get_item_pairs(order_item):
    order_item = order_item.reset_index().to_numpy()
    for order_id, order_object in groupby(order_item, lambda x: x[0]):
        item_list = [item[1] for item in order_object]
              
        for item_pair in combinations(item_list, 2):
            yield item_pair
            

# Returns frequency and support associated with item
def merge_item_stats(item_pairs, item_stats):
    return (item_pairs
                .merge(item_stats.rename(columns={'freq': 'freqA', 'support': 'supportA'}), left_on='item_A', right_index=True)
                .merge(item_stats.rename(columns={'freq': 'freqB', 'support': 'supportB'}), left_on='item_B', right_index=True))


# Returns name associated with item
def merge_item_name(rules, item_name):
    columns = ['itemA','itemB','freqAB','supportAB','freqA','supportA','freqB','supportB', 
               'confidenceAtoB','confidenceBtoA','lift']
    rules = (rules
                .merge(item_name.rename(columns={'item_name': 'itemA'}), left_on='item_A', right_on='item_id')
                .merge(item_name.rename(columns={'item_name': 'itemB'}), left_on='item_B', right_on='item_id'))
    return rules[columns]   

def association_rules(order_item, min_support):

    print("Starting order_item: {:22d}".format(len(order_item)))


    # Calculate item frequency and support
    item_stats             = freq(order_item).to_frame("freq")
    item_stats['support']  = item_stats['freq'] / order_count(order_item) * 100


    # Filter from order_item items below min support 
    qualifying_items       = item_stats[item_stats['support'] >= min_support].index
    order_item             = order_item[order_item.isin(qualifying_items)]

    print("Items with support >= {}: {:15d}".format(min_support, len(qualifying_items)))
    print("Remaining order_item: {:21d}".format(len(order_item)))


    # Filter from order_item orders with less than 2 items
    order_size             = freq(order_item.index)
    qualifying_orders      = order_size[order_size >= 2].index
    order_item             = order_item[order_item.index.isin(qualifying_orders)]

    print("Remaining orders with 2+ items: {:11d}".format(len(qualifying_orders)))
    print("Remaining order_item: {:21d}".format(len(order_item)))


    # Recalculate item frequency and support
    item_stats             = freq(order_item).to_frame("freq")
    item_stats['support']  = item_stats['freq'] / order_count(order_item) * 100


    # Get item pairs generator
    item_pair_gen          = get_item_pairs(order_item)


    # Calculate item pair frequency and support
    item_pairs              = freq(item_pair_gen).to_frame("freqAB")
    item_pairs['supportAB'] = item_pairs['freqAB'] / len(qualifying_orders) * 100

    print("Item pairs: {:31d}".format(len(item_pairs)))
    
    # Filter from item_pairs those below min support
    item_pairs              = item_pairs[item_pairs['supportAB'] >= min_support]

    print("Item pairs with support >= {}: {:10d}\n".format(min_support, len(item_pairs)))


    # Create table of association rules and compute relevant metrics
    item_pairs = item_pairs.reset_index().rename(columns={'level_0': 'item_A', 'level_1': 'item_B'})
    item_pairs = merge_item_stats(item_pairs, item_stats)
    
    item_pairs['confidenceAtoB'] = item_pairs['supportAB'] / item_pairs['supportA']
    item_pairs['confidenceBtoA'] = item_pairs['supportAB'] / item_pairs['supportB']
    item_pairs['lift']           = item_pairs['supportAB'] / (item_pairs['supportA'] * item_pairs['supportB'])
    
    
    # Return association rules sorted by lift in descending order
    return item_pairs.sort_values('lift', ascending=False)


segment1_rules = association_rules(segment1_orders, 0.01)
segment2_rules = association_rules(segment2_orders, 0.01)

# Replace item ID with item name and display association rules
item_name = df_products
item_name = item_name.rename(columns={'product_id':'item_id', 'product_name':'item_name'})
segment1_rules_final = merge_item_name(segment1_rules, item_name).sort_values('lift', ascending=False)
segment2_rules_final = merge_item_name(segment2_rules, item_name).sort_values('lift', ascending=False)

# Only keeping associations that have a lift score higher than 1
segment1_association_list=segment1_rules_final[segment1_rules_final['lift']>1][['itemA','itemB']]
segment2_association_list=segment2_rules_final[segment2_rules_final['lift']>1][['itemA','itemB']]

# Check the final association list for the two segments
segment1_association_list
segment2_association_list
