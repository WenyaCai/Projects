import pandas as pd
import numpy as np
import seaborn as sns                       #visualisation
import matplotlib.pyplot as plt             #visualisation
#%matplotlib inline     
df = pd.read_excel("Kickstarter.xlsx")

#preprocessing
df['goal_usd'] = df['goal']*df['static_usd_rate']
# Function to determine the season based on the month
def get_season(month):
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Autumn'
    else:
        return 'Winter'

# Extract year and season for the specified columns
for col in ['deadline', 'state_changed_at', 'created_at', 'launched_at']:
    df[col + '_year_season'] = df[col].dt.year.astype(str) + df[col].dt.month.apply(get_season)
    df[col + '_season'] = df[col].dt.month.apply(get_season)
#we are not focusing on the time of the day, so we will drop the hour. we also drop the day of the week. day of month is also dropped as its less relevant, as most days are on a multiple of 5
#blurb_len and name_len are dropped as we are keeping the cleaned versions
df = df.drop(['name','state_changed_at','deadline','created_at','launched_at','pledged','goal','currency','id','static_usd_rate','name_len','blurb_len'
              ,'state_changed_at_hr','launched_at_hr','created_at_hr','deadline_hr',
              'launched_at_day','state_changed_at_day','deadline_day','created_at_day'
              ], axis=1)
df['category'] = df['category'].fillna('None')

df = df.drop(df[(df.state != 'successful') & (df.state != 'failed')].index)

df['goal_fulfilled'] = df['usd_pledged']/df['goal_usd']
#goal fulfilled rate is highly skewed and has a lot of outliers, so we use below to categorize it
df['goal_fulfilled_cat'] = df['goal_fulfilled'].apply(lambda x: True if x >= 1 else False)
df['goal_over_fulfilled'] = df['goal_fulfilled'].apply(lambda x: 'high' if x >= 5 else ('not_overfulfilled' if x <= 1 else 'overfulfilled'))

#we do not include month as it has almost uniform distribution
#most predictors did not turn on communication, so we will drop it

#we only focus on US and GB as they have the most data, others are less than 5% of the data
df = df.drop(df[(df.country != 'US') & (df.country != 'GB')].index)
# below are the predictors we will use
im3 = ['state', 'country', 'staff_pick',
       'backers_count', 'usd_pledged', 'category', 'spotlight',
       'name_len_clean', 'blurb_len_clean',
       'launched_at_yr', 'state_changed_at_yr',
        'launch_to_state_change_days', 'create_to_launch_days','goal_usd',
       'state_changed_at_year_season', 
       'launched_at_year_season', 'goal_fulfilled_cat','goal_over_fulfilled']

df = df[im3]

# drop outliers backers > 30000, usd_pledged >2000000,deadline_year < 2013,state_change_year < 2013,create_to launch > 900,goal_usd < 10000000
df = df.drop(df[(df.backers_count > 30000) | (df.usd_pledged > 2000000)| (df.state_changed_at_yr < 2013) | (df.create_to_launch_days > 900) | (df.goal_usd > 10000000)].index)

from kmodes.kprototypes import KPrototypes
import matplotlib.pyplot as plt
import numpy as np

numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.select_dtypes(include=['object', 'bool','category']).columns

#normalizing numerical data
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
scaler = MinMaxScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
# Convert DataFrame to NumPy array
data = df.to_numpy()

# Indices of categorical columns
categorical_indices = [df.columns.get_loc(col) for col in categorical_cols]

# Finding the best number of clusters using the Elbow Method
costs = []
K_range = range(2, 10)  # Testing for k from 2 to 10
for k in K_range:
    kproto = KPrototypes(n_clusters=k, init='Cao', n_init=10, verbose=0)
    clusters = kproto.fit_predict(data, categorical=categorical_indices)
    costs.append(kproto.cost_)

plt.figure(figsize=(10, 6))
plt.plot(K_range, costs, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('Cost')
plt.show()

# optimal number of clusters is 4 according to the elbow method
kproto = KPrototypes(n_clusters=4, init='Cao', n_init=10, verbose=0)
clusters = kproto.fit_predict(data, categorical=categorical_indices)

# see characteristics of each cluster
df['cluster'] = clusters
mode_df = pd.DataFrame()

# Calculate the mode for each variable in each cluster
for cluster in df['cluster'].unique():
    cluster_mode = df[df['cluster'] == cluster].mode().iloc[0]  # .iloc[0] is used to select the first mode in case of multiple modes
    cluster_mode.name = f'cluster {cluster}'  # Naming the series with the cluster label
    mode_df = pd.concat([mode_df, cluster_mode.to_frame().T])  # Concatenate the cluster mode dataframe

# Display the resulting table
print(mode_df)

from sklearn.preprocessing import OneHotEncoder
new_df = df.copy()
new_df = new_df.drop(['cluster'], axis=1)
for column in df.select_dtypes(include=['object','bool']).columns:
    new_df[column] = df[column].astype('category')
    
dummy = OneHotEncoder(drop='first', sparse=False)  
dummy_df = dummy.fit_transform(new_df.select_dtypes(include=['category']))
columns = dummy.get_feature_names_out()

# Create a new DataFrame from the one-hot encoded matrix
encoded_df = pd.DataFrame(dummy_df, columns=columns)

# Reset index if necessary before concatenation
new_df.reset_index(drop=True, inplace=True)
encoded_df.reset_index(drop=True, inplace=True)

# Concatenate the dataframes
df1 = pd.concat([new_df, encoded_df], axis=1)

df1=df1.drop(new_df.select_dtypes(include=['category']).columns, axis=1)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Dimensionality Reduction (e.g., PCA)
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(df1) 

# Plot using Seaborn's scatterplot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=df['cluster'], palette='viridis')
plt.title("Cluster Visualization with PCA")
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()