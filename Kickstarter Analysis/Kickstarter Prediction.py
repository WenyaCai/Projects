import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline     
df = pd.read_excel("Kickstarter.xlsx")

#preprocessing
df['goal_usd'] = df['goal']*df['static_usd_rate']
#drop unnecessary columns
df = df.drop(['name','pledged','currency','state_changed_at','staff_pick','backers_count','usd_pledged','spotlight','state_changed_at_weekday','state_changed_at_month','state_changed_at_day','state_changed_at_yr','state_changed_at_hr','launch_to_state_change_days','deadline','created_at','launched_at','goal'], axis=1)
#we find that there are some projects that dont have a category, so we fill them with None instead of dropping them
df['category'] = df['category'].fillna('None')

#drop projects that are not successful or failed
df = df.drop(df[(df.state != 'successful') & (df.state != 'failed')].index)

#during EDA, we found that most projects takes place in 2014 and 2015, so we categorize the years into 2014, 2015, and other
df['deadline_yr_category'] = df['deadline_yr'].astype('category').apply(lambda x: '2014' if x == 2014 else ('2015' if x == 2015 else 'other'))
df['created_at_yr_category'] = df['created_at_yr'].astype('category').apply(lambda x: '2014' if x == 2014 else ('2015' if x == 2015 else 'other'))
df['launched_at_yr_category'] = df['launched_at_yr'].astype('category').apply(lambda x: '2014' if x == 2014 else ('2015' if x == 2015 else 'other'))


#during EDA, we found that below variables are dangerously skewed, even though gradient boosting classifier can handle skewed data,
# but we found that the accuaracy is higher after scaled, probably due to it's now easier to group with other features to represent
# so we scale them
df['launch_to_deadline_days'] = np.log1p(df['launch_to_deadline_days'])
df['create_to_launch_days'] = np.sqrt(df['create_to_launch_days'])
df['goal_usd'] = np.cbrt(df['goal_usd'])

# we dummy the categorical variables
from sklearn.preprocessing import OneHotEncoder

for column in df.select_dtypes(include=['object']).columns:
    df[column] = df[column].astype('category')
    
dummy = OneHotEncoder(drop='first', sparse=False)  
dummy_df = dummy.fit_transform(df.select_dtypes(include=['category']))
columns = dummy.get_feature_names_out()

# Create a new DataFrame from the one-hot encoded matrix
encoded_df = pd.DataFrame(dummy_df, columns=columns)

# Reset index if necessary before concatenation
df.reset_index(drop=True, inplace=True)
encoded_df.reset_index(drop=True, inplace=True)

# Concatenate the dataframes
new_df = pd.concat([df, encoded_df], axis=1)


new_df=new_df.drop(df.select_dtypes(include=['category']).columns, axis=1)
X = new_df.drop(['id','state_successful'],axis=1)
y = new_df['state_successful']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 5)

# we use below code to find the best parameters for the model, but it takes too long to run, so we just use the best parameters we found

# from sklearn.ensemble import IsolationForest
# from sklearn import model_selection
# model = IsolationForest(random_state=0)

# param_grid = {'n_estimators': [300, 200], 
#               'max_samples': [10, 15, 20], 
#               'contamination': ['auto', 0.0001, 0.005, 0.001, 0.01,0.02], 
#               'max_features': [10, 15,20], 
#               'bootstrap': [True], 
#               'n_jobs': [-1]}

# grid_search = model_selection.GridSearchCV(model, 
#                                            param_grid,
#                                            scoring="neg_mean_squared_error", 
#                                            refit=True,
#                                            cv=10, 
#                                            return_train_score=True)

# best_model = grid_search.fit(X_train, y_train)
# print('Optimum parameters', best_model.best_params_)

from sklearn.ensemble import IsolationForest
iforest = IsolationForest(bootstrap=True,
                          contamination=0.0001, 
                          max_features=10, 
                          max_samples=10, 
                          n_estimators=300, 
                          n_jobs=-1,
                          random_state=0)
y_pred=iforest.fit_predict(X_train)
indices = np.where(y_pred != -1)[0]
X_train, y_train = X_train.iloc[indices, :], y_train.iloc[indices]

#due to imbalanced data, we use SMOTE to oversample the data
from imblearn.over_sampling import SMOTE


smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

from sklearn.ensemble import GradientBoostingClassifier
#below is the best parameters we found for GradientBoostingClassifier
# we use grid search to find the best parameters, but it takes too long to run, so we just use the best parameters we found

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# # Initialize the models
# gbt = GradientBoostingClassifier()

# # Setting up the parameter grid for hyperparameter tuning

# param_grid_gbt = {
#     'n_estimators': [100, 150, 200,300],
#     'learning_rate': [0.1, 0.2, 0.3, 0.15],
#     'max_depth': [3, 5, 4,7],
#     'min_samples_split': [2, 4,6, 7],
#     'random_state': [0]
# }

# from sklearn.model_selection import GridSearchCV
# # Grid Search for GradientBoostingClassifier
# grid_search_gbt = GridSearchCV(estimator=gbt, param_grid=param_grid_gbt, cv=4, n_jobs=-1, verbose=2,scoring='f1_weighted')
# grid_search_gbt.fit(X_train, y_train)

#best_params_gbt = {'learning_rate': 0.1, 'max_depth': 3, 'min_samples_split': 4, 'n_estimators': 200, 'random_state':0} #76.4%/75.4%
#best_params_gbt = {'learning_rate': 0.2, 'max_depth': 3, 'min_samples_split': 2, 'n_estimators': 300,'random_state':0} #77.3%/74.6%
#best_params_gbt = {'learning_rate': 0.2, 'max_depth': 3, 'min_samples_split': 6, 'n_estimators': 100, 'random_state': 0} #76.9%/75.2%
#best_params_gbt = {'learning_rate': 0.01, 'max_depth': 5, 'min_samples_split': 5, 'n_estimators': 200, 'random_state': 0} #73%/72%
#best_params_gbt = {'learning_rate': 0.1, 'max_depth': 5, 'min_samples_split': 6, 'n_estimators': 200, 'random_state': 0} #76.5%/75.5%
best_params_gbt = {'learning_rate': 0.1, 'max_depth': 5, 'min_samples_split': 6, 'n_estimators': 100,'random_state': 0} 
#best_params_gbt = {'learning_rate': 0.1, 'max_depth': 4, 'min_samples_split': 6, 'n_estimators': 300, 'random_state': 0} #76.5
#print(best_params_gbt)

#we choose f1_weighted as the scoring method, because it's a weighted average of the f1 score for each class
new_gbt = GradientBoostingClassifier(**best_params_gbt)

model_gbt = new_gbt.fit(X_train, y_train)


y_pred_gbt = model_gbt.predict(X_test)


from sklearn.metrics import accuracy_score
#Accuracy score for GradientBoostingClassifier
accuracy_gbt = accuracy_score(y_test, y_pred_gbt)
print(accuracy_gbt)

#we test on the grading data
df1 = pd.read_excel("Kickstarter-Grading.xlsx")

#preprocessing
df1['goal_usd'] = df1['goal']*df1['static_usd_rate']
df1 = df1.drop(['name','pledged','currency','state_changed_at','staff_pick','backers_count','usd_pledged','spotlight','state_changed_at_weekday','state_changed_at_month','state_changed_at_day',
              'state_changed_at_yr','state_changed_at_hr','launch_to_state_change_days','deadline','created_at','launched_at','goal'], axis=1)
df1['category'] = df1['category'].fillna('None')

df1 = df1.drop(df1[(df1.state != 'successful') & (df1.state != 'failed')].index)
df1['deadline_yr_category'] = df1['deadline_yr'].astype('category').apply(lambda x: '2014' if x == 2014 else ('2015' if x == 2015 else 'other'))
df1['created_at_yr_category'] = df1['created_at_yr'].astype('category').apply(lambda x: '2014' if x == 2014 else ('2015' if x == 2015 else 'other'))
df1['launched_at_yr_category'] = df1['launched_at_yr'].astype('category').apply(lambda x: '2014' if x == 2014 else ('2015' if x == 2015 else 'other'))


df1['launch_to_deadline_days'] = np.log1p(df1['launch_to_deadline_days'])
df1['create_to_launch_days'] = np.sqrt(df1['create_to_launch_days'])
df1['goal_usd'] = np.cbrt(df1['goal_usd'])


from sklearn.preprocessing import OneHotEncoder

for column in df1.select_dtypes(include=['object','bool']).columns:
    df1[column] = df1[column].astype('category')
    
dummy1 = OneHotEncoder(drop='first', sparse=False)  
dummy_df1 = dummy1.fit_transform(df1.select_dtypes(include=['category']))
columns = dummy1.get_feature_names_out()

# Create a new DataFrame from the one-hot encoded matrix
encoded_df1 = pd.DataFrame(dummy_df1, columns=columns)

# Reset index if necessary before concatenation
df1.reset_index(drop=True, inplace=True)
encoded_df1.reset_index(drop=True, inplace=True)

# Concatenate the dataframes
new_df1 = pd.concat([df1, encoded_df1], axis=1)



new_df1=new_df1.drop(df1.select_dtypes(include=['category']).columns, axis=1)

#Add missing columns in new_df2 with default value of 0
missing_cols = set(new_df.columns) - set(new_df1.columns)
for c in missing_cols:
    new_df1[c] = 0

# Ensure the order of columns in new_df2 is the same as in new_df1
new_df1 = new_df1[new_df.columns]

X1 = new_df1.drop(['id','state_successful'],axis=1)
y1 = new_df1['state_successful']
# Apply the model previously trained to the grading data
y_grading_pred = model_gbt.predict(X1)


#Calculate the accuracy score
print(accuracy_score(y1, y_grading_pred))
# the accuracy score is 0.763