# Code you have previously used to load data
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)




#Return mean absolute error with changes to max leaf node parameter
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):

	model= RandomForestRegressor(n_estimators=10,max_leaf_nodes=max_leaf_nodes, random_state=1)
	model.fit(train_X,train_y)
	predict_val=model.predict(val_X)
	mae=mean_absolute_error(predict_val,val_y)

	return mae
#Return mean absolute error with changes to minimum sampes split parameter
def get_mae_2(min_split, train_X, val_X, train_y, val_y):
 

	model= RandomForestRegressor(min_samples_split=min_split,random_state=1)
	model.fit(train_X,train_y)
	predict_val=model.predict(val_X)
	mae=mean_absolute_error(predict_val,val_y)

	return mae

# Path of the file to read. 
train_path= 'train.csv'
#test_path='test.csv'

train_data = pd.read_csv(train_path)
#test_data=pd.read_csv(test_path)

#This can be used by configuring which features you would want to use, however,
#non integer data types need to be accounted for, empty spaces, or NaN values
#NaN values can easily be fixed by using .isnan( # ) and entering the mean value of that column
all_features=train_data.columns[1:80].tolist()

#Target value is the price of the home we want to predict
y_train = train_data.SalePrice
#features that make the most sense
starting_features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X_train = train_data[starting_features]

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X_train, y_train, random_state=1)

#comparative value for minimum number
min=999_999
for i in range(2,100):
	mae=get_mae_2(i,train_X, val_X, train_y, val_y)

	if mae<min:
		min=mae
		index=i
		
	best_min_split=index #number of 
	best_mae=min


print("Validation MAE for RFR with  {:,.0f} max leaf nodes: {:,.0f}".format(best_min_split,best_mae))

"""import test data and compare it with fitted RFR with best max leaf nodes
#y_test=test_data.SalePrice
#X_test = test_data[starting_features]

#final_mae=get_mae(best_max_leaf_nodes, X_train, X_test, y_train, y_test)
#print("MAE after training a random forest regressor with best fitted max leaf: {:,.0f}".format(final_mae))
"""


# Specify Model
"""train_house_model = RandomForestRegressor(random_state=1)
# Fit Model
train_house_model.fit(train_X, train_y)

# Make validation predictions and calculate mean absolute error
val_predictions = train_house_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for RFR when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))
"""
"""
# Using best value for max_leaf_nodes
iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
iowa_model.fit(train_X, train_y)
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))

# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)


print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))"""