import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

train = pd.read_csv("train.csv")
test_init  = pd.read_csv("test.csv")

#exploratory data analysis
print("train set")
print(train.shape)
print(train.head)
print(train.info())
print(train.describe())
print(train.isnull().sum())
print()

print("test set")
print(test_init.isnull().sum())

X = train.drop(["SalePrice"],axis = 1)
y = train["SalePrice"]


features = ["LotArea","MSSubClass","OverallCond","OverallQual","TotalBsmtSF","FullBath","BedroomAbvGr","TotRmsAbvGrd","GarageCars","PoolArea","YearBuilt","YearRemodAdd","HouseStyle"]
X = train[features]
test = test_init[features]
print("new X",X)
print(X.isnull().sum())
print("new test",test)
print(test.isnull().sum())
