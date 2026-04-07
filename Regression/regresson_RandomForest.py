import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error

#load data
train= pd.read_csv("train.csv")
test_initial = pd.read_csv("test.csv")

#exploratory data analysis
train.info()
#print(train.shape)
#print(train.describe())
#print(train.head())
#print(train.isnull().sum())

print(test_initial.isnull().sum())

#X = train[['emotional_charge_0','emotional_charge_1','emotional_charge_2','groove_efficiency_0','groove_efficiency_1','groove_efficiency_2','artist_count','tonal_mode_0','tonal_mode_1','tonal_mode_2','beat_frequency_0','beat_frequency_1','beat_frequency_2','harmonic_scale_0','harmonic_scale_1','harmonic_scale_2']]
features = [col for col in train.columns if 'emotional' in col or 'groove' in col or 'beat' in col or 'tonal' in col or 'harmonic' in col or 'artist' in col  ]
print(features)

X = train[features]
y = train['target']
#test =[col for col in test_initial.columns if 'emotional' in col or 'groove' in col or 'beat' in col or 'tonal' in col or 'harmonic' in col or 'artist' in col  ] 
test = test_initial[features]
ID = test_initial['id']

full_data = pd.concat([X,test])
full_data = pd.get_dummies(full_data,drop_first = True)
X = full_data[:len(X)]
test = full_data[len(X):]

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size = 0.2,random_state = 42)

rf = RandomForestRegressor(
        n_jobs = -1,
        max_depth = 10,
        n_estimators = 500,
        min_samples_leaf = 3,
        min_samples_split = 5,
        random_state = 42
    )

rf.fit(X_train,y_train)

predict_train = rf.predict(X_val)

mse = mean_squared_error(y_val,predict_train)
print(mse)
rmse = root_mean_squared_error(y_val,predict_train)
print(rmse)

rf.fit(X,y)
predict_final = rf.predict(test)

submission = pd.DataFrame({
    "id" :ID,
    "target" : predict_final,
    }
    )
submission.to_csv('submission.csv',index = False)




