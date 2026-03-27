import pandas as pd
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score , f1_score

train=pd.read_csv("train.csv")
test_init = pd.read_csv("test.csv")
'''
print(train.shape)
print(train.head())
print(train.describe())
print(train.info())
print(train.isnull().sum())
'''
#print(test.isnull().sum())

X=train.drop(["PassengerId","Name","Transported","Cabin"],axis =1)
y=train["Transported"]
test = test_init.drop(["PassengerId","Name","Cabin"],axis =1)
Id = test_init["PassengerId"]


#fillna
#numerical
num_cols = X.select_dtypes(include = "number").columns
num_median = X[num_cols].median()
X[num_cols] = X[num_cols].fillna(num_median)
test[num_cols] = test[num_cols].fillna(num_median)

#categorical
cat_cols = X.select_dtypes(include = ["object","string"]).columns
cat_mode = X[cat_cols].mode().iloc[0]
X[cat_cols] = X[cat_cols].fillna(cat_mode)
test[cat_cols] = test[cat_cols].fillna(cat_mode)

#print(X.isnull().sum())
#print(test.isnull().sum())

#Encoding

full_data = pd.concat([X,test])
full_data = pd.get_dummies(full_data,drop_first = True)
X=full_data[:len(X)]
test=full_data[len(X):]
print("Here")

rf = RandomForestClassifier(
    n_jobs = -1,
    n_estimators=500,
    max_depth=5,
    min_samples_leaf=3,
    min_samples_split=5,
    random_state=42
    )

score = cross_val_score(rf,X,y,cv=5,scoring="accuracy")
print(score.mean())

'''
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,random_state=42)
rf.fit(X_train,y_train)
predict_t=rf.predict(X_val)
accuracy = accuracy_score(y_val,predict_t)
print("accuracy : ",accuracy)
'''

rf.fit(X,y)
print("trained")
predict = rf.predict(test)

Submission = pd.DataFrame(
    {"PassengerId":Id,
     "Transported":predict
        }
    )
Submission.to_csv("submission2.csv",index=False)
print("Done")





