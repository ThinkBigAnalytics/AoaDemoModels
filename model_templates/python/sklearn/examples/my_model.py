import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from sklearn import metrics
from sklearn.model_selection import train_test_split

# load data & engineer
df = pd.read_csv('https://datahub.io/machine-learning/iris/r/iris.csv')
# split dataset
train_df, predict_df = train_test_split(df, test_size = 0.5) 
features = 'sepallength,sepalwidth,petallength,petalwidth'.split(',')
X_train = train_df.loc[:, features]
y_train = train_df['class']

print("Starting training...")
# fit model to training data
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
print("Finished training")

# evaluate model against test dataset
X_predict = predict_df.loc[:, features]
y_test = predict_df['class']
y_predict = knn.predict(X_predict)
print("model accuracy is ", metrics.accuracy_score(y_test, y_predict))

# save model
joblib.dump(knn, 'iris_knn.joblib')
print("Saved trained model")