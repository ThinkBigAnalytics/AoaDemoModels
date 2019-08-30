import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from sklearn import metrics

# load data & engineer
train_df = pd.read_csv('iris_train1.csv')
features = 'SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm'.split(',')
X_train = train_df.loc[:, features]
y_train = train_df.Species

print("Starting training...")
# fit model to training data
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
print("Finished training")

# evaluate model against test dataset
predict_df = pd.read_csv('iris_evaluate.csv')
X_predict = predict_df.loc[:, features]
y_test = predict_df.Species
y_predict = knn.predict(X_predict)
print("model accuracy is ", metrics.accuracy_score(y_test, y_predict))

# save model
joblib.dump(knn, 'iris_knn.joblib')
print("Saved trained model")