from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

data = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data[:,2:4], data.target, test_size=0.2,
                                                    random_state=42,stratify=data.target)
knn = KNeighborsClassifier()
parameters = {'n_neighbors': range(1,11)}
knn_best = GridSearchCV(knn, parameters, cv=5)
knn_best.fit(X_train,y_train)

print(knn_best.best_estimator_)

answer = knn_best.predict(X_test)
acc_score = accuracy_score(y_test, answer)

print('Accuracy = ', acc_score)
print(classification_report(y_test, answer))