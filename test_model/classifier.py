from sklearn.neural_network import MLPClassifier
import joblib

X = [[0., 0.], [1., 1.]]
y = [[0, 1], [1, 1]]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(15,),
                    random_state=1)
clf.fit(X, y)

filename = '../api/models/ma_model.pcl'
joblib.dump(clf, filename)

print(joblib.load(filename).predict(X))
