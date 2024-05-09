from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class EJM_SVM:
    def __init__(self, C=1.0, kernel='rbf'):
        self.model = SVC(C=C, kernel=kernel)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        return accuracy_score(y, predictions)