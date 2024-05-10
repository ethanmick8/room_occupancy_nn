from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
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
    
    def decision_function(self, X):
        """Return decision function values."""
        return self.model.decision_function(X)
    
    def get_support_vectors(self):
        """Return support vectors."""
        return self.model.support_vectors_