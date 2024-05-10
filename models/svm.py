from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
class EJM_SVM:
    """
    This class encapsulates an SVM model allowing for both simple SVM
    and primarly the ease of variations in the kernel and C. Scikit-learn
    is utilized to streamline implementation.
    """
    def __init__(self, C=1.0, kernel='rbf'):
        self.model = SVC(C=C, kernel=kernel)

    def train(self, X_train, y_train):
        """_summary_ This method trains the SVM model using the provided training data.

        Args:
            X_train (_type_): _description_ The input features
            y_train (_type_): _description_ The target values
        """
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """_summary_ This method predicts the labels for the given data using the trained SVM model.
        """ 
        return self.model.predict(X)

    def evaluate(self, X, y):
        """This method evaluates the accuracy of the SVM model on the provided dataset."""
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
    
    def decision_function(self, X):
        """Return decision function values."""
        return self.model.decision_function(X)
    
    def get_support_vectors(self):
        """Return support vectors."""
        return self.model.support_vectors_