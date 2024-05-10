from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

class EJM_LDA:
    """
    This class wraps the sklearn LinearDiscriminantAnalysis model to provide a more straightforward
    interface for training and predicting labels based on linear discriminant analysis.
    """
    def __init__(self):
        """
        Initializes the Linear Discriminant Analysis (LDA) model.
        """
        self.model = LinearDiscriminantAnalysis()

    def train(self, X_train, y_train):
        """
        Trains the LDA model using the provided training data.

        Args:
            X_train (array-like): Feature dataset for training.
            y_train (array-like): Labels corresponding to X_train.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """
        Predicts the labels for the given data using the trained LDA model.

        Args:
            X (array-like): Feature dataset for which labels need to be predicted.

        Returns:
            array: Predicted labels for the input data.
        """
        return self.model.predict(X)

    def evaluate(self, X, y):
        """
        Evaluates the accuracy of the LDA model on the provided dataset.

        Args:
            X (array-like): Feature dataset for evaluation.
            y (array-like): True labels for X.

        Returns:
            float: The accuracy of the model on the provided data.
        """
        predictions = self.predict(X)
        return accuracy_score(y, predictions)