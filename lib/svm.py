import numpy as np

class SVM:
    """Multi-class SVM using One-vs-Rest with linear kernel."""
    def __init__(self, learning_rate=0.001, lambda_param=0.01, num_epochs=1000, num_classes=None):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.num_epochs = num_epochs
        self.num_classes = num_classes
        self.weights = None  # shape: (num_classes, n_features)
        self.bias = None     # shape: (num_classes,)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.num_classes = self.num_classes or len(np.unique(y))
        self.weights = np.zeros((self.num_classes, n_features))
        self.bias = np.zeros(self.num_classes)

        # One-vs-Rest training
        for cls in range(self.num_classes):
            y_binary = np.where(y == cls, 1, -1)  # +1 for current class, -1 for others
            w = self.weights[cls]
            b = self.bias[cls]

            for epoch in range(self.num_epochs):
                for idx, x_i in enumerate(X):
                    condition = y_binary[idx] * (np.dot(w, x_i) + b)
                    if condition >= 1:
                        # No hinge loss
                        w -= self.lr * (2 * self.lambda_param * w)
                    else:
                        # Hinge loss gradient
                        w -= self.lr * (2 * self.lambda_param * w - np.dot(x_i, y_binary[idx]))
                        b -= self.lr * y_binary[idx]

            self.weights[cls] = w
            self.bias[cls] = b

    def predict(self, X):
        # Compute scores for each class
        scores = np.dot(X, self.weights.T) + self.bias
        # Pick the class with the highest score
        return np.argmax(scores, axis=1)

# ----------------- Metrics -----------------
def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true, y_pred):
    num_classes = len(np.unique(y_true))
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def classification_report(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    num_classes = cm.shape[0]
    report = {}
    for cls in range(num_classes):
        tp = cm[cls, cls]
        fn = np.sum(cm[cls, :]) - tp
        fp = np.sum(cm[:, cls]) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = np.sum(cm[cls, :])
        report[cls] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": support
        }
    return report