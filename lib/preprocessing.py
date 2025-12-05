import numpy as np

class StandardScaler:
    """Normalizes each feature to zero mean & unit variance."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        # Compute per-feature mean and std
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-8  # avoid /0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class MinMaxScaler:
    """Scales data to a given range (e.g., [-1,1])."""

    def __init__(self, feature_range=(-1, 1)):
        self.min_ = None
        self.max_ = None
        self.feature_range = feature_range

    def fit(self, X):
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        return self

    def transform(self, X):
        a, b = self.feature_range
        return (b - a) * (X - self.min_) / (self.max_ - self.min_ + 1e-8) + a

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def shuffle_data(X, y):
    """Randomly shuffle dataset preserving correspondence."""
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def train_test_split(X, y, test_ratio=0.2):
    """Simple train/test splitter."""
    N = len(X)
    split_idx = int(N * (1 - test_ratio))
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]
