import torch


def standard_scaler(tensor, dim=0):
    mean = torch.mean(tensor, dim=dim, keepdim=True)
    std = torch.std(tensor, dim=dim, keepdim=True)
    return (tensor - mean) / std


def minmax_scaler(tensor, dim=0):
    min_ = torch.min(tensor, dim=dim, keepdim=True).values
    max_ = torch.max(tensor, dim=dim, keepdim=True).values
    return (tensor - min_) / (max_ - min_)

class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.min_ = None
        self.max_ = None
        self.feature_range = feature_range

    def fit(self, X):
        self.min_ = torch.min(X, dim=0).values
        self.max_ = torch.max(X, dim=0).values

    def transform(self, X):
        if self.min_ is None or self.max_ is None:
            raise RuntimeError(
                "The MinMaxScaler instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        data_range = self.max_ - self.min_
        scaled_X = (X - self.min_) / data_range
        scaled_X = scaled_X * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
        return scaled_X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        self.mean_ = torch.mean(X, dim=0)
        self.std_ = torch.std(X, dim=0)

    def transform(self, X):
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError(
                "The StandardScaler instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class RobustScaler:
    def __init__(self):
        self.median_ = None
        self.iqr_ = None

    def fit(self, X):
        # Calculate the median and IQR (Interquartile Range)
        self.median_ = torch.median(X, dim=0).values
        q75, q25 = torch.quantile(X, 0.75, dim=0), torch.quantile(X, 0.25, dim=0)
        self.iqr_ = q75 - q25

    def transform(self, X):
        if self.median_ is None or self.iqr_ is None:
            raise RuntimeError(
                "The RobustScaler instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        # Perform the scaling
        return (X - self.median_) / self.iqr_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)