"""
CompLeak attack models for membership inference (LR, RF, DT, MLP).
Used by CompLeakNR-related attacks.
"""
import joblib
import random
import numpy as np
from sklearn.metrics import roc_curve, auc
from attack.utils import calculate_tpr_at_fpr
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing


class DT:
    def __init__(self):
        self.model = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)

    def train_model(self, train_x, train_y, save_name=None):
        self.model.fit(train_x, train_y)
        if save_name is not None:
            joblib.dump(self.model, save_name, compress=9)

    def load_model(self, save_name):
        self.model = joblib.load(save_name)
        return self.model

    def predict_proba(self, test_x):
        return self.model.predict_proba(test_x)

    def test_model_acc(self, test_x, test_y):
        pred_y = self.model.predict(test_x)
        pred_proba = self.model.predict_proba(test_x)
        return accuracy_score(test_y, pred_y), pred_y, pred_proba

    def test_model_auc(self, test_x, test_y):
        pred_y = self.model.predict_proba(test_x)
        fpr, tpr, _ = roc_curve(test_y, pred_y[:, 1])
        tpr_at_001 = calculate_tpr_at_fpr(test_y, pred_y[:, 1], 0.001)
        tpr_at_01 = calculate_tpr_at_fpr(test_y, pred_y[:, 1], 0.01)
        auc1 = auc(fpr, tpr)
        return auc1, tpr_at_001, tpr_at_01


class RF:
    def __init__(self, min_samples_leaf=30):
        self.model = RandomForestClassifier(random_state=0, n_estimators=500, min_samples_leaf=min_samples_leaf)

    def train_model(self, train_x, train_y, save_name=None):
        self.model.fit(train_x, train_y)
        if save_name is not None:
            joblib.dump(self.model, save_name, compress=9)

    def load_model(self, save_name):
        self.model = joblib.load(save_name)
        return self.model

    def predict_proba(self, test_x):
        return self.model.predict_proba(test_x)

    def test_model_acc(self, test_x, test_y):
        pred_y = self.model.predict(test_x)
        pred_proba = self.model.predict_proba(test_x)
        return accuracy_score(test_y, pred_y), pred_y, pred_proba

    def test_model_auc(self, test_x, test_y):
        pred_y = self.model.predict_proba(test_x)
        fpr, tpr, _ = roc_curve(test_y, pred_y[:, 1])
        tpr_at_001 = calculate_tpr_at_fpr(test_y, pred_y[:, 1], 0.001)
        tpr_at_01 = calculate_tpr_at_fpr(test_y, pred_y[:, 1], 0.01)
        auc1 = auc(fpr, tpr)
        return auc1, tpr_at_001, tpr_at_01


class MLP:
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        random.seed(random_seed)
        self.model = MLPClassifier(early_stopping=True, learning_rate_init=0.01, random_state=random_seed)

    def train_model(self, train_x, train_y, save_name=None):
        self.model.fit(train_x, train_y)
        if save_name is not None:
            joblib.dump(self.model, save_name, compress=9)

    def load_model(self, save_name):
        self.model = joblib.load(save_name)
        return self.model

    def predict_proba(self, test_x):
        return self.model.predict_proba(test_x)

    def test_model_acc(self, test_x, test_y):
        pred_y = self.model.predict(test_x)
        pred_proba = self.model.predict_proba(test_x)
        return accuracy_score(test_y, pred_y), pred_y, pred_proba

    def test_model_auc(self, test_x, test_y):
        pred_y = self.model.predict_proba(test_x)
        fpr, tpr, _ = roc_curve(test_y, pred_y[:, 1])
        tpr_at_001 = calculate_tpr_at_fpr(test_y, pred_y[:, 1], 0.001)
        tpr_at_01 = calculate_tpr_at_fpr(test_y, pred_y[:, 1], 0.01)
        auc1 = auc(fpr, tpr)
        return auc1, tpr_at_001, tpr_at_01


class LR:
    def __init__(self):
        self.model = LogisticRegression(random_state=0, solver='lbfgs', max_iter=400)

    def train_model(self, train_x, train_y, save_name=None):
        self.scaler = preprocessing.StandardScaler().fit(train_x)
        self.model.fit(self.scaler.transform(train_x), train_y)
        if save_name is not None:
            joblib.dump(self.model, save_name, compress=9)

    def load_model(self, save_name):
        self.model = joblib.load(save_name)
        return self.model

    def predict_proba(self, test_x):
        return self.model.predict_proba(self.scaler.transform(test_x))

    def test_model_acc(self, test_x, test_y):
        pred_proba = self.model.predict_proba(self.scaler.transform(test_x))
        pred_y = self.model.predict(self.scaler.transform(test_x))
        return accuracy_score(test_y, pred_y), pred_y, pred_proba

    def test_model_auc(self, test_x, test_y):
        pred_y = self.model.predict_proba(self.scaler.transform(test_x))
        fpr, tpr, _ = roc_curve(test_y, pred_y[:, 1])
        tpr_at_001 = calculate_tpr_at_fpr(test_y, pred_y[:, 1], 0.001)
        tpr_at_01 = calculate_tpr_at_fpr(test_y, pred_y[:, 1], 0.01)
        auc1 = auc(fpr, tpr)
        return auc1, tpr_at_001, tpr_at_01
