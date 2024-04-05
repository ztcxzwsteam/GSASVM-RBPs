import os
import numpy as np
import joblib
from sklearn import svm, ensemble, neighbors
from sklearn.model_selection import ParameterGrid, train_test_split
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, accuracy_score, recall_score, precision_score
from main_bio import train_deep  # Assuming main_bio.train_deep exists and is correctly imported
from multiprocessing import Pool, cpu_count

class ModelSelector:
    def __init__(self):
        self.models = {
            # Model initializations...
        }
        self.param_grids = {
            # Parameter grids...
        }

    def get_models(self, model_list=[]):
        print('选择部分模型...' if model_list else '选择所有模型...')
        return {key: self.models[key] for key in model_list} if model_list else self.models

    def train_with_grid_search(self, X_train, y_train, X_test, y_test, models_dict=None):
        models_dict = models_dict or self.models
        results = {}

        for model_name, model in models_dict.items():
            print(f"Training {model_name}...")
            best_score, best_params = self.evaluate_model(model_name, model, X_train, y_train, X_test, y_test)
            results[model_name] = {'best_score': best_score, 'best_params': best_params}

        return results

    def evaluate_model(self, model_name, model, X_train, y_train, X_test, y_test):
        best_score, best_params = -1, None
        for params in ParameterGrid(self.param_grids[model_name]):
            model.set_params(**params).fit(X_train, y_train)
            score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            if score > best_score:
                best_score, best_params = score, params
        return best_score, best_params

def mult_func(data_file):
    print(f'-----开始：{data_file}-------')
    results = []

    for i in range(1, 6):
        file_path = f'../mid_data/{data_file}_{i}.dict'
        data = joblib.load(file_path)

        X_train, X_test, y_train, y_test = data['x_train'], data['x_test'], data['y_train'].ravel(), data['y_test'].ravel()
        models = ModelSelector().get_models()

        ls_dict = ModelSelector().train_with_grid_search(X_train, y_train, X_test, y_test, models)
        results.append(ls_dict)

    save_path = f'./save_best_file/{data_file}nl_auc_best.list'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(results, save_path)

def train_ML():
    range_list = ['human_m', 'mouse_m']

    with Pool(processes=min(len(range_list), cpu_count() - 1)) as pool:
        pool.map(mult_func, range_list)

if __name__ == '__main__':
    train_deep()
    train_ML()
