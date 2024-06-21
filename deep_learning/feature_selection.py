import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest


def rough_set_feature_selection(X, y, top=100):
    # Assuming binary classification for simplicity
    features_importance = []
    for i in range(X.shape[1]):  # iterate over each feature
        feature_values = X[:, i]
        # Calculate the decision table: group by feature value and calculate the majority class
        decision_table = {}
        for value, decision in zip(feature_values, y):
            if value not in decision_table:
                decision_table[value] = [0, 0]
            decision_table[value][decision] += 1

        # Calculate the contribution of each feature using simple majority vote count
        majority_count = sum(max(counts) for counts in decision_table.values())
        features_importance.append(majority_count)

    # Select the top 5 features with the highest majority vote count
    top_features = np.argsort(features_importance)[::-1][:top]
    return X[top_features], top_features


def bayesian_network_feature_selection(X, y, top=100):
    # 实例化贝叶斯分类器
    gnb = GaussianNB()

    # Fit the classifier to the data to assess feature importances
    gnb.fit(X, y)

    # 获取每个特征的对数概率差的绝对值作为特征重要性
    # 这里我们简单地使用贝叶斯概率估计中的类概率（特征重要性的代理）
    abs_diff_class_probs = np.abs(gnb.theta_[0] - gnb.theta_[1])

    # 获取重要性得分最高的特征索引
    top_features_indices = np.argsort(abs_diff_class_probs)[::-1][:top]

    # 根据选定的特征索引筛选数据
    X_new = X[:, top_features_indices]
    return X_new, top_features_indices

def fuzzy_logic_feature_selection(X, y, top=100):
    # Compute the absolute correlation coefficient with the target
    correlation = np.array([np.abs(np.corrcoef(X[:, i], y)[0, 1]) for i in range(X.shape[1])])
    top_features = np.argsort(correlation)[::-1][:top]
    X_new = X[:, top_features]
    return X_new, top_features



if __name__ == '__main__':
    pass