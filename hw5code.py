import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator
def gini_index(y):
    total_count = len(y)
    if total_count == 0:
        return 0  # Чтобы избежать деления на ноль
    
    p0 = np.sum(y == 0) / total_count
    p1 = np.sum(y == 1) / total_count
    
    # Используем формулу для критерия Джини
    gini = 1 - (p0**2 + p1**2)
    
    return gini

def find_best_split(feature_vector, target_vector):
    # Сортируем по значению признака
    sorted_indices = np.argsort(feature_vector)
    sorted_feature = feature_vector[sorted_indices]
    sorted_target = target_vector[sorted_indices]

    # Находим уникальные значения признака
    unique_values = np.unique(sorted_feature)
    thresholds = (unique_values[:-1] + unique_values[1:]) / 2  # Пороги как средние между соседними значениями

    # Инициализация переменных для хранения значений Джини
    gini_best = float('inf')
    threshold_best = None
    ginis = []

    # Подсчет общего количества объектов
    total_count = len(target_vector)

    # Подсчет количества объектов каждого класса
    total_count_0 = np.sum(target_vector == 0)
    total_count_1 = np.sum(target_vector == 1)

    # Инициализация счетчиков для левой и правой части
    left_count_0 = 0
    left_count_1 = 0

    # Проходим по всем порогам
    for i in range(len(thresholds)):
        threshold = thresholds[i]

        # Обновляем счетчики для левой части
        left_count_0 += (sorted_target[i] == 0)
        left_count_1 += (sorted_target[i] == 1)

        # Количество объектов в левой и правой частях
        left_count = i + 1
        right_count = total_count - left_count

        # Пропускаем, если нет возможности разбить
        if right_count == 0:
            continue

        # Вычисляем Джини для левой и правой частей
        gini_left = gini_index(sorted_target[:left_count])
        gini_right = gini_index(sorted_target[left_count:])

        # Общий критерий Джини
        gini = (left_count / total_count) * gini_left + (right_count / total_count) * gini_right
        ginis.append(gini)

        # Проверяем, является ли это наилучшим значением
        if gini < gini_best:
            gini_best = gini
            threshold_best = threshold

    return thresholds, np.array(ginis), threshold_best, gini_best
class DecisionTree(BaseEstimator):
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        # Проверка на однородность классов
        if len(set(sub_y)) == 1:
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        # Проверка на возможность разбиения
        if sub_X.shape[0] < 2:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, float('inf'), None
        
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                categories = np.unique(sub_X[:, feature])
                feature_vector = np.vectorize(lambda x: np.where(categories == x)[0][0])(sub_X[:, feature])
            else:
                raise ValueError("Unknown feature type")

            # Пропускаем, если нет возможности разбить
            if len(np.unique(feature_vector)) < 2:
                continue

            thresholds, gini, threshold, gini_value = find_best_split(feature_vector, sub_y)

            if gini_value is not None and gini_value < gini_best:
                feature_best = feature
                gini_best = gini_value
                split = feature_vector < threshold
                threshold_best = threshold

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        node["threshold"] = threshold_best
        node["left_child"], node["right_child"] = {}, {}

        # Рекурсивно строим дерево
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[~split], sub_y[~split], node["right_child"])

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]
        
        feature_value = x[node["feature_split"]]
        if self._feature_types[node["feature_split"]] == "real":
            if feature_value < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        elif self._feature_types[node["feature_split"]] == "categorical":
            if feature_value == node["threshold"]:  # Здесь предполагается, что threshold - это значение категории
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
    	self._fit_node(X, y, self._tree)

    def predict(self, X):
    	predicted = []
    	for x in X:
        	predicted.append(self._predict_node(x, self._tree))
    	return np.array(predicted)