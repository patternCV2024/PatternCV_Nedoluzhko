#1
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Функція для створення сітки
def get_grid(data):
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    return np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Ваші дані
x_train_14 = np.array([[21, 46], [6, 33], [19, 22], [30, 49], [46, 36], [35, 26], [43, 11], [6, 5], [19, 20], [42, 41]])
y_train_14 = np.array([-1, -1, -1, 1, -1, -1, 1, 1, -1, 1])

# Ініціалізація та навчання моделі дерева ухвалення рішень
clf_tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=5)
clf_tree.fit(x_train_14, y_train_14)

# Отримання прогнозів та відображення сітки та точок даних
xx, yy = get_grid(x_train_14)
predicted = clf_tree.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.pcolormesh(xx, yy, predicted, cmap='spring', shading='auto')
plt.scatter(x_train_14[:, 0], x_train_14[:, 1], c=y_train_14, s=50, cmap='spring', edgecolors='black', linewidth=1.5)
plt.show()

# Візуалізація дерева
plt.figure(figsize=(12, 8))
plot_tree(clf_tree, filled=True, feature_names=['Feature 1', 'Feature 2'], class_names=['-1', '1'])
plt.show()
