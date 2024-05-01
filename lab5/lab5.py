import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# Ваші дані
x_train_14 = np.array([[21, 46], [6, 33], [19, 22], [30, 49], [46, 36], [35, 26], [43, 11], [6, 5], [19, 20], [42, 41]])
y_train_14 = np.array([-1, -1, -1, 1, -1, -1, 1, 1, -1, 1])

# Лінійний SVM
clf_linear = svm.SVC(kernel='linear')
clf_linear.fit(x_train_14, y_train_14)

# Нелінійний SVM з ядром Radial Basis Function (RBF)
clf_nonlinear = svm.SVC(kernel='rbf', gamma='auto')
clf_nonlinear.fit(x_train_14, y_train_14)

# Візуалізація результатів
plt.figure(figsize=(12, 5))

# Графік результатів лінійного SVM
plt.subplot(1, 2, 1)
plt.scatter(x_train_14[:, 0], x_train_14[:, 1], c=y_train_14, cmap=plt.cm.coolwarm, s=30)
plt.scatter(clf_linear.support_vectors_[:, 0], clf_linear.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k')
plt.title('Лінійний SVM')
plt.xlabel('Ознака 1')
plt.ylabel('Ознака 2')

# Графік результатів нелінійного SVM з ядром RBF
plt.subplot(1, 2, 2)
plt.scatter(x_train_14[:, 0], x_train_14[:, 1], c=y_train_14, cmap=plt.cm.coolwarm, s=30)
plt.scatter(clf_nonlinear.support_vectors_[:, 0], clf_nonlinear.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k')
plt.title('Нелінійний SVM з ядром RBF')
plt.xlabel('Ознака 1')
plt.ylabel('Ознака 2')

plt.tight_layout()
plt.show()
