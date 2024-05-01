#1
import numpy as np
import matplotlib.pyplot as plt

# Ваші дані
x_train_14 = np.array([[21, 46],
                      [6, 33],
                      [19, 22],
                      [30, 49],
                      [46, 36],
                      [35, 26],
                      [43, 11],
                      [6, 5],
                      [19, 20],
                      [42, 41]])

y_train_14 = np.array([-1, -1, -1, 1, -1, -1, 1, 1, -1, 1])

# Адаптація даних до формату вихідного алгоритму
x = x_train_14

# Обчислення середніх та дисперсій
M = np.mean(x, axis=0)
D = np.var(x, axis=0)

# Визначення кількості кластерів
K = len(np.unique(y_train_14))

# Генерація початкових центрів кластерів
ma = [np.random.normal(M, np.sqrt(D / 10), 2) for n in range(K)]

# Функція для обчислення евклідової метрики
ro = lambda x_vect, m_vect: np.mean((x_vect - m_vect) ** 2)

# Колірування кластерів
COLORS = ('green', 'blue', 'brown', 'black')

# Ініціалізація графіка
plt.ion()

n = 0
while n < 10:
    X = [[] for i in range(K)]

    for x_vect, y_label in zip(x, y_train_14):
        r = [ro(x_vect, m) for m in ma]
        X[np.argmin(r)].append(x_vect)

    ma = [np.mean(xx, axis=0) for xx in X]

    plt.clf()

    # Відображення кластерів
    for i in range(K):
        xx = np.array(X[i]).T
        plt.scatter(xx[0], xx[1], s=10, color=COLORS[i])

    # Відображення центрів кластерів
    mx = [m[0] for m in ma]
    my = [m[1] for m in ma]
    plt.scatter(mx, my, s=50, color='red')

    plt.draw()
    plt.gcf().canvas.flush_events()

    n += 1

plt.ioff()
plt.show()

#2

from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

# Дані
x_train_14 = np.array([[21, 46], [6, 33], [19, 22], [30, 49], [46, 36], [35, 26], [43, 11], [6, 5], [19, 20], [42, 41]])
y_train_14 = np.array([-1, -1, -1, 1, -1, -1, 1, 1, -1, 1])

# Розмір епсилон-околу
eps = 10
# Мінімальна кількість об'єктів для повного епсилон-околу
m = 2

# Використання алгоритму DBSCAN
dbscan = DBSCAN(eps=eps, min_samples=m)
dbscan.fit(x_train_14)
labels = dbscan.labels_

# Вивід кластерів
unique_labels = np.unique(labels)
for label in unique_labels:
    if label == -1:
        plt.scatter(x_train_14[labels == label][:, 0], x_train_14[labels == label][:, 1], color='k', label='Кластер 0')
    else:
        plt.scatter(x_train_14[labels == label][:, 0], x_train_14[labels == label][:, 1], label='Кластер 1')

plt.legend()
plt.show()

#3

from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# вхідні зображення для кластеризації
x_train_6 = np.array([[49, 21], [5, 5], [37, 32], [21, 25], [34, 28], [44, 35], [39, 41], [17, 45], [31, 24]])

NC = 2  # максимальна кількість кластерів (кінцевих)

# використання алгоритму KMeans
kmeans = KMeans(n_clusters=NC)
kmeans.fit(x_train_6)
x_pr = kmeans.predict(x_train_6)

# відображення результату кластеризації
for c, n in zip(cycle('bgrcmykgrcmykgrcmykgrcmykgrcmykgrcmyk'), range(NC)):
    clst = x_train_6[x_pr == n].T
    plt.scatter(clst[0], clst[1], s=10, color=c)

plt.show()
