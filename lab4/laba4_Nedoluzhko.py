#1
import numpy as np

# Дані
x_train_14 = np.array([[5, 24],
                       [26, 13],
                       [14, 17],
                       [21, 14],
                       [30, 16],
                       [24, 15],
                       [30, 47],
                       [48, 38],
                       [21, 49],
                       [23, 43]])
y_train_14 = np.array([1, 1, -1, -1, 1, 1, 1, 1, 1, -1])

# Параметри першого класу
mw_1_14, ml_1_14 = np.mean(x_train_14[y_train_14 == 1], axis=0)
sw_1_14, sl_1_14 = np.var(x_train_14[y_train_14 == 1], axis=0)

# Параметри другого класу
mw1_14, ml1_14 = np.mean(x_train_14[y_train_14 == -1], axis=0)
sw1_14, sl1_14 = np.var(x_train_14[y_train_14 == -1], axis=0)

print('Середнє для класу 1: ', mw_1_14, ml_1_14)
print('Середнє для класу -1:', mw1_14, ml1_14)
print('Дисперсії для класу 1:', sw_1_14, sl_1_14)
print('Дисперсії для класу -1:', sw1_14, sl1_14)

# Новий приклад
x_14 = np.array([15, 20])  # Довжина, ширина жука

# Класифікатор для першого класу
a_1_14 = -(x_14[0] - ml_1_14) ** 2 / (2 * sl_1_14) - (x_14[1] - mw_1_14) ** 2 / (2 * sw_1_14)
# Класифікатор для другого класу
a1_14 = -(x_14[0] - ml1_14) ** 2 / (2 * sl1_14) - (x_14[1] - mw1_14) ** 2 / (2 * sw1_14)

# Обираємо клас
y_14 = np.argmax([a_1_14, a1_14]) + 1

print('Номер класу (1 - гусениця, -1 - божа корівка):', y_14)

#2
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

# Вхідні параметри для першого кластеру
rho1 = 0.9
sigma_x1_squared = 0.8
mu_x1 = [1, -1]
sigma_y1_squared = 1.1
mu_y1 = [1, -1]

# Вхідні параметри для другого кластеру
rho2 = 0.7
sigma_x2_squared = 2.0
mu_x2 = [0, 3]
sigma_y2_squared = 2.0
mu_y2 = [0, 3]

# моделювання навчальної вибірки для першого кластеру
N = 1000
x1 = np.random.multivariate_normal(mu_x1, [[sigma_x1_squared, rho1], [rho1, sigma_y1_squared]], N).T

# моделювання навчальної вибірки для другого кластеру
x2 = np.random.multivariate_normal(mu_y2, [[sigma_x2_squared, rho2], [rho2, sigma_y2_squared]], N).T

# обчислення оцінок середнього та коваріаційних матриць для кожного кластеру
mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)

a = (x1.T - mm1).T
VV1 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N],
                [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N]])

a = (x2.T - mm2).T
VV2 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N],
                [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N]])

# модель гауссівського баєсівського класифікатора
Py1, L1 = 0.5, 1  # ймовірності появи класів
Py2, L2 = 1 - Py1, 1  # та величини штрафів невірної класифікації

b = lambda x, v, m, l, py: np.log(l * py) - 0.5 * (x - m) @ np.linalg.inv(v) @ (x - m).T - 0.5 * np.log(
    np.linalg.det(v))

x = np.array([-2, -2])  # вхідний вектор у форматі (x, y)
a = np.argmax([b(x, VV1, mm1, L1, Py1), b(x, VV2, mm2, L2, Py2)])  # класифікатор
print(a)

# виведення графіків
plt.figure(figsize=(4, 4))
plt.title(f"Кореляції: rho1 = {rho1}, rho2 = {rho2}")
plt.scatter(x1[0], x1[1], s=10)
plt.scatter(x2[0], x2[1], s=10)
plt.show()

#3

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

# Вхідні параметри для першого кластеру
rho1 = 0.9
sigma_x1_squared = 0.8
mu_x1 = [1, -1]
sigma_y1_squared = 1.1
mu_y1 = [1, -1]

# Вхідні параметри для другого кластеру зі зміненим знаком кореляції
rho2 = -0.7  # Змінено на протилежне число
sigma_x2_squared = 2.0
mu_x2 = [0, 3]
sigma_y2_squared = 2.0
mu_y2 = [0, 3]

# моделювання навчальної вибірки для першого кластеру
N = 1000
x1 = np.random.multivariate_normal(mu_x1, [[sigma_x1_squared, rho1], [rho1, sigma_y1_squared]], N).T

# моделювання навчальної вибірки для другого кластеру зі зміненим знаком кореляції
x2 = np.random.multivariate_normal(mu_y2, [[sigma_x2_squared, rho2], [rho2, sigma_y2_squared]], N).T

# обчислення оцінок середнього та коваріаційних матриць для кожного кластеру
mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)

a = (x1.T - mm1).T
VV1 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N],
                [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N]])

a = (x2.T - mm2).T
VV2 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N],
                [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N]])

# модель гауссівського баєсівського класифікатора
Py1, L1 = 0.5, 1  # ймовірності появи класів
Py2, L2 = 1 - Py1, 1  # та величини штрафів невірної класифікації

b = lambda x, v, m, l, py: np.log(l * py) - 0.5 * (x - m) @ np.linalg.inv(v) @ (x - m).T - 0.5 * np.log(
    np.linalg.det(v))

x = np.array([-2, -2])  # вхідний вектор у форматі (x, y)
a = np.argmax([b(x, VV1, mm1, L1, Py1), b(x, VV2, mm2, L2, Py2)])  # класифікатор
print(a)

# виведення графіків
plt.figure(figsize=(4, 4))
plt.title(f"Кореляції: rho1 = {rho1}, rho2 = {rho2}")
plt.scatter(x1[0], x1[1], s=10)
plt.scatter(x2[0], x2[1], s=10)
plt.show()


#4

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

# Вхідні параметри для першого кластеру
rho1 = 0.9
sigma_x1_squared = 0.8
mu_x1 = [1, -1]
sigma_y1_squared = 1.1
mu_y1 = [1, -1]

# Вхідні параметри для другого кластеру
rho2 = -0.7
sigma_x2_squared = 2.0
mu_x2 = [0, 3]
sigma_y2_squared = 2.0
mu_y2 = [0, 3]

# Вхідні параметри для третього кластеру
rho3 = 0.8
sigma_x3_squared = 1.5
mu_x3 = [-3, -2]
sigma_y3_squared = 0.9
mu_y3 = [-3, -2]

# Кількість точок у кожному кластері
N = 1000

# моделювання навчальної вибірки для першого кластеру
x1 = np.random.multivariate_normal(mu_x1, [[sigma_x1_squared, rho1], [rho1, sigma_y1_squared]], N).T

# моделювання навчальної вибірки для другого кластеру
x2 = np.random.multivariate_normal(mu_x2, [[sigma_x2_squared, rho2], [rho2, sigma_y2_squared]], N).T

# моделювання навчальної вибірки для третього кластеру
x3 = np.random.multivariate_normal(mu_x3, [[sigma_x3_squared, rho3], [rho3, sigma_y3_squared]], N).T

# обчислення оцінок середнього та коваріаційних матриць для кожного кластеру
mm1 = np.mean(x1.T, axis=0)
mm2 = np.mean(x2.T, axis=0)
mm3 = np.mean(x3.T, axis=0)

a = (x1.T - mm1).T
VV1 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N],
                [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N]])

a = (x2.T - mm2).T
VV2 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N],
                [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N]])

a = (x3.T - mm3).T
VV3 = np.array([[np.dot(a[0], a[0]) / N, np.dot(a[0], a[1]) / N],
                [np.dot(a[1], a[0]) / N, np.dot(a[1], a[1]) / N]])

# модель гауссівського баєсівського класифікатора
Py1, L1 = 0.33, 1  # ймовірності появи класів
Py2, L2 = 0.33, 1
Py3, L3 = 0.34, 1
Py = [Py1, Py2, Py3]
L = [L1, L2, L3]

b = lambda x, v, m, l, py: np.log(l * py) - 0.5 * (x - m) @ np.linalg.inv(v) @ (x - m).T - 0.5 * np.log(
    np.linalg.det(v))

x = np.array([-2, -2])  # вхідний вектор у форматі (x, y)
a = np.argmax([b(x, VV1, mm1, L1, Py1), b(x, VV2, mm2, L2, Py2), b(x, VV3, mm3, L3, Py3)])  # класифікатор
print(a)

# виведення графіків
plt.figure(figsize=(6, 6))
plt.title("Класифікація для трьох кластерів")
plt.scatter(x1[0], x1[1], s=10, label='Кластер 1')
plt.scatter(x2[0], x2[1], s=10, label='Кластер 2')
plt.scatter(x3[0], x3[1], s=10, label='Кластер 3')
plt.legend()
plt.show()
