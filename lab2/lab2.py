import numpy as np
import matplotlib.pyplot as plt

# Дані
x_train_14 = np.array([[23, 36],
                       [31, 28],
                       [31, 20],
                       [37, 37],
                       [5, 42],
                       [15, 49],
                       [50, 39],
                       [27, 28],
                       [18, 45],
                       [5, 38]])
x_train_14_bias = np.c_[x_train_14, np.ones(len(x_train_14))]  # Додаємо зміщення для кожного прикладу
y_train_14 = np.array([1, 1, 1, 1, 1, 1, 1, 1, -1, 1])

# Розрахунок коефіцієнтів
pt = np.sum([x * y for x, y in zip(x_train_14_bias, y_train_14)], axis=0)  # Обчислення підсумку
xxt = np.sum([np.outer(x, x) for x in x_train_14_bias], axis=0)  # Обчислення підсумку зовнішнього добутку
w = np.dot(pt, np.linalg.inv(xxt))  # Обчислення вагових коефіцієнтів

# Формування координат для лінії розділення (вертикальна лінія)
x_line = [max(x_train_14[:, 0]), max(x_train_14[:, 0])]
y_line = [min(x_train_14[:, 1]), max(x_train_14[:, 1])]

# Формування точок для класу 1 та класу -1
x_minus_1 = x_train_14[y_train_14 == -1]
x_1 = x_train_14[y_train_14 == 1]

# Відображення графіку
plt.figure(figsize=(8, 6))
plt.scatter(x_minus_1[:, 0], x_minus_1[:, 1], color='blue', label='Клас -1')
plt.scatter(x_1[:, 0], x_1[:, 1], color='red', label='Клас 1')
plt.plot(x_line, y_line, color='green', label='Лінія розділення')

plt.xlabel("Ознака 1")
plt.ylabel("Ознака 2")
plt.title("Бінарний МНК-класифікатор")
plt.legend()
plt.grid(True)
plt.show()
