import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Ваші дані
x_train_14 = np.array([[21, 46], [6, 33], [19, 22], [30, 49], [46, 36], [35, 26], [43, 11], [6, 5], [19, 20], [42, 41]])
y_train_14 = np.array([-1, -1, -1, 1, -1, -1, 1, 1, -1, 1])

# Тестовий набір даних
x_test_14 = np.array([[35, 10], [36, 39], [41, 9], [43, 19], [32, 42], [38, 48], [12, 39], [22, 45], [20, 29], [48, 21]])
y_test_14 = np.array([-1, -1, -1, 1, 1, 1, -1, 1, -1, 1])

# Визначення класифікатора та навчання моделі
k = 3  # Кількість найближчих сусідів
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(x_train_14, y_train_14)

# Прогнозування класів для тестових даних
y_pred_14 = knn.predict(x_test_14)

# Оцінка точності класифікації
accuracy_14 = accuracy_score(y_test_14, y_pred_14)
print(f"Точність класифікації методом k найближчих сусідів: {accuracy_14:.2f}")

# Вивід таблиці частот точності класифікації
print("Таблиця частот точності класифікації:")
print("--------------------------------------------------")
print("| Клас | Правильно класифіковано | Неправильно класифіковано |")
print("--------------------------------------------------")
for target in np.unique(y_test_14):
    correct = np.sum((y_test_14 == target) & (y_pred_14 == target))
    incorrect = np.sum((y_test_14 == target) & (y_pred_14 != target))
    print(f"|  {target}  | {correct:^25} | {incorrect:^28} |")
print("--------------------------------------------------")

# Візуалізація
plt.figure(figsize=(10, 6))
plt.scatter(x_test_14[:, 0], x_test_14[:, 1], c=y_pred_14, cmap='viridis', s=50)
plt.title('Класифікація методом k найближчих сусідів')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.colorbar(label='Клас')
plt.show()
