import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB




# pobranie danych z pliku z pominieciem pierwszego wiersza
data = np.genfromtxt('winequality-red.csv', delimiter=';', skip_header=1)
print(data)


y = data[:, -1]  # oryginalne przyporzadkowanie
X = data[:, :-1]  # argumenty

# podzial zbioru na dane treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

svc = svm.SVC()
svc.fit(X_train, y_train)
print(svc.score(X_train, y_train))

naive_model = GaussianNB()
model_prediction = naive_model.fit(X_train, y_train.ravel())
x = model_prediction.score(X_train, y_train)
print(x)

