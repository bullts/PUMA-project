import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn import tree
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_score, recall_score

from sklearn.linear_model import LogisticRegression

def svm_method (data):
    X_train, X_test, y_train, y_test = split_data(data)
    # Propozycje parametrów oraz algorytmów
    parameters = {'kernel': ('linear', 'rbf'),
                  'C': [2 ** -6, 2 ** 6],
                  'gamma': [2 ** -6, 2 ** 6],
                  'degree': [1, 2, 3, 4, 5, 6, 7, 8]}
    svc = svm.SVC()

    # Uruchomienie metody GridSearch
    print("GRID: ")
    setGS = GridSearchCV(svc, parameters, cv=2, scoring='accuracy', verbose=10)
    setGS.fit(X_train, y_train)
    print('Najlepsze parametry: ' + str(setGS.best_params_))
    print('Sredni wynik kross-walidacji: ' + str(setGS.best_score_))
    bestGS = setGS.score(X_test, y_test)
    print('Dokladnosc zb testowego: ' + str(bestGS))
    bestGSt = setGS.score(X_train, y_train)
    print('Dokladnosc zb treningowego: ' + str(bestGSt))

    # Urychomienie metody RandomizedSearch
    print("RANDOMIZED: ")
    setRS = RandomizedSearchCV(svc, parameters, random_state=0, cv=2, scoring='accuracy', verbose=10)
    setRS.fit(X_train, y_train)
    print('Najlepsze parametry: ' + str(setRS.best_params_))
    print('Sredni wynik kross-walidacji: ' + str(setRS.best_score_))
    bestRS = setRS.score(X_test, y_test)
    print('Dokladnosc zb testowego: ' + str(bestRS))
    bestRSt = setRS.score(X_train, y_train)
    print('Dokladnosc zb treningowego: ' + str(bestRSt))

def pjk_method (data):
    X_train, X_test, y_train, y_test = split_data(data)
    classifier = tree.DecisionTreeClassifier(random_state=0)
    path = classifier.cost_complexity_pruning_path(X_train, y_train)
    alphas = path.ccp_alphas

    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    # Przycinanie drzewa decyzyjnego
    classifiers = []
    for alpha in alphas:
        classifier = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=alpha)
        classifier.fit(X_train, y_train)
        classifiers.append(classifier)

    # Sprawdzenie dokładności klasyfikacji dla obu zbiorów
    train_scores = [clf.score(X_train, y_train) for clf in classifiers]
    test_scores = [clf.score(X_test, y_test) for clf in classifiers]

    classifiers = classifiers[:-1]
    alphas = alphas[:-1]

    # Wykres alphy w zestawieniu dokładnością klasyfikacji (dla danej alphy) przyciętego drzewa
    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy (dokladnosc)")
    ax.set_title("accuracy vs alpha")
    ax.plot(alphas, train_scores[:-1], marker='o', label="zbior treningowy")
    ax.plot(alphas, test_scores[:-1], marker='o', label="zbior testowy")
    ax.legend()
    plt.show()

    # Wyuczenie lasu losowego
    forest = RandomForestClassifier(n_estimators=200, max_leaf_nodes=15, n_jobs=-1, random_state=0)
    forest.fit(X_train, y_train)

    # Edit
    # Wyuczenie klasyfikatora regresji logistycznej
    logistic_regression = LogisticRegression(max_iter=10000)
    logistic_regression.fit(X_train, y_train)

    # Sprawdzenie dokładności klasyfikatora lasu losowego dla obu zbiorów
    forest_train_score = forest.score(X_train, y_train)
    forest_test_score = forest.score(X_test, y_test)
    print("Dokladnosc klasyfikacji lasu losowego na zbiorze treningowym wynosi {0:3f}, \
    zas na zbiorze testowym {1:3f}".format(forest_train_score, forest_test_score))

    # Edit
    # Sprawdzenie dokladnosci klasyfikatora regresji logistycznej
    logistic_regression_train_score = logistic_regression.score(X_train, y_train)
    logistic_regression_test_score = logistic_regression.score(X_test, y_test)
    print("Dokladnosc klasyfikacji regresji logistycznej na zbiorze treningowym wynosi {0:3f}, \
    zas na zbiorze testowym {1:3f}".format(logistic_regression_train_score, logistic_regression_test_score))

    print("")

    # Ocena jakości klasyfikatora, confusion matrix (dla zb testowego i dla całego zb
    forest_prediction = forest.predict(X_test)
    forest_prediction_all = forest.predict(X)
    cf_matrix_all = confusion_matrix(y, forest_prediction_all)
    cf_matrix = confusion_matrix(y_test, forest_prediction)
    print("cf_matrix dla lasu:")  # Edit do ogarniecia wyswietlanych danych
    print(cf_matrix)
    print("cf_matrix_all dla lasu:")  # Edit do ogarniecia wyswietlanych danych
    print(cf_matrix_all)

    # Edit
    # Ocena jakości klasyfikatora, confusion matrix (dla zb testowego i dla całego zb DLA KL REGRESJI LOGISTYCZNEJ
    l_prediction = logistic_regression.predict(X_test)
    l_prediction_all = logistic_regression.predict(X)
    cf_matrix_l_all = confusion_matrix(y, l_prediction_all)
    cf_matrix_l = confusion_matrix(y_test, l_prediction)
    print("cf_matrix dla logistycznej:")  # potocznie do ogarniecia wyswietlanych danych
    print(cf_matrix_l)
    print("cf_matrix_all dla logistycznej:")  # potocznie do ogarniecia wyswietlanych danych
    print(cf_matrix_l_all)

    print("")

    # Sprawdzenie precyzji oraz pełności
    precision = precision_score(y_test, forest_prediction, average='micro') #average : string, [None, ‘binary’ (default), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]
    recall = recall_score(y_test, forest_prediction, average='micro')       # można pomyśleć nad różnymi wartościami
    precision_all = precision_score(y, forest_prediction_all, average='micro')
    recall_all = recall_score(y, forest_prediction_all, average='micro')
    print("Precyzja wynosi {0:3f}, zas pelnosc {1:3f}".format(precision, recall))
    print("DLA CALEGO ZB: Precyzja wynosi {0:3f}, zas pelnosc {1:3f}".format(precision_all, recall_all))

    # Edit
    # Sprawdzenie precyzji oraz pełności dla kl regresji logistycznej
    precision_l = precision_score(y_test, l_prediction, average='micro')
    recall_l = recall_score(y_test, l_prediction, average='micro')
    precision_all_l = precision_score(y, l_prediction_all, average='micro')
    recall_all_l = recall_score(y, l_prediction_all, average='micro')
    print("DLA KL REGRESJI LOGISTYCZNEJ: ")
    print("Precyzja wynosi {0:3f}, zas pelnosc {1:3f}".format(precision_l, recall_l))
    print("DLA CALEGO ZB: Precyzja wynosi {0:3f}, zas pelnosc {1:3f}".format(precision_all_l, recall_all_l))

def dt_method (data):
    X_train, X_test, y_train, y_test = split_data(data)
    # Trenowanie klasyfikatora
    tree_classifier = tree.DecisionTreeClassifier(random_state=0)
    tree_classifier.fit(X_train, y_train)

    # Wyniki dla zbioru treningowego
    prediction = tree_classifier.predict(X_train)
    score = tree_classifier.score(X_train, y_train)
    print('zb treningowy ------------------------------')

    #print('prediction:')
    #print(prediction)
    print('score:')
    print(score)

    # Macierz pomyłek zbioru treningowego
    cf_matrix = confusion_matrix(y_train, prediction)
    print('conf matrix:')
    print(cf_matrix)

    # Wyniki dla zbioru testowego
    prediction = tree_classifier.predict(X_test)
    score = tree_classifier.score(X_test, y_test)
    print('zb testowy ------------------------------')

    #print('prediction:')
    #print(prediction)
    print('score:')
    print(score)

    # Macierz pomyłek zbioru testowego
    cf_matrix = confusion_matrix(y_test, prediction)
    print('conf matrix:')
    print(cf_matrix)

    # Drukowanie drzewa do png
    tree.plot_tree(tree_classifier.fit(X_test, y_test))
    plt.savefig("tree.svg")

def gaussian_method (data):
    X_train, X_test, y_train, y_test = split_data(data)
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    train_prediction = clf.predict(X_train)
    train_score = clf.score(X_train, y_train)
    #print("Predykcja zbioru treningowego: ")
    #print(train_prediction)
    print("Score zbioru treningowego: ")
    print(train_score)
    test_prediction = clf.predict(X_test)
    test_score = clf.score(X_test, y_test)
    #print("Predykcja zbioru testowego: ")
   # print(test_prediction)
    print("Score zbioru testowego: ")
    print(test_score)

def split_data_complex (data):
    y = data[:, -1]  # oryginalne przyporzadkowanie
    X = data[:, :-1]  # argumenty
    # podzial zbioru na dane treningowe i testowe
    # return X_train, X_test, y_train, y_test
    return train_test_split(X, y, test_size=0.3, random_state=0)


def split_data(data):
    # zamiana ostatniej kolumny na one-hot vectory
    # i podział na 3 klasy
    note = data[:, -1].astype(int)

    for i in range(len(note)):
        note[i] = qualityclass(note[i])
    one_hot_note = np.eye(3)[note]

    data = np.delete(data, 11, axis=1)
    data = np.concatenate((data, one_hot_note), axis=1)

    y = data[:, -1]  # oryginalne przyporzadkowanie
    X = data[:, :-1]  # argumenty

    # podzial zbioru na dane treningowe i testowe
    return train_test_split(X, y, test_size=0.3, random_state=0)

def qualityclass(x):
    if x == 3:
        return 0
    if x == 4:
        return 0
    if x == 5:
        return 0
    if x == 6:
        return 1
    if x == 7:
        return 1
    if x == 8:
        return 2
    if x == 9:
        return 2

if __name__ == '__main__':
    # pobranie danych z pliku z pominieciem pierwszego wiersza
    data_red = np.genfromtxt('winequality-red.csv', delimiter=';', skip_header=1)
    data_white = np.genfromtxt('winequality-white.csv', delimiter=';', skip_header=1)

    # print("METODA GAUSSA")
    # gaussian_method(data_red)

    print("METODA SVM")
    svm_method(data_white)

    # print("METODA PJK")
    # pjk_method(data_red)

    # print("METODA DT")
    # dt_method(data_red)

    ########################################################################################

    # Wymagania:
    # Projekt powinien zawierać porównanie działania co najmniej dwóch metod uczenia maszynowego. Kryteriami oceny będą między innymi:
    #
    # a) dobranie odpowiednich parametrów pracy algorytmów;
    #
    # b) przedstawienie zbioru danych poddawanego analizie (co znaczą poszczególne argumenty, z jakim typem zmiennych mamy do czynienia: w skali nominalnej czy porządkowej; zakres poszczególnych argumentów np. [0, 100];
    #
    # c) analiza i interpretacja wyników opatrzona wykresami i komentarzami; wykresy powinny być czytelne, osie oznaczone, wstawiona legenda.



    #### Co nowego? ###
    #
    # 1) Dodanie metody gaussa
    # 2) Ujednolicenie parametrów wszystkich metod
    # 3) Dodanie parametrów powodujących wyświetlanie progressu metody svc  
    # 4) Oczyszczenie outputów metod by ułatwić analize danych oraz podsumowanie


    ### TO DO ###
    #
    # - można puścić na jakąś godzinkę svc do liczenia by potem zebrać logi progressu i umieścić w sprawku ( ty zrób puść na red ja na white)
    # - porównanie metod względem red white i względem innych metod

