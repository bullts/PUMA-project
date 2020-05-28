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

def svm_method (X_train, X_test, y_train, y_test) :
    # Propozycje parametrów oraz algorytmów
    parameters = {'kernel': ('linear', 'rbf', 'poly'),
                  'C': [2 ** -5, 2 ** 5],
                  'gamma': [2 ** -5, 2 ** 5],
                  'degree': [1, 2, 3, 4, 5, 6]}
    svc = svm.SVC()

    # Uruchomienie metody GridSearch
    print("GRID: ")
    setGS = GridSearchCV(svc, parameters)
    setGS.fit(X_train, y_train)
    print('Najlepsze parametry: ' + str(setGS.best_params_))
    print('Sredni wynik kross-walidacji: ' + str(setGS.best_score_))
    bestGS = setGS.score(X_test, y_test)
    print('Dokladnosc zb testowego: ' + str(bestGS))
    bestGSt = setGS.score(X_train, y_train)
    print('Dokladnosc zb treningowego: ' + str(bestGSt))

    # Urychomienie metody RandomizedSearch
    print("RANDOMIZED: ")
    setRS = RandomizedSearchCV(svc, parameters, cv=3, random_state=2)
    setRS.fit(X, y)
    print('Najlepsze parametry: ' + str(setRS.best_params_))
    print('Sredni wynik kross-walidacji: ' + str(setRS.best_score_))
    bestRS = setRS.score(X_test, y_test)
    print('Dokladnosc zb testowego: ' + str(bestRS))
    bestRSt = setRS.score(X_train, y_train)
    print('Dokladnosc zb treningowego: ' + str(bestRSt))

    # Podsumowanie
    # Wyniki:
    # GRID:
    # Najlepsze parametry: {'C': 32, 'degree': 1, 'gamma': 0.03125, 'kernel': 'linear'}
    # Sredni wynik kross-walidacji: 0.9499058380414312
    # Dokladnosc zb testowego: 0.8974358974358975
    # Dokladnosc zb treningowego: 0.9608938547486033
    # RANDOMIZED:
    # Najlepsze parametry: {'kernel': 'linear', 'gamma': 32, 'degree': 2, 'C': 32}
    # Sredni wynik kross-walidacji: 0.9222526219790241
    # Dokladnosc zb testowego: 0.9487179487179487
    # Dokladnosc zb treningowego: 0.9720670391061452
    #
    # Dla tych danych metoda RandomizedSearch spisała się lepiej od metody GridSerach. Pomimo niższego wyniku cross-validation
    # wyszukała lepsze parametry potrzebene do wyuczenia kalsyfikatora w efekcie dokładność zb testowego jak i treningowego jest wyższa
    # dla kladyfikatora wyuczonego z parametrami znalezionymi przez metode RandomizedSearch niż GridSearch.

def pjk_method (X_train, X_test, y_train, y_test):
    classifier = tree.DecisionTreeClassifier(random_state=0)
    path = classifier.cost_complexity_pruning_path(X_train, y_train)
    alphas = path.ccp_alphas

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
    logistic_regression = LogisticRegression()
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
    cf_matrix_all = confusion_matrix(y, forest_prediction_all, labels=[1, 0])
    cf_matrix = confusion_matrix(y_test, forest_prediction, labels=[1, 0])
    print("cf_matrix dla lasu:")  # Edit do ogarniecia wyswietlanych danych
    print(cf_matrix)
    print("cf_matrix_all dla lasu:")  # Edit do ogarniecia wyswietlanych danych
    print(cf_matrix_all)

    # Edit
    # Ocena jakości klasyfikatora, confusion matrix (dla zb testowego i dla całego zb DLA KL REGRESJI LOGISTYCZNEJ
    l_prediction = logistic_regression.predict(X_test)
    l_prediction_all = logistic_regression.predict(X)
    cf_matrix_l_all = confusion_matrix(y, l_prediction_all, labels=[1, 0])
    cf_matrix_l = confusion_matrix(y_test, l_prediction, labels=[1, 0])
    print("cf_matrix dla logistycznej:")  # potocznie do ogarniecia wyswietlanych danych
    print(cf_matrix_l)
    print("cf_matrix_all dla logistycznej:")  # potocznie do ogarniecia wyswietlanych danych
    print(cf_matrix_l_all)

    print("")

    # Sprawdzenie precyzji oraz pełności
    precision = precision_score(y_test, forest_prediction)
    recall = recall_score(y_test, forest_prediction)
    precision_all = precision_score(y, forest_prediction_all)
    recall_all = recall_score(y, forest_prediction_all)
    print("Precyzja wynosi {0:3f}, zas pelnosc {1:3f}".format(precision, recall))
    print("DLA CALEGO ZB: Precyzja wynosi {0:3f}, zas pelnosc {1:3f}".format(precision_all, recall_all))

    # Edit
    # Sprawdzenie precyzji oraz pełności dla kl regresji logistycznej
    precision_l = precision_score(y_test, l_prediction)
    recall_l = recall_score(y_test, l_prediction)
    precision_all_l = precision_score(y, l_prediction_all)
    recall_all_l = recall_score(y, l_prediction_all)
    print("DLA KL REGRESJI LOGISTYCZNEJ: ")
    print("Precyzja wynosi {0:3f}, zas pelnosc {1:3f}".format(precision_l, recall_l))
    print("DLA CALEGO ZB: Precyzja wynosi {0:3f}, zas pelnosc {1:3f}".format(precision_all_l, recall_all_l))

    print("")

    # Krzywa ROC
    FPR, TPR, _ = roc_curve(forest_prediction, y_test)
    plt.plot(FPR, TPR, linewidth=2, color='red')  # wykres krzywej ROC
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')  # wykres linii przerywanej
    plt.xlim([0.0, 1.0])  # zakres na osi OX
    plt.ylim([0.0, 1.03])  # zakres na osi OY, minimalnie ponad 1
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC curve)')
    plt.show()

    # Edit
    # Krzywa ROC dla kl regresji logistycznej
    FPR, TPR, _ = roc_curve(l_prediction, y_test)
    plt.plot(FPR, TPR, linewidth=2, color='red')  # wykres krzywej ROC
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')  # wykres linii przerywanej
    plt.xlim([0.0, 1.0])  # zakres na osi OX
    plt.ylim([0.0, 1.03])  # zakres na osi OY, minimalnie ponad 1
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC curve) for l')
    plt.show()

    # Krzywa ROC dla całego zb
    FPR, TPR, _ = roc_curve(forest_prediction_all, y)
    plt.plot(FPR, TPR, linewidth=2, color='red')  # wykres krzywej ROC
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')  # wykres linii przerywanej
    plt.xlim([0.0, 1.0])  # zakres na osi OX
    plt.ylim([0.0, 1.03])  # zakres na osi OY, minimalnie ponad 1
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC curve) ALL')
    plt.show()

    # Edit
    # Krzywa ROC dla całego zb dla kl regresji logistycznej
    FPR, TPR, _ = roc_curve(l_prediction_all, y)
    plt.plot(FPR, TPR, linewidth=2, color='red')  # wykres krzywej ROC
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')  # wykres linii przerywanej
    plt.xlim([0.0, 1.0])  # zakres na osi OX
    plt.ylim([0.0, 1.03])  # zakres na osi OY, minimalnie ponad 1
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC curve) for l  ALL')
    plt.show()

    # Wartość AUC
    AUC = roc_auc_score(forest_prediction, y_test)
    print("AUC:")
    print(AUC)

    # Edit
    # Wartosc AUC dla kl regresji logistycznej
    AUC_l = roc_auc_score(l_prediction, y_test)
    print("AUC l:")
    print(AUC_l)

    # Wartość AUC dla całego zb
    AUC_all = roc_auc_score(forest_prediction_all, y)
    print("AUC all:")
    print(AUC_all)

    # Edit
    # Wartość AUC dla całego zb dla kl regresji logistycznej
    AUC_all_l = roc_auc_score(l_prediction_all, y)
    print("AUC l all:")
    print(AUC_all_l)

    # Podsumowanie
    # Najlepszą dokładność dla zbioru treningowego otzrymujemy dla pełnego drzewa - wtedy też jest najgorsza dla zbioru testowego.
    # W miarę przycinania drzewa dokładność zbioru treningowego spada ale dokładność zbioru testowego rośnie.
    # W tym przypadku najlepsza dokładność dla zbioru testowego jest dla alphy = 0.016 - wtedy też jest najgorsza dla zb treningowego
    # Precyzja zb testowego to ok 80% co nie jest takie złe zato precyzja dla zb treningowego to ok 68% co nie jest doskonałym wynikiem.
    # Confusion matrix mówi nam, że większość przypadków należących do zb "A" jest prypisywana do zb "B", Klasyfikacja przypadków
    # należących do zb "B" jest już bardziej prawidłowa.
    # Pole pod krzywą ROC, czyli wartość AUC, wynosi ok 0.6. Jest to lepszy wynik niż dla losowego klasyfikatora,
    # lecz daelki do ideału (wartości 1).
    # Dla całego zbioru Krzywa ROC jak i pole AUC jest lepsze niż tylko dla zb testowego (AUC wynosi ok 0.77)
    # Edit
    # Dokladnosc klasyfikacji regresji logistycznej na zbiorze treningowym wynosi 0.708333, zas na zbiorze testowym 0.706250
    # Confusion matrix wyglada podobnia jak dla lasu losowego
    # AUC dla kl regresji logistycznej jest lepszy dla zb testowego niz AUC lasu losowego dla tego samego zb,
    # lecz AUC dla calego zb wypada juz slabiej na tle lasu losowego.

def dt_method (X_train, X_test, y_train, y_test):
    # Trenowanie klasyfikatora
    tree_classifier = tree.DecisionTreeClassifier(random_state=0)
    tree_classifier.fit(X_train, y_train)

    # Wyniki dla zbioru treningowego
    prediction = tree_classifier.predict(X_train)
    score = tree_classifier.score(X_train, y_train)
    print('zb treningowy ------------------------------')

    print('prediction:')
    print(prediction)
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

    print('prediction:')
    print(prediction)
    print('score:')
    print(score)

    # Macierz pomyłek zbioru testowego
    cf_matrix = confusion_matrix(y_test, prediction)
    print('conf matrix:')
    print(cf_matrix)

    # Drukowanie drzewa do png
    tree.plot_tree(tree_classifier.fit(X_test, y_test))
    plt.savefig("tree.png")

    ##############################################
    # Rysunek drzewa decyzyjnego znajduje się w pliku tree.png
    # Wyniki zbioru treningowego to 100% poprawności. Wszystkie kwiaty zosatły prawidłowo sklasyfikowane
    # conf matrix:
    # [[33  0  0]
    #  [ 0 35  0]
    #  [ 0  0 37]]
    # Wyniki zbioru testowego to 95% poprawności. Pierwszy gatunek został sklasyfikowany prawidłowo. W drugim jeden
    # egzemplarz został sklasyfikowany jako gatunek trzeci a w trzecim jeden gatunek został sklasyfikowany jako gatunek drugi.
    # conf matrix:
    # [[17  0  0]
    #  [ 0 14  1]
    #  [ 0  1 12]]
    # Jest to dość dobra dokładność co oznacza że drzewo nie jest przetrenowane, ponieważ dobrze radzi sobie również
    # ze zbiorem testowym.

    # Wynik:
    # METODA DT
    # zb treningowy ------------------------------
    # prediction:
    # [6. 5. 6. ... 6. 6. 7.]
    # score:
    # 1.0
    # conf matrix:
    # [[  8   0   0   0   0   0]
    #  [  0  39   0   0   0   0]
    #  [  0   0 504   0   0   0]
    #  [  0   0   0 417   0   0]
    #  [  0   0   0   0 140   0]
    #  [  0   0   0   0   0  11]]
    # zb testowy ------------------------------
    # prediction:
    # [6. 7. 6. 3. 6. 4. 5. 6. 5. 5. 7. 6. 5. 6. 5. 8. 6. 6. 6. 7. 6. 7. 5. 6.
    #  5. 6. 5. 6. 7. 6. 5. 5. 7. 7. 6. 5. 6. 6. 6. 6. 5. 7. 6. 5. 5. 6. 5. 6.
    #  7. 5. 6. 5. 5. 7. 4. 5. 5. 6. 7. 6. 6. 6. 5. 4. 7. 6. 7. 6. 6. 7. 5. 6.
    #  6. 3. 5. 7. 6. 6. 7. 5. 5. 6. 7. 5. 7. 6. 6. 6. 5. 5. 6. 5. 5. 6. 4. 5.
    #  5. 7. 5. 5. 6. 5. 5. 5. 6. 6. 7. 4. 6. 6. 7. 5. 5. 5. 7. 6. 6. 6. 6. 7.
    #  6. 7. 7. 6. 7. 6. 6. 6. 6. 5. 5. 5. 6. 6. 6. 5. 6. 6. 5. 7. 6. 6. 6. 6.
    #  3. 6. 4. 5. 5. 6. 4. 6. 6. 5. 6. 6. 7. 5. 5. 6. 5. 5. 6. 6. 6. 6. 6. 7.
    #  5. 5. 6. 6. 7. 6. 6. 6. 5. 6. 5. 8. 4. 6. 5. 6. 5. 6. 6. 5. 7. 7. 6. 5.
    #  5. 5. 5. 5. 6. 5. 5. 8. 6. 6. 6. 5. 4. 5. 6. 7. 6. 5. 6. 6. 6. 5. 7. 6.
    #  6. 6. 6. 6. 6. 6. 5. 6. 6. 6. 6. 6. 6. 6. 6. 5. 5. 5. 6. 5. 5. 6. 5. 5.
    #  7. 5. 6. 8. 5. 7. 5. 5. 5. 6. 6. 5. 6. 5. 6. 5. 5. 5. 6. 5. 6. 5. 5. 5.
    #  6. 6. 7. 5. 5. 6. 5. 7. 5. 7. 5. 5. 5. 5. 5. 5. 4. 6. 5. 7. 5. 5. 5. 7.
    #  6. 7. 7. 7. 5. 5. 6. 5. 8. 5. 5. 5. 7. 6. 7. 6. 6. 6. 7. 5. 5. 7. 5. 5.
    #  6. 5. 5. 6. 5. 5. 5. 5. 5. 6. 6. 5. 4. 6. 6. 6. 6. 7. 6. 5. 6. 6. 6. 6.
    #  6. 5. 5. 6. 6. 6. 5. 5. 6. 7. 7. 7. 6. 6. 8. 6. 6. 6. 6. 4. 6. 6. 6. 6.
    #  5. 5. 5. 6. 5. 6. 5. 5. 6. 5. 6. 5. 6. 6. 6. 5. 5. 5. 6. 6. 5. 5. 5. 8.
    #  5. 7. 6. 6. 6. 6. 5. 5. 7. 5. 5. 6. 4. 6. 7. 5. 6. 6. 5. 6. 6. 6. 5. 5.
    #  5. 7. 4. 7. 6. 6. 4. 6. 6. 6. 7. 6. 5. 5. 5. 5. 5. 6. 7. 6. 5. 7. 7. 6.
    #  6. 8. 6. 7. 6. 5. 6. 5. 5. 6. 6. 5. 5. 5. 6. 5. 5. 6. 6. 6. 4. 5. 5. 5.
    #  6. 7. 5. 7. 5. 6. 6. 6. 6. 7. 6. 5. 6. 7. 6. 5. 8. 6. 5. 6. 5. 6. 5. 5.]
    # score:
    # 0.55625
    # conf matrix:
    # [[  0   1   1   0   0   0]
    #  [  1   2   7   2   2   0]
    #  [  0   6 108  51  11   1]
    #  [  2   7  60 125  23   4]
    #  [  0   0   2  27  29   1]
    #  [  0   0   0   3   1   3]]

if __name__ == '__main__':
    # pobranie danych z pliku z pominieciem pierwszego wiersza
    data = np.genfromtxt('winequality-red.csv', delimiter=';', skip_header=1)
    print(data)


    y = data[:, -1]  # oryginalne przyporzadkowanie
    X = data[:, :-1]  # argumenty

    # print("x: ", X)
    # print("y", y)

    # podzial zbioru na dane treningowe i testowe
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    print("X_test: ", X_test)
    print("X_train: ", X_train)
    print("y_test: ", y_test)
    print("y_train: ", y_train)

    # nie działa nie wiem czemu (liczy i nie przestaje ,nic sie nie dzieje po godzinie czekania)
    # print("METODA SVM")
    # svm_method(X_train, X_test, y_train, y_test)

    # coś liczy ale sypie błędami
    # print("METODA PJK")
    # pjk_method(X_train, X_test, y_train, y_test)

    # działą, wyniki w komentarzu  pod metodą
    # print("METODA DT")
    # dt_method(X_train, X_test, y_train, y_test)

    # svc = svm.SVC()
    # svc.fit(X_train, y_train)
    # print(svc.score(X_train, y_train))

    # naive_model = GaussianNB()
    # model_prediction = naive_model.fit(X_train, y_train.ravel())
    # x = model_prediction.score(X_train, y_train)
    # print(x)

