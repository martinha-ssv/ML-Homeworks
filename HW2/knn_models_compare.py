models_accuracies = []

for k in ks:
    for weight in ['uniform', 'distance']:
        knn = KNeighborsClassifier(n_neighbors=k, weights=weight)
        knn.fit(X_train, y_train)
        train_acc = knn.score(X_train, y_train)
        test_acc = knn.score(X_test, y_test)
        models_accuracies.append({'k': k, 'weight': weight, 'train_acc': train_acc, 'test_acc': test_acc})

models_accuracies = pd.DataFrame(models_accuracies)
models_accuracies
