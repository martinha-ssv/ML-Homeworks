# Run models and store accuracies

N_NEIGHBORS = 5
knn = KNeighborsClassifier(n_neighbors=5)
nbayes = GaussianNB()

def fit_and_score(model, train, test):
    model = model.fit(train.drop('target', axis=1, inplace=False), train['target'])
    return model.score(test.drop('target', axis=1, inplace=False), test['target'])


def knn_vs_nbayes(df, scale=False):
    knn_accuracies = []
    nbayes_accuracies = []
    for train_index, test_index in generate_splits(df):
        train = df.loc[train_index]
        test = df.loc[test_index]
        if scale: 
            MINMAXSCALER = MinMaxScaler()
            MINMAXSCALER.fit_transform(train[['age','trestbps','chol','thalach','oldpeak']])

            train[['age','trestbps','chol','thalach','oldpeak']] = MINMAXSCALER.transform(train[['age','trestbps','chol','thalach','oldpeak']].copy())
            test[['age','trestbps','chol','thalach','oldpeak']] = MINMAXSCALER.transform(test[['age','trestbps','chol','thalach','oldpeak']].copy())            #test['target'] = target

        knn_accuracies.append(fit_and_score(knn, train, test))
        nbayes_accuracies.append(fit_and_score(nbayes, train, test))
    return knn_accuracies, nbayes_accuracies

knn_accuracies, nbayes_accuracies = knn_vs_nbayes(data)
