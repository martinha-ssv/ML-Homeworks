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
            MINMAXSCALER.fit(train.drop('target', axis=1, inplace=False))
            target = train['target'].copy().reset_index(drop=True)
            train = pd.DataFrame(MINMAXSCALER.transform(train.drop('target', axis=1, inplace=False)))
            train['target'] = target

            target = test['target'].copy().reset_index(drop=True)
            test = pd.DataFrame(MINMAXSCALER.transform(test.drop('target', axis=1, inplace=False)))
            test['target'] = target

        knn_accuracies.append(fit_and_score(knn, train, test))
        nbayes_accuracies.append(fit_and_score(nbayes, train, test))
    return knn_accuracies, nbayes_accuracies

knn_accuracies, nbayes_accuracies = knn_vs_nbayes(data)
