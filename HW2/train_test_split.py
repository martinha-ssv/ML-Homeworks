X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1, inplace=False), data['target'], test_size=0.20, random_state=0, stratify=data['target'])
ks = [1,5,10,20,30]