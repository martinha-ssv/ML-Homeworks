def generate_splits(df, N_SPLITS = 5):
    strat_kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)
    split = strat_kfold.split(np.zeros(len(df)), df['target'])
    return split
