# SCALED DATA ######################################
print(colored('H0: KNN Classifier is not better than Naive Bayes Classifier when using MinMax Scaling', 'blue'))
test_result = stats.ttest_rel(knn_accuracies_scaled, nbayes_accuracies_scaled, alternative='greater')     # [alternative=] ‘greater’: the mean of the distribution underlying the first sample is greater than the mean of the distribution underlying the second sample.

print('T-statistic: ', test_result.statistic)
print('P-value: ', test_result.pvalue)
print('Degrees of Freedom: ', test_result.df)

if test_result.pvalue < 0.05:
    print(colored('We can reject the null hypothesis, and conclude that the KNN Classifier is better than the Naive Bayes Classifier, when using MinMax Scaling', "green"))
else:
    print(colored('We cannot reject the null hypothesis, thus we cannot conclude that the KNN model is better than the Naive Bayes model, when using MinMax Scaling', "red"))
