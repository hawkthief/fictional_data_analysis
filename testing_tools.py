from configurations import TSNE_TESTING_PERPLEXITY, TSNE_TESTING_LEARNING_RATE, TSNE_TESTING_EARLY_EXAGGERATION, \
    K_VALUES
from data_classification import evaluateknn
from data_processing import reducedimensions


def parsetsne(dataframe):
    """
    Calls for evaluation of each possible combination of PERPLEXITY, LEARNING_RATE and EARLY_EXAGGERATION
    defined as TESTING variables in the configurations file and chooses the one with the best overall accuracy.
    (RNG seed is kept constant for the sake of reproducibility)
    @type dataframe: Pandas DataFrame
    @:param: dataframe - dataframe to be cleaned must have the following column structure:
        image_id(int), subject_id(int), syndrome_id(int), dimension 1 (NumPy float),
        dimension 2 (NumPy float), [...], dimension 320 (NumPy float)
    """

    best_result_euclidean = {'accuracy': 0, 'perplexity': 0, 'early_exaggeration': 0, 'learning_rate': 0}
    best_result_cosine = {'accuracy': 0, 'perplexity': 0, 'early_exaggeration': 0, 'learning_rate': 0}

    for perplexity in TSNE_TESTING_PERPLEXITY:
        for learning_rate in TSNE_TESTING_LEARNING_RATE:
            for early_exaggeration in TSNE_TESTING_EARLY_EXAGGERATION:
                tsnedataframe = reducedimensions(dataframe, {
                    'perplexity': perplexity, 'learning_rate': learning_rate, 'early_exaggeration': early_exaggeration
                })

                metrics_euclidian = evaluateknn(K_VALUES, tsnedataframe, log=False)
                if metrics_euclidian['accuracy'] > best_result_euclidean['accuracy']:
                    best_result_euclidean['accuracy'] = metrics_euclidian['accuracy']
                    best_result_euclidean['perplexity'] = perplexity
                    best_result_euclidean['learning_rate'] = learning_rate
                    best_result_euclidean['early_exaggeration'] = early_exaggeration
                metrics_cosine = evaluateknn(K_VALUES, tsnedataframe, True,False)
                if metrics_cosine['accuracy'] > best_result_cosine['accuracy']:
                    best_result_cosine['accuracy'] = metrics_cosine['accuracy']
                    best_result_cosine['perplexity'] = perplexity
                    best_result_cosine['learning_rate'] = learning_rate
                    best_result_cosine['early_exaggeration'] = early_exaggeration

    print(best_result_euclidean)
    print(best_result_cosine)
