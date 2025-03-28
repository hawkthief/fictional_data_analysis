from configurations import (TESTING_MODE, DATASET_PATH, RNG, TSNE_PERPLEXITY, TSNE_LEARNING_RATE,
                            TSNE_EARLY_EXAGGERATION, K_VALUES, PLOT_FOLDER, TSNE_TESTING_EARLY_EXAGGERATION,
                            TSNE_TESTING_LEARNING_RATE, TSNE_TESTING_PERPLEXITY)
from data_processing import (loadpicklefile, flattenhierarchy, cleandataframe, generatestatistics,
                             generatetables, reducedimensions)
from data_visualisation import plottsne, rocauccurve
from data_classification import evaluateknn
from testing_tools import parsetsne

def getdata():
    """
    @:rtype: Data Frame
    """
    dataset = loadpicklefile(DATASET_PATH)
    dataframe = flattenhierarchy(dataset)
    dataframe = cleandataframe(dataframe)
    return dataframe


def plotall(dataframe):
    """
    Calls for plot creation of each possible combination of PERPLEXITY, LEARNING_RATE and EARLY_EXAGGERATION
    defined as TESTING variables in the configurations file. (RNG seed is kept constant for the sake of reproducibility)
    @type dataframe: Pandas DataFrame
    @:param: dataframe - dataframe to be cleaned must have the following column structure:
        image_id(int), subject_id(int), syndrome_id(int), dimension 1 (NumPy float),
        dimension 2 (NumPy float), [...], dimension 320 (NumPy float)
    """

    for perplexity in TSNE_TESTING_PERPLEXITY:
        for learning_rate in TSNE_TESTING_LEARNING_RATE:
            for early_exaggeration in TSNE_TESTING_EARLY_EXAGGERATION:
                tsnedataframe = reducedimensions(dataframe, {
                    'random_state': RNG, 'perplexity': perplexity,
                    'learning_rate': learning_rate, 'early_exaggeration': early_exaggeration
                })
                plottsne(tsnedataframe, PLOT_FOLDER + '\\plot seed' + str(RNG) + " perplexity" + str(perplexity) +
                         " learning rate" + str(learning_rate) + " early exaggeration" + str(early_exaggeration)
                         )


if __name__ == '__main__':
    # Main instance of Data Frame
    dataframe_i = getdata()

    if TESTING_MODE:
        # This one takes a while
        parsetsne(dataframe_i)

    else:
        # Statistics
        print(generatestatistics(dataframe_i))

        # Reduces dimensionality for KNN
        dataframe_i = reducedimensions(dataframe_i, {'perplexity': TSNE_PERPLEXITY,
                                                     'learning_rate': TSNE_LEARNING_RATE,
                                                     'early_exaggeration': TSNE_EARLY_EXAGGERATION})

        # Data Visualisation plot generated to /plots folder
        plottsne(dataframe_i, "t-SNE 2D Visualization of 320D Image Embeddings - plot seed" + str(RNG) + " perplexity" +
                 str(TSNE_PERPLEXITY) + " learning rate" + str(TSNE_LEARNING_RATE) +
                 " early exaggeration" + str(TSNE_EARLY_EXAGGERATION))

        # KNN evaluation for Euclidean an Cosine Metrics
        metrics_euclidian = evaluateknn(K_VALUES, dataframe_i)
        metrics_cosine = evaluateknn(K_VALUES, dataframe_i, True)

        # ROC AUC visualisation plots generated to /plots folder
        rocauccurve(metrics_euclidian, metrics_cosine, dataframe_i['syndrome_id'].unique())

        # Generates XLSX metrics Table to /tables folder
        generatetables(metrics_euclidian, metrics_cosine)
