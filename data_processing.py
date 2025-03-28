import pandas as pd
from configurations import TOP_K, TABLE_FOLDER, RNG
from sklearn.manifold import TSNE


def loadpicklefile(path):
    """
    Reads piclke file
    @type path: string
    @:param: path - path to pickle file, DATASET_PATH will be used as declared in the configuration file
    @:rtype: dict
    """
    return pd.read_pickle(path)


def flattenhierarchy(dataset):
    """
    @type dataset: dict
    @param dataset: Dataset to be flattened must have the following structure:
        dict(one dataset){
            dict(any number of syndromes){
                dict(any number of subjects){
                    array(any number of images)[
                        float (320 dimensions)
                    ]
                }
            }
        }
        images must contain exactly 320 float numbers, one for each embedded dimension, ordered by dimension
    @:return Pandas dataframe will be returned with the following column structure:
        image_id(int), subject_id(int), syndrome_id(int), dimension_1 (NumPy float),
        dimension_2 (NumPy float), [...], dimension_320 (NumPy float)
    @:rtype: DataFrame
    """
    image_data = []
    # iterates dataset
    for syndrome in dataset:
        for subject in dataset[syndrome]:
            for image in dataset[syndrome][subject]:
                image_data.append([])
                # appends hierarchical info to individual images
                image_data[-1].append(int(image))
                image_data[-1].append(int(subject))
                image_data[-1].append(int(syndrome))
                for dimension in dataset[syndrome][subject][image]:
                    # iterates and appends dimensions
                    image_data[-1].append(dimension)
    dataframe = pd.DataFrame(image_data)
    # renames hierarchical data columns
    dataframe = dataframe.rename(columns={0: 'image_id', 1: 'subject_id', 2: 'syndrome_id'})
    # iterates dimensions
    for i in range(320):
        # renames dimension column
        dataframe = dataframe.rename(columns={i + 3: 'dimension_' + str(i + 1)})
    return dataframe


def cleandataframe(dataframe):
    """
    Removes items with missing dimensions or with incorrect dimension types
    @type dataframe: Pandas DataFrame
    @:param: dataframe - dataframe to be cleaned must have the following column structure:
        image_id(int), subject_id(int), syndrome_id(int), dimension_1 (NumPy float),
        dimension_2 (NumPy float), [...], dimension_320 (NumPy float)
    @:rtype: dict
    """
    response = dataframe
    # iterates rows
    for i, r in response.iterrows():
        # iterates items in row
        for x in r:
            # checks and deletes rows with missing or unusable info
            if type(x) is not float or pd.isnull(x):
                response = response.drop(index=i)
    return response


def reducedimensions(dataframe, params):
    """
    Reduces the dimensions of a dataframe to 2D
    @type dataframe: Pandas DataFrame
    @:param: dataframe - dataframe to be cleaned must have the following column structure:
        image_id(int), subject_id(int), syndrome_id(int), dimension 1 (NumPy float),
        dimension 2 (NumPy float), [...], dimension 320 (NumPy float)
    @:rtype: dict
    """
    # the line defining 'indexes' is turning a dataframe to an array to then turns the array back to a dataframe
    # there HAS to be a better way to select only the indexes
    # TODO: optimise redundant assignment
    indexes = pd.DataFrame(dataframe.iloc[:,0:3])
    embeddings = dataframe.iloc[:,3:322]

    tsne_result = pd.DataFrame(TSNE(
        n_components=2, random_state=RNG, perplexity=params['perplexity'],
        learning_rate=params['learning_rate'], early_exaggeration=params['early_exaggeration'],)
        .fit_transform(embeddings))
    final_result = pd.concat([indexes, tsne_result], axis=1).rename(columns={0: 'dimension_1', 1: 'dimension_2'})

    return final_result

def generatestatistics(dataframe):
    """
    Generates a statistics text
    @type dataframe: Pandas DataFrame
    @:param: dataframe - dataframe to be analyzed must have the following column structure:
        image_id(int), subject_id(int), syndrome_id(int), dimension_1 (NumPy float),
        dimension_2 (NumPy float), [...], dimension_320 (NumPy float)
    @:rtype: string
    """
    response_text = ""

    syndromes = dataframe['syndrome_id'].unique()
    subjects = dataframe['subject_id'].unique()
    images = dataframe['image_id'].unique()

    response_text += "Dataframe has " + str(len(syndromes)) + " syndromes, " + str(
        len(subjects)) + " subjects and " + str(len(images)) + " images.\n"

    # Counts images and subjects for each syndrome
    for syndrome in syndromes:
        local_images = dataframe[dataframe['syndrome_id'] == syndrome]
        local_subjects = local_images['subject_id'].unique()
        response_text += "\n\tSyndrome " + str(syndrome) + " has " + str(len(local_subjects)) + " subjects and " + str(
            len(local_images)) + " images"

    return response_text


def generatetables(euclidean_metrics, cosine_metrics):
    """
        Generates a xlsx (MS Excel) table with 3 columns [metric|euclidean_result|cosine_result] and
        7 rows [best_k|accuracy|top_k_accuracy|precision|recall|f1-score|roc_auc_score]
        """

    # Table structure
    df = pd.DataFrame(data={
        'metric': ['best_k',
                   'accuracy',
                   'top_' + str(TOP_K) + '_accuracy',
                   'precision',
                   'recall',
                   'f1_score',
                   'roc_auc_score',],
        'euclidean_result': [euclidean_metrics['best_k'],
                             euclidean_metrics['accuracy'],
                             euclidean_metrics['top_k_accuracy'],
                             euclidean_metrics['precision'],
                             euclidean_metrics['recall'],
                             euclidean_metrics['f1'],
                             euclidean_metrics['auc_score']],
        'cosine_result': [cosine_metrics['best_k'],
                          cosine_metrics['accuracy'],
                          cosine_metrics['top_k_accuracy'],
                          cosine_metrics['precision'],
                          cosine_metrics['recall'],
                          cosine_metrics['f1'],
                          cosine_metrics['auc_score']]
    })
    # Be sure to have openpyxl package installed
    df.to_excel(TABLE_FOLDER + "\\" + "Comparison metrics between euclidean and cosine.xlsx", index=False)

