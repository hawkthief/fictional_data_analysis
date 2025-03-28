import numpy as np
from scipy.spatial.distance import cosine
from configurations import RNG, TOP_K
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, f1_score, top_k_accuracy_score, precision_score, recall_score


def cosinedistance(x, y):
    """
    Calculates cosine distance between x and y.
    """
    # Import from SCIPY.SPATIAL.DISTANCE, there are other functions named cosine in that library.
    return cosine(x, y)


def evaluateknn(k_values, dataframe, cosine=False, log=True):
    """
    Runs and evaluates KNN classifier for a given data set with 10-fold cross-validation.
    @:param: k_values - K values as defined in the configuration file
    @:param: dataframe - dataframe to be cleaned must have the following column structure:
        image_id(int), subject_id(int), syndrome_id(int), dimension_1 (NumPy float), dimension_2 (NumPy float)
    @:return: Dictionary will have the following structure:
    {'y_true', 'y_score', 'f1', 'auc_score', 'best_k', 'accuracy', 'top_k_accuracy', 'precision', 'recall'}
    @:rtype: dict
    """
    embeddings = dataframe.iloc[:, 3:5]
    labels = dataframe['syndrome_id']
    best_k = None
    best_score = 0
    best_knn = None
    results = {}
    k_fold = KFold(n_splits=10, shuffle=True, random_state=RNG)

    for k in k_values:
        if cosine:
            knn = KNeighborsClassifier(n_neighbors=k, metric=cosinedistance)
        else:
            knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')

        # Cross validation
        scores = cross_val_score(knn, embeddings, labels, cv=k_fold, scoring='accuracy')
        avg_score = np.mean(scores)
        results[k] = avg_score

        if avg_score > best_score:
            best_knn = knn
            best_score = avg_score
            best_k = k

    # Returns only the accuracy related to the best_k
    result = {'best_k': best_k, 'accuracy': results[best_k]}
    if log:
        print("Best K (" + ("Cosine" if cosine else "Euclidian") + "):" + str(best_k) + " Accuracy:" + str(
            results[best_k]))
    if best_knn:
        return calculatemetrics(best_knn, embeddings, labels, result, cosine, log)
    else:
        return None


def calculatemetrics(knn, embeddings, labels, result, cosine=False, log=True):
    """
    Runs and evaluates KNN predictions, calculating the mean results between folds.
    @:return: Dictionary will have the following structure:
    {'y_true', 'y_score', 'f1', 'auc_score', 'best_k', 'accuracy', 'top_k_accuracy', 'precision', 'recall'}
    @:rtype: dict
    """
    k_fold = KFold(n_splits=10, shuffle=True, random_state=RNG)

    embeddings = embeddings.to_numpy()
    labels = labels.to_numpy()

    y_true = []
    y_pred = []
    y_prob = []
    f1 = []
    auc_score = []
    top_k_accuracy = []
    precision = []
    recall = []

    for train_idx, test_idx in k_fold.split(embeddings):
        knn.fit(embeddings[train_idx], labels[train_idx])
        predictions = knn.predict(embeddings[test_idx])
        probabilities = knn.predict_proba(embeddings[test_idx])

        y_true.extend(labels[test_idx])
        y_pred.extend(predictions)
        y_prob.extend(probabilities)

        f1.extend([f1_score(y_true, y_pred, average="weighted")])
        auc_score.extend([roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")])
        top_k_accuracy.extend([top_k_accuracy_score(y_true, y_prob, k=TOP_K)])
        precision.extend([precision_score(y_true, y_pred, average="weighted")])
        recall.extend([recall_score(y_true, y_pred, average="weighted")])

    f1 = np.mean(f1)
    auc_score = np.mean(auc_score)
    top_k_accuracy = np.mean(top_k_accuracy)
    precision = np.mean(precision)
    recall = np.mean(recall)

    if log:
        print("F1 Score (" + ("Cosine" if cosine else "Euclidian") + "):" + str(f1) + " Accuracy:" + str(auc_score) +
              "Top " + str(TOP_K) + " Accuracy:" + str(top_k_accuracy))

    return {'y_true': y_true, 'y_score': y_prob, 'f1': f1, 'auc_score': auc_score, 'best_k': result['best_k'],
            'accuracy': result['accuracy'], 'top_k_accuracy': top_k_accuracy, 'precision': precision, 'recall': recall}
