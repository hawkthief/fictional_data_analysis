from configurations import PLOT_FOLDER
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plottsne(dataframe,name):
    """
    Generates a plot of the dimensions in the dataframe trying to correlate them with their syndromes
    @type dataframe: Pandas DataFrame
    @:param: dataframe - dataframe to be cleaned must have the following column structure:
        image_id(int), subject_id(int), syndrome_id(int), dimension_1 (NumPy float), dimension_2 (NumPy float)
    """
    plt.style.use("dark_background")
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x=dataframe["dimension_1"], y=dataframe["dimension_2"],
                    hue=dataframe['syndrome_id'], palette="tab10", s=10, alpha=0.75)
    plt.title("t-SNE 2D Visualization of 320D Image Embeddings")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.savefig(PLOT_FOLDER + '\\'+name+'.png', dpi=1000)
    plt.close()


def rocauccurve(euclidian_metrics, cosine_metrics, syndromes):
    """
    Generates a plot of the ROC AUC curves for both Cosine and Euclidean distance.
    """

    y_true_euclidian = euclidian_metrics['y_true']
    y_score_euclidian = list(zip(*euclidian_metrics['y_score']))
    y_true_cosine = cosine_metrics['y_true']
    y_score_cosine = list(zip(*cosine_metrics['y_score']))

    false_positives_euclidean = []
    true_positives_euclidean = []
    auc_euclidean = []

    false_positives_cosine = []
    true_positives_cosine = []
    auc_cosine = []

    for i in range(len(syndromes)):
        # Euclidian data for ROC AUC
        fpe, tpe, _ = roc_curve(y_true_euclidian == syndromes[i], y_score_euclidian[i])
        roc_e = auc(fpe, tpe)
        false_positives_euclidean.append(fpe)
        true_positives_euclidean.append(np.interp(np.linspace(0, 1, 200), fpe, tpe))
        auc_euclidean.append(roc_e)
        # Cosine data for ROC AUC
        fpc, tpc, _ = roc_curve(y_true_cosine == syndromes[i], y_score_cosine[i])
        roc_c = auc(fpc, tpc)
        false_positives_cosine.append(fpc)
        true_positives_cosine.append(np.interp(np.linspace(0, 1, 200), fpc, tpc))
        auc_cosine.append(roc_c)

        # Builds individual syndrome plots
        plt.plot(fpe, tpe, label="AUC (Euclidian) = " + str(roc_e))
        plt.plot(fpc, tpc, label="AUC (Cosine) = " + str(roc_c))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title("ROC Curve for syndrome: " + str(syndromes[i]))
        plt.legend()
        plt.savefig(PLOT_FOLDER + '\\' + "ROC Curve for syndrome " + str(syndromes[i]) + '.png', dpi=1000)
        plt.close()

    # Calculates means and standard deviations for Euclidian data
    mean_true_positive_euclidean = np.mean(true_positives_euclidean, axis=0)
    std_true_positive_euclidean = np.std(true_positives_euclidean, axis=0)
    mean_false_positive_euclidean = np.linspace(0, 1, 200)
    mean_auc_euclidean = np.mean(auc_euclidean)
    # Calculates means and standard deviations for Cosine data
    mean_true_positive_cosine = np.mean(true_positives_cosine, axis=0)
    std_true_positive_cosine = np.std(true_positives_cosine, axis=0)
    mean_false_positive_cosine = np.linspace(0, 1, 200)
    mean_auc_cosine = np.mean(auc_cosine)

    # Plots overall graph with ROC AUC mean curve and standard deviation
    plt.plot(mean_false_positive_euclidean, mean_true_positive_euclidean, label="AUC (Euclidian) = " + str(mean_auc_euclidean))
    plt.plot(mean_false_positive_cosine, mean_true_positive_cosine, label="AUC (Cosine) = " + str(mean_auc_cosine))
    plt.fill_between(mean_false_positive_euclidean, mean_true_positive_euclidean - std_true_positive_euclidean,
                     mean_true_positive_euclidean + std_true_positive_euclidean, color='b', alpha=0.2)
    plt.fill_between(mean_false_positive_cosine, mean_true_positive_cosine - std_true_positive_cosine,
                     mean_true_positive_cosine + std_true_positive_cosine, color='y', alpha=0.2)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC Curve with standard deviation")
    plt.legend()
    plt.savefig(PLOT_FOLDER + '\\'+"ROC Curve KNN.png", dpi=1000)
    plt.close()
