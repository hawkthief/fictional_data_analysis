import os

# Set this to True to test for the best t-SNE parameters
TESTING_MODE = False

DATASET_NAME = 'mini_gm_public_v0.1.p'
DATASET_PATH = os.getcwd() + "\\" + DATASET_NAME
PLOT_FOLDER = os.getcwd() + "\\plots"
TABLE_FOLDER = os.getcwd() + "\\tables"

# All functions with a random seed will be set to this
RNG = 1337
# Values for t-SNE testing
TSNE_TESTING_PERPLEXITY = [55, 60, 65]
TSNE_TESTING_LEARNING_RATE = [7, 8, 9]
TSNE_TESTING_EARLY_EXAGGERATION = [50, 90, 100]
# Chosen Values
TSNE_PERPLEXITY = 60.0
TSNE_LEARNING_RATE = 8.0
TSNE_EARLY_EXAGGERATION = 90.0

K_VALUES = range(1, 16)
TOP_K = 3
