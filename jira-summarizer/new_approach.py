from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import ParameterGrid
from sklearn.cluster import KMeans

TEXT = ""
model = SentenceTransformer(r"sentence-transformers/paraphrase-MiniLM-L6-v2")


def compile_text(x):
    text = (
        f"Jira Id: {x['Issue key']} "
        f"Assignee: {x['Assignee name']} "
        f"Status: {x['Status']} "
        f"Description: {x['Description']} "
        f"Comments: {x['Comments']} "
    )

    text.replace('\n', '')
    text.replace('\r', '')
    return text


def output_embedding(txt):
    embd = model.encode(txt)
    print(embd.shape)
    return pd.DataFrame(embd)


def preprocess_text(x):
    global TEXT
    TEXT += compile_text(x) + "\n"


def load_embeddings(filename):
    pd.options.display.max_colwidth = 10000
    df = pd.read_json('sample_comments/' + filename, encoding="utf8")
    df.apply(lambda x: preprocess_text(x), axis=1)
    jira_raw = output_embedding(TEXT)
    # checking data shape
    print(jira_raw.shape)
    row, col = jira_raw.shape
    print(f'There are {row} rows and {col} columns')

    # to work on copy of the data
    jira_raw_scaled = jira_raw.copy()

    # Scaling the data to keep the different attributes in same range.
    jira_raw_scaled[jira_raw_scaled.columns] = StandardScaler().fit_transform(jira_raw_scaled)
    print(jira_raw_scaled.describe())

    return jira_raw_scaled



def kmean_hyper_param_tuning(data):
    """
    Hyper parameter tuning to select the best from all the parameters on the basis of silhouette_score.

    :param data: dimensionality reduced data after applying PCA
    :return: best number of clusters for the model (used for KMeans n_clusters)
    """
    # candidate values for our number of cluster
    parameters = [2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40]

    # instantiating ParameterGrid, pass number of clusters as input
    parameter_grid = ParameterGrid({'n_clusters': parameters})

    best_score = -1
    kmeans_model = KMeans()  # instantiating KMeans model
    silhouette_scores = []

    # evaluation based on silhouette_score
    for p in parameter_grid:
        kmeans_model.set_params(**p)  # set current hyper parameter
        kmeans_model.fit(data)  # fit model on wine dataset, this will find clusters based on parameter p

        ss = metrics.silhouette_score(data, kmeans_model.labels_)  # calculate silhouette_score
        silhouette_scores += [ss]  # store all the scores

        print('Parameter:', p, 'Score', ss)

        # check p which has the best score
        if ss > best_score:
            best_score = ss
            best_grid = p

    # plotting silhouette score
    plt.bar(range(len(silhouette_scores)), list(silhouette_scores), align='center', color='#722f59', width=0.5)
    plt.xticks(range(len(silhouette_scores)), list(parameters))
    plt.title('Silhouette Score', fontweight='bold')
    plt.xlabel('Number of Clusters')
    plt.show()

    return best_grid['n_clusters']


def visualizing_results(pca_result, label, centroids_pca):
    """ Visualizing the clusters

    :param pca_result: PCA applied data
    :param label: K Means labels

    """
    # ------------------ Using Matplotlib for plotting-----------------------
    x = pca_result[:, 0]
    y = pca_result[:, 1]

    plt.scatter(x, y, c=label, alpha=0.5, s=200)  # plot different colors per cluster
    plt.title('Wine clusters')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')

    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker='X', s=200, linewidths=1.5,
                color='red', edgecolors="black", lw=1.5)

    plt.show()


def main():
    print("1. Loading the dataset\n")
    data_scaled = load_embeddings('data.json')
    # fitting KMeans
    clustering_model = KMeans(n_clusters=40)
    clustering_model.fit(data_scaled)

    arl = {i: np.where(clustering_model.labels_ == i)[0] for i in range(clustering_model.n_clusters)}
    df = pd.read_json('sample_comments/data.json', encoding="utf8")

    arl1 = {}
    for i in range(clustering_model.n_clusters):
        arr = arl[i]
        for a in range(len(arr)):
            arl1[i][a] = df.loc[i].at['Jira Id']

    print(arl1)


if __name__ == "__main__":
    main()