import dask.dataframe as dd
from dask_ml.cluster import KMeans
from dask.diagnostics import ProgressBar
from tqdm.auto import tqdm
from sklearn.impute import SimpleImputer

import joblib
import pandas as pd
import os

# Read in all the part files in the folder
df = dd.read_csv("hhs_kmeans_data.csv", assume_missing=True).compute()
df = df.drop(["fips","county","state"], axis=1)
imp = SimpleImputer(strategy="mean")

imputed_df = imp.fit_transform(df)
df = dd.from_array(imputed_df)

# Define the range of cluster numbers to try
cluster_range = range(100, 3200, 100)
cluster_range = list(range(2,21,1)) + list(cluster_range)

# Define the progress bar
pbar = tqdm(total=len(cluster_range), desc="Clustering")


results_folder = "./kmeans_dd_hhs"
classifiers_folder = './kmeans_classifiers'
os.makedirs(results_folder, exist_ok=True)
os.makedirs(classifiers_folder, exist_ok=True)


# Define a function to train the model and save the classifier
def train_and_save_classifier(n_clusters):
    # Create the KMeans model with the specified number of clusters
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, init='k-means++', max_iter=10000)
    
        # Fit the model to the data
        labels = kmeans.fit(df).predict(df)

        #labels_df = pd.DataFrame(labels)
        #labels_df = labels.to_frame(name="cluster_label")
        #labels_df = labels_df.persist()
        #labels_df.to_csv(os.path.join(results_folder,f'kmeans_labels_{n_clusters}.csv'), index=False)
    
        kmeans_fname = os.path.join(classifiers_folder,f'kmeans_{n_clusters}.joblib')
        print("Writing {}".format(kmeans_fname))
        joblib.dump(kmeans, kmeans_fname)
    except:
        print("Something went wrong for k={}".format(n_clusters))
    
    pbar.update(1)
    return

# Train the models in parallel and track the progress with the progress bar
with ProgressBar():
    results = [train_and_save_classifier(n_clusters) for n_clusters in cluster_range]
#print("Some process went wrong")
#for n_clusters in tqdm(cluster_range):
#    try:
#        train_and_save_classifier(n_clusters)
#    except:
#        print("Something went wrong with k={}".format(n_clusters))
#        continue

# Close the progress bar
pbar.close()

