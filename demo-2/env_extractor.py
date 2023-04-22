import tensorflow_hub as hub
import pandas as pd

def load_env_variables(model_url, labels_path, labels_column):
    detector = hub.load(model_url)
    all_labels = pd.read_csv(labels_path, sep=';', index_col='ID')
    labels = all_labels[labels_column]

    return detector, labels