import os
import logging
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader

import src.utilities.preprocess as pp


def download_data(dataset_name):
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    data_dir = "../../data"
    dataset_url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"

    if not os.path.exists(os.path.join(data_dir, dataset_name)):
        data_path = util.download_and_unzip(dataset_url, data_dir)
    else:
        data_path = os.path.join(data_dir, dataset_name)

    # Load documents from the dataset.
    # Queries and ground truth are not needed, so they are referred as '_'
    data: pp.Dataset = GenericDataLoader(data_folder=data_path).load(split="test")
    documents, _, _ = data

    return documents, data_path
