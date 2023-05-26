import os
import logging
import json
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader

import src.utilities.preprocess as pp
from src.utilities.utils import DEFAULT_DATASET_NAME

DATA_DIR = "../../data"


def get_data(dataset_name: str = DEFAULT_DATASET_NAME):
    data_path = download_data(dataset_name)
    data: pp.Dataset = GenericDataLoader(data_folder=data_path).load(split="test")
    documents, queries, qrels = data

    return documents, data_path


def get_corpus_path(dataset_name: str = DEFAULT_DATASET_NAME):
    data_path = download_data(dataset_name)
    corpus_path = os.path.join(data_path, "corpus.jsonl")

    return corpus_path


def download_data(dataset_name: str = DEFAULT_DATASET_NAME):
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    dataset_url = get_beir_dataset_url(dataset_name)

    if not os.path.exists(os.path.join(DATA_DIR, dataset_name)):
        data_path = util.download_and_unzip(dataset_url, DATA_DIR)
    else:
        data_path = os.path.join(DATA_DIR, dataset_name)

    return data_path


def get_beir_dataset_url(dataset_name: str = DEFAULT_DATASET_NAME):
    return f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"


def get_sample_data(file_path: str) -> pp.Documents:
    sample = {}
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            doc_id = data['id']
            text = data['text']
            sample[doc_id] = pp.Document(doc_id, text)
    return sample


def save_sample_data(file_path, docs):
    with open(file_path, 'w') as f:
        for doc_id, doc in docs.items():
            f.write(json.dumps({'title': doc['title'], 'text': doc['text']}) + '\n')
