"""BEIR dataset download and loading."""
import logging
import os

from beir import util
from beir.datasets.data_loader import GenericDataLoader

logger = logging.getLogger(__name__)

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"


class BeirSplit:

    def __init__(self, corpus: dict, queries: dict, qrels: dict):
        self.corpus = corpus
        self.queries = queries
        self.qrels = qrels


def load_dataset(name: str, data_dir: str, train_split: str, eval_split: str) -> tuple[BeirSplit, BeirSplit]:
    dataset_dir = os.path.join(data_dir, name)
    if not os.path.exists(dataset_dir):
        os.makedirs(data_dir, exist_ok=True)
        zip_path = os.path.join(data_dir, f"{name}.zip")
        util.download_url(BEIR_URL.format(name), zip_path)
        util.unzip(zip_path, data_dir)

    corpus, train_queries, train_qrels = GenericDataLoader(data_folder=dataset_dir).load(split=train_split)
    _, eval_queries, eval_qrels = GenericDataLoader(data_folder=dataset_dir).load(split=eval_split)

    logger.info(
        "%s: %d docs, %d train queries, %d eval queries",
        name, len(corpus), len(train_queries), len(eval_queries),
    )
    return BeirSplit(corpus, train_queries, train_qrels), BeirSplit(corpus, eval_queries, eval_qrels)


def doc_text(corpus: dict, doc_id: str) -> str:
    doc = corpus[doc_id]
    title = doc.get("title", "").strip()
    text = doc.get("text", "").strip()
    return f"{title} {text}".strip() if title else text
