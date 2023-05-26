import abc
import dataclasses
from typing import Tuple, Type, Sequence, Any, TypeVar

import pyspark as ps
import pyspark.sql as psql
from pyspark.ml.linalg import SparseVector

from src.utilities.utils import AVAILABLE_CORES

DocId: Type = Any

SparseScores: Type = SparseVector

Document: Type = Tuple[DocId, SparseScores]

DocsSimilarity: Type = Tuple[DocId, DocId, float]
"""
(doc_id, doc_id, similarity_score) tuple
"""


@dataclasses.dataclass
class Results:
    similar_docs: Sequence[DocsSimilarity]
    """
    Sequence containing tuples of document pairs and their similarity
    """

    time: float
    """
    Time taken [seconds] to compute the results
    """


class APDSS(abc.ABC):
    @abc.abstractmethod
    def apdss(
            self,
            spark: psql.SparkSession,
            docs_scores_df: psql.DataFrame,
            threshold: float,
            num_partitions: int = AVAILABLE_CORES
    ) -> Results:
        pass


def cosine_sim(a: SparseVector, b: SparseVector):
    a_l2 = a.norm(2)
    b_l2 = b.norm(2)
    return float(a.dot(b) / (a_l2*b_l2)) if a_l2*b_l2 != 0 else 0


_T = TypeVar("_T")


def repartition_if_required(rdd: ps.RDD[_T], expected_num_partitions: int) -> ps.RDD[_T]:
    if rdd.getNumPartitions() != expected_num_partitions:
        return rdd.repartition(numPartitions=expected_num_partitions)
