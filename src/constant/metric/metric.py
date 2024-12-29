from enum import Enum


class Metric(Enum):
    AP = "ap"
    MAP = "map"
    NDCG = "ndcg"
    NO_CANDIDATE_COUNT = "no_candidate_count"
    RANKED_PREC = "ranked_prec"
    NEAR_CANDIDATE_RECALL = "near_candidate_recall"
    NEAR_CANDIDATE_PREC_COUNT = "near_candidate_prec_count"
    NEAR_CANDIDATE_RECALL_COUNT = "near_candidate_recall_count"