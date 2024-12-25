from typing import Dict, Tuple, List
import pickle
import networkx as nx
from numpy.typing import NDArray

import torch
from torch import Tensor

from embedding.node2vec import Node2Vec
from evaluation.qualitative.base_qualitative_evaluation import BaseQualitativeEvaluation
from tools.utils import convert_tensor
from tools.parse_args import parse_args_eval
from constant.evaluation.qualitative import QualitativeReviewerId


class Node2VecQualitativeEvaluation(BaseQualitativeEvaluation):
    def __init__(
            self,
            model_path: str,
            user_ids: Tensor,
            diner_ids: Tensor,
            graph: nx.Graph,
            num_nodes: int,
            embedding_dim: int,
            user_mapping: Dict[int, int],
            diner_mapping: Dict[int, int],
    ):
        super().__init__(
            user_mapping=user_mapping,
            diner_mapping=diner_mapping,
        )
        self.model = Node2Vec(
            user_ids=user_ids,
            diner_ids=diner_ids,
            graph=graph,
            embedding_dim=embedding_dim, # trained model embedding dim
            walk_length=20, # dummy value
            context_size=10, # dummy value
            num_nodes=num_nodes,
            inference=True,
        )

        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()

    def _recommend(
            self,
            user_id: Tensor,
            tr_liked_diners: List[int],
            latitude: float = None,
            longitude: float = None,
            top_k: int = 10,
    ) -> Tuple[NDArray, NDArray]:
        return self.model._recommend(
            user_id=user_id,
            already_liked_item_id=tr_liked_diners,
            latitude=latitude,
            longitude=longitude,
            top_k=top_k,
        )


if __name__ == "__main__":
    args = parse_args_eval()

    data = pickle.load(open(args.data_obj_path, "rb"))
    num_nodes = data["num_users"] + data["num_diners"]

    train_liked = convert_tensor(data["X_train"], list)
    val_liked = convert_tensor(data["X_val"], list)

    # qualitative evaluation
    qualitative_eval = Node2VecQualitativeEvaluation(
        model_path=args.model_path,
        user_ids=torch.tensor(list(data["user_mapping"].values())),
        diner_ids=torch.tensor(list(data["diner_mapping"].values())),
        graph=nx.Graph(), # dummy graph
        num_nodes=num_nodes,
        embedding_dim=args.embedding_dim,
        user_mapping=data["user_mapping"],
        diner_mapping=data["diner_mapping"],
    )

    for enum in QualitativeReviewerId:
        reviewer_id = enum.value
        reviewer_name = enum.name
        reviewer_id_mapping = data["user_mapping"].get(reviewer_id)
        if reviewer_id_mapping is None:
            print(f"reviewer {reviewer_name} not existing in training dataset")
            continue
        qualitative_eval.recommend(
            user_id=reviewer_id,
            user_name=reviewer_name,
            tr_liked_diners=train_liked[reviewer_id_mapping],
            val_liked_diners=val_liked[reviewer_id_mapping],
            top_k=10,
        )