"""CatBoost classifier for category classification."""

from catboost import CatBoostClassifier

from yamyam_lab.model.classification.base_classifier import BaseClassifier
from yamyam_lab.model.embedding.base_embedder import BaseEmbedder


class CatBoostCategoryClassifier(BaseClassifier):
    """
    CatBoost-based category classifier.

    Uses gradient boosting on text embeddings for multi-class classification.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        iterations: int = 300,
        learning_rate: float = 0.1,
        depth: int = 8,
        l2_leaf_reg: float = 3.0,
        early_stopping_rounds: int = 30,
        verbose: int = 50,
        **kwargs,
    ):
        super().__init__(embedder=embedder, **kwargs)

        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose

    @property
    def name(self) -> str:
        return f"catboost_{self.embedder.name}"

    def build_model(self) -> None:
        """Build CatBoost classifier."""
        print("\nBuilding CatBoost model...")

        self.model = CatBoostClassifier(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            random_seed=self.random_state,
            verbose=self.verbose,
            early_stopping_rounds=self.early_stopping_rounds,
            task_type="CPU",
            thread_count=-1,
        )

        print(f"  Iterations: {self.iterations}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Depth: {self.depth}")

    def train_model(self) -> None:
        """Train CatBoost model."""
        print("\nTraining CatBoost model...")
        print(f"  X_train: {self.X_train.shape}")
        print(f"  X_val: {self.X_val.shape}")

        self.model.fit(
            self.X_train,
            self.data.y_train,
            eval_set=(self.X_val, self.data.y_val),
            verbose=self.verbose,
        )

        print(f"  Best iteration: {self.model.best_iteration_}")
