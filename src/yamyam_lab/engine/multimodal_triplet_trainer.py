"""Trainer for multimodal triplet embedding model.

This module implements the trainer for the multimodal triplet embedding model using
triplet loss for learning restaurant embeddings.
"""

import copy
import os
import pickle
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from yamyam_lab.data.multimodal_triplet import (
    MultimodalTripletDataset,
    create_multimodal_triplet_dataloader,
)
from yamyam_lab.engine.base_trainer import BaseTrainer
from yamyam_lab.loss.triplet import triplet_margin_loss_with_multiple_negatives
from yamyam_lab.model.embedding.multimodal_triplet import Model, MultimodalTripletConfig
from yamyam_lab.tools.plot import plot_diner_embedding_metrics


class MultimodalTripletTrainer(BaseTrainer):
    """Trainer for multimodal triplet embedding model.

    Extends the GraphTrainer pattern for triplet-based training of diner embeddings.
    Uses Recall@10 and MRR for validation with early stopping.

    Config is loaded from config/models/embedding/multimodal_triplet.yaml.
    Args can override specific values (epochs, lr, batch_size, patience).
    """

    def __init__(self, args):
        """Initialize trainer.

        Args:
            args: Parsed command-line arguments.
        """
        super().__init__(args)
        self.dataset: Optional[MultimodalTripletDataset] = None
        self.val_dataset: Optional[MultimodalTripletDataset] = None
        self.model: Optional[Model] = None
        self.multimodal_triplet_config: Optional[MultimodalTripletConfig] = None

        # Validation metrics history for plotting
        self.val_metrics_history: Dict[str, list] = {
            "recall@1": [],
            "recall@5": [],
            "recall@10": [],
            "recall@20": [],
            "mrr": [],
        }

    def _get_config(self, key: str, section: str = "training") -> Any:
        """Get config value, with args override if provided.

        Args:
            key: Config key name.
            section: Config section ('model', 'training', 'data').

        Returns:
            Value from args if set, otherwise from config.
        """
        # Check if args has a non-None override
        args_value = getattr(self.args, key, None)
        if args_value is not None:
            return args_value

        # Get from config
        config_section = getattr(self.config, section, {})
        return getattr(config_section, key, None)

    def load_data(self) -> None:
        """Load and prepare multimodal triplet dataset."""
        # Get data paths from config
        data_config = self.config.data
        features_path = data_config.features_path
        pairs_path = data_config.pairs_path
        category_mapping_path = data_config.category_mapping_path
        val_pairs_path = data_config.val_pairs_path

        # Get training config
        training_config = self.config.training
        batch_size = self._get_config("batch_size")
        num_workers = getattr(self.args, "num_workers", 4)
        random_seed = getattr(self.args, "random_seed", 42)

        # Create training dataloader
        self.train_loader, self.dataset = create_multimodal_triplet_dataloader(
            features_path=features_path,
            pairs_path=pairs_path,
            category_mapping_path=category_mapping_path,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            num_hard_negatives=training_config.num_hard_negatives,
            num_nearby_negatives=training_config.num_nearby_negatives,
            num_random_negatives=training_config.num_random_negatives,
            random_seed=random_seed,
        )

        # Create validation dataloader if val_pairs exists
        if os.path.exists(val_pairs_path):
            self.val_loader, self.val_dataset = create_multimodal_triplet_dataloader(
                features_path=features_path,
                pairs_path=val_pairs_path,
                category_mapping_path=category_mapping_path,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                num_hard_negatives=training_config.num_hard_negatives,
                num_nearby_negatives=training_config.num_nearby_negatives,
                num_random_negatives=training_config.num_random_negatives,
                random_seed=random_seed,
            )
        else:
            self.val_loader = None
            self.val_dataset = None
            self.logger.warning(
                f"Validation pairs not found at {val_pairs_path}. "
                "Validation will be skipped."
            )

        # Store data for later use
        self.data = {
            "num_diners": self.dataset.num_diners,
            "num_large_categories": len(set(self.dataset.large_category_ids.numpy())),
            "num_middle_categories": len(set(self.dataset.middle_category_ids.numpy())),
            "num_small_categories": len(set(self.dataset.small_category_ids.numpy())),
            "all_features": self.dataset.get_all_features(),
        }

        self.log_data_statistics()

    def log_data_statistics(self) -> None:
        """Log data statistics after loading."""
        self.logger.info("=" * 50)
        self.logger.info("Multimodal Triplet Embedding Data Statistics")
        self.logger.info("=" * 50)
        self.logger.info(f"Number of diners: {self.data['num_diners']}")
        self.logger.info(
            f"Number of large categories: {self.data['num_large_categories']}"
        )
        self.logger.info(
            f"Number of middle categories: {self.data['num_middle_categories']}"
        )
        self.logger.info(
            f"Number of small categories: {self.data['num_small_categories']}"
        )
        self.logger.info(f"Number of training pairs: {len(self.dataset)}")
        if self.val_dataset:
            self.logger.info(f"Number of validation pairs: {len(self.val_dataset)}")
        self.logger.info("=" * 50)

    def build_model(self) -> None:
        """Build multimodal triplet embedding model."""
        top_k_values = self.get_top_k_values()
        model_config = self.config.model

        # Create model config
        self.multimodal_triplet_config = MultimodalTripletConfig(
            num_large_categories=self.data["num_large_categories"],
            num_middle_categories=self.data["num_middle_categories"],
            num_small_categories=self.data["num_small_categories"],
            embedding_dim=model_config.embedding_dim,
            category_dim=model_config.category_dim,
            menu_dim=model_config.menu_dim,
            diner_name_dim=model_config.diner_name_dim,
            price_dim=model_config.price_dim,
            num_attention_heads=model_config.num_attention_heads,
            dropout=model_config.dropout,
            kobert_model_name=model_config.kobert_model_name,
            use_precomputed_menu_embeddings=model_config.use_precomputed_menu_embeddings,
            use_precomputed_name_embeddings=model_config.use_precomputed_name_embeddings,
            device=self.args.device,
            top_k_values=top_k_values,
            diner_ids=torch.arange(self.data["num_diners"]),
            recommend_batch_size=self.config.training.evaluation.recommend_batch_size,
        )

        # Create model
        self.model = Model(config=self.multimodal_triplet_config).to(self.args.device)

        self.logger.info(f"Model created with {self._count_parameters()} parameters")

    def _count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def build_metric_calculator(self) -> None:
        """Build metric calculator for multimodal triplet model.

        Uses custom metrics: Recall@K, MRR.
        """
        # Metrics will be computed inline in _evaluate_epoch
        pass

    def train_loop(self) -> None:
        """Training loop with early stopping based on Recall@10 and MRR."""
        training_config = self.config.training

        # Get hyperparameters (args override config)
        lr = self._get_config("lr")
        epochs = self._get_config("epochs")
        patience = self._get_config("patience")

        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=training_config.weight_decay,
        )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3, verbose=True
        )

        # Get loss hyperparameters from config
        margin = training_config.margin
        category_weight = training_config.category_weight
        gradient_clip = training_config.gradient_clip

        # Early stopping variables
        best_val_metric = -float("inf")
        best_val_epoch = -1
        best_model_weights = None
        patience_counter = patience

        # Move all features to device
        all_features = {
            k: v.to(self.args.device) for k, v in self.data["all_features"].items()
        }

        for epoch in range(epochs):
            self.logger.info(f"################## Epoch {epoch} ##################")

            # Training
            self.model.train()
            total_loss = 0.0
            num_batches = len(self.train_loader)

            for batch_idx, batch in enumerate(self.train_loader):
                optimizer.zero_grad()

                # Get batch indices
                anchor_indices = batch["anchor_indices"].to(self.args.device)
                positive_indices = batch["positive_indices"].to(self.args.device)
                negative_indices = batch["negative_indices"].to(
                    self.args.device
                )  # (B, num_negatives)
                anchor_categories = batch["anchor_categories"].to(self.args.device)
                positive_categories = batch["positive_categories"].to(self.args.device)
                negative_categories = batch["negative_categories"].to(self.args.device)

                # Get features for anchors, positives, and negatives
                anchor_features = self._get_features_by_indices(
                    all_features, anchor_indices
                )
                positive_features = self._get_features_by_indices(
                    all_features, positive_indices
                )

                # Compute anchor and positive embeddings
                anchor_emb = self.model(anchor_features)
                positive_emb = self.model(positive_features)

                # Batch all negatives into a single forward pass
                # negative_indices: (B, num_negatives) -> (B * num_negatives,)
                batch_size = negative_indices.size(0)
                num_negatives = negative_indices.size(1)
                flat_negative_indices = negative_indices.view(-1)

                # Get features for all negatives at once
                flat_negative_features = self._get_features_by_indices(
                    all_features, flat_negative_indices
                )

                # Single forward pass for all negatives
                flat_negative_emb = self.model(flat_negative_features)

                # Reshape to (B, num_negatives, embedding_dim)
                embedding_dim = flat_negative_emb.size(-1)
                negative_emb = flat_negative_emb.view(
                    batch_size, num_negatives, embedding_dim
                )

                # Compute batched triplet loss with all negatives
                batch_loss = triplet_margin_loss_with_multiple_negatives(
                    anchor=anchor_emb,
                    positive=positive_emb,
                    negatives=negative_emb,
                    anchor_category=anchor_categories,
                    positive_category=positive_categories,
                    negative_categories=negative_categories,
                    margin=margin,
                    category_weight=category_weight,
                )

                # Backward pass
                batch_loss.backward()

                # Gradient clipping
                if gradient_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)

                optimizer.step()

                total_loss += batch_loss.item()

                if batch_idx % 100 == 0:
                    self.logger.info(
                        f"Batch {batch_idx}/{num_batches}, "
                        f"Loss: {batch_loss.item():.4f}"
                    )

            # Average loss
            avg_loss = total_loss / num_batches
            self.model.tr_loss.append(avg_loss)
            self.logger.info(f"Epoch {epoch}: Average training loss: {avg_loss:.4f}")

            # Compute and store embeddings for evaluation
            self.model.compute_and_store_embeddings(
                all_features=all_features,
                batch_size=self.args.batch_size,
            )

            # Validation
            if self.val_loader is not None:
                val_metrics = self._evaluate_epoch(epoch)
                val_metric = val_metrics.get("recall@10", 0.0)

                # Track metrics history for plotting
                for metric_name, value in val_metrics.items():
                    if metric_name in self.val_metrics_history:
                        self.val_metrics_history[metric_name].append(value)

                # Update learning rate scheduler
                scheduler.step(val_metric)

                # Early stopping
                if val_metric > best_val_metric:
                    best_val_metric = val_metric
                    best_val_epoch = epoch
                    best_model_weights = copy.deepcopy(self.model.state_dict())
                    patience_counter = patience

                    # Save checkpoint
                    self._save_checkpoint(epoch, val_metrics)
                    self.logger.info(
                        f"New best Recall@10: {best_val_metric:.4f} at epoch {epoch}"
                    )
                else:
                    patience_counter -= 1
                    self.logger.info(
                        f"Recall@10 did not improve. Patience: {patience_counter}/{patience}"
                    )

                    if patience_counter <= 0:
                        self.logger.info(
                            f"Early stopping at epoch {epoch}. "
                            f"Best Recall@10: {best_val_metric:.4f} at epoch {best_val_epoch}"
                        )
                        break

        # Load best model weights
        if best_model_weights is not None:
            self.model.load_state_dict(best_model_weights)
            self.logger.info("Loaded best model weights")
            self._save_checkpoint(best_val_epoch, {"best": True})

    def _get_features_by_indices(
        self, all_features: Dict[str, Tensor], indices: Tensor
    ) -> Dict[str, Tensor]:
        """Extract features for given indices.

        Args:
            all_features: Dictionary of all feature tensors.
            indices: Tensor of indices to extract.

        Returns:
            Dictionary of feature tensors for the given indices.
        """
        return {
            "large_category_ids": all_features["large_category_ids"][indices],
            "middle_category_ids": all_features["middle_category_ids"][indices],
            "small_category_ids": all_features["small_category_ids"][indices],
            "menu_embeddings": all_features["menu_embeddings"][indices],
            "diner_name_embeddings": all_features["diner_name_embeddings"][indices],
            "price_features": all_features["price_features"][indices],
        }

    def _evaluate_epoch(self, epoch: int) -> Dict[str, float]:
        """Evaluate model on validation set.

        Computes Recall@K and MRR metrics.

        Args:
            epoch: Current epoch number.

        Returns:
            Dictionary of metric values.
        """
        self.model.eval()
        metrics = {}

        # Get all embeddings
        all_embeddings = self.model._embedding

        # Compute metrics on validation pairs
        val_pairs = self.val_dataset.pairs_df
        diner_idx_to_pos = self.val_dataset.diner_idx_to_position

        # Sample validation queries (anchor -> positive)
        num_samples = min(1000, len(val_pairs))
        sample_pairs = val_pairs.sample(n=num_samples, random_state=42)

        # Compute Recall@K and MRR
        k_values = [1, 5, 10, 20]
        recalls = {k: [] for k in k_values}
        reciprocal_ranks = []

        with torch.no_grad():
            for _, row in sample_pairs.iterrows():
                anchor_diner_idx = int(row["anchor_idx"])
                positive_diner_idx = int(row["positive_idx"])

                # Convert to position indices
                anchor_pos = diner_idx_to_pos.get(anchor_diner_idx)
                positive_pos = diner_idx_to_pos.get(positive_diner_idx)

                # Skip if either diner not in features
                if anchor_pos is None or positive_pos is None:
                    continue

                # Get anchor embedding
                anchor_emb = all_embeddings[anchor_pos : anchor_pos + 1]

                # Compute similarities with all diners
                similarities = self.model.similarity(anchor_emb, all_embeddings)
                similarities = similarities.squeeze(0)

                # Exclude anchor itself
                similarities[anchor_pos] = -float("inf")

                # Get ranked indices
                _, ranked_indices = torch.sort(similarities, descending=True)
                ranked_indices = ranked_indices.cpu().numpy()

                # Find position of positive
                pos_rank = np.where(ranked_indices == positive_pos)[0]
                if len(pos_rank) > 0:
                    rank = pos_rank[0] + 1  # 1-indexed

                    # Recall@K
                    for k in k_values:
                        recalls[k].append(1.0 if rank <= k else 0.0)

                    # Reciprocal rank
                    reciprocal_ranks.append(1.0 / rank)
                else:
                    for k in k_values:
                        recalls[k].append(0.0)
                    reciprocal_ranks.append(0.0)

        # Compute mean metrics
        for k in k_values:
            metrics[f"recall@{k}"] = np.mean(recalls[k])

        metrics["mrr"] = np.mean(reciprocal_ranks)

        # Log metrics
        self.logger.info(f"Validation metrics at epoch {epoch}:")
        for name, value in metrics.items():
            self.logger.info(f"  {name}: {value:.4f}")

        self.model.train()
        return metrics

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """Save model checkpoint.

        Args:
            epoch: Current epoch number.
            metrics: Dictionary of metric values.
        """
        file_name = self.config.post_training.file_name

        # Save model weights and config
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self.multimodal_triplet_config,
                "epoch": epoch,
                "metrics": metrics,
            },
            os.path.join(self.result_path, file_name.weight),
        )

        # Save training loss history
        pickle.dump(
            self.model.tr_loss,
            open(os.path.join(self.result_path, file_name.training_loss), "wb"),
        )

    def evaluate_validation(self) -> None:
        """Evaluate on validation set. Called after training."""
        if self.val_loader is not None:
            self.logger.info("Final validation evaluation:")
            self._evaluate_epoch(epoch=-1)

    def evaluate_test(self) -> None:
        """Evaluate on test set."""
        data_config = self.config.data
        test_pairs_path = data_config.test_pairs_path

        if not os.path.exists(test_pairs_path):
            self.logger.warning(
                f"Test pairs not found at {test_pairs_path}. Test evaluation skipped."
            )
            return

        # Temporarily swap validation dataset for test
        original_val_dataset = self.val_dataset

        _, self.val_dataset = create_multimodal_triplet_dataloader(
            features_path=data_config.features_path,
            pairs_path=test_pairs_path,
            category_mapping_path=data_config.category_mapping_path,
            batch_size=self._get_config("batch_size"),
            shuffle=False,
            num_workers=1,
        )

        self.logger.info("=" * 50)
        self.logger.info("Test Set Evaluation")
        self.logger.info("=" * 50)
        test_metrics = self._evaluate_epoch(epoch=-1)

        # Restore original validation dataset
        self.val_dataset = original_val_dataset

        return test_metrics

    def post_process(self) -> None:
        """Post-processing after training."""
        # Plot training metrics
        self.logger.info("Generating training plots...")
        plot_diner_embedding_metrics(
            tr_loss=self.model.tr_loss,
            val_metrics_history=self.val_metrics_history,
            parent_save_path=self.result_path,
        )
        self.logger.info(f"Saved plots to {self.result_path}")

        # Save validation metrics history
        pickle.dump(
            self.val_metrics_history,
            open(os.path.join(self.result_path, "val_metrics_history.pkl"), "wb"),
        )

        # Generate candidate similarities for all diners
        if getattr(self.args, "save_candidate", False):
            self.logger.info("Generating candidate similarities...")
            top_k = self.config.post_training.candidate_generation.top_k
            candidates_df = self.model.generate_candidates_for_each_diner(top_k)
            candidates_df.to_parquet(
                os.path.join(
                    self.result_path, self.config.post_training.file_name.candidate
                ),
                index=False,
            )
            self.logger.info(f"Saved {len(candidates_df)} candidate pairs")
