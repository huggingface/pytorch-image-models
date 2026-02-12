"""Evaluation task abstractions.

This module provides EvalTask classes that mirror TrainingTask but for evaluation.
Each EvalTask receives the full trainable_module (or EMA version) and handles:
- Per-batch forward passes with metric accumulation
- Running metric computation for logging during eval
- Final metric computation at end of eval epoch

Usage pattern:
    eval_task = task.get_eval_task(use_ema=True)
    eval_task.reset()  # Clear accumulators before eval epoch

    for input, target in eval_loader:
        result = eval_task(input, target)
        # result contains running metrics for logging (e.g., 'acc1', 'acc5')

    metrics = eval_task.compute_metrics()  # Final metrics for the epoch
"""
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class EvalTask(nn.Module):
    """Base class for evaluation tasks with metric accumulation.

    Receives the full trainable_module (or EMA version) so it can:
    - Access core model via .trainable_module.model for inference
    - Access other components (projector, decoder) for eval loss if needed
    - Decide internally what it needs for evaluation

    Subclasses must implement:
    - reset(): Clear accumulated metrics
    - forward(): Process batch and accumulate metrics, return running values
    - compute_metrics(): Compute final metrics from accumulated values

    Args:
        trainable_module: The trainable module from TrainingTask (or EMA version)
    """

    def __init__(self, trainable_module: nn.Module):
        super().__init__()
        self.trainable_module = trainable_module
        self.trainable_module.eval()

    @property
    def model(self) -> nn.Module:
        """Convenience access to core model.

        Returns .model attribute of trainable_module if it exists,
        otherwise returns trainable_module itself.
        """
        return getattr(self.trainable_module, 'model', self.trainable_module)

    def reset(self) -> None:
        """Reset accumulated metrics. Call before each eval epoch."""
        raise NotImplementedError

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Process batch and accumulate metrics.

        Args:
            input: Input tensor [B, C, H, W]
            target: Target labels [B]

        Returns:
            Dictionary with running metrics for logging. All tasks should return
            at least 'output' for compatibility. May include running accuracy, loss, etc.
            Some tasks may only return basic info if per-step metrics are expensive.
        """
        raise NotImplementedError

    def compute_metrics(self) -> Dict[str, float]:
        """Compute final metrics from accumulated values.

        Call at end of eval loop. Returns metrics computed over the full
        evaluation set (e.g., accuracy, F1, mAP).

        Returns:
            Dictionary of metric_name -> metric_value (floats, not tensors)
        """
        raise NotImplementedError


class ClassificationEvalTask(EvalTask):
    """Evaluation for classification models with accuracy metrics.

    For classification, trainable_module IS the model. Accumulates predictions
    and targets for computing accuracy metrics.

    Running metrics: acc1, acc5 (incremental, updated each batch)
    Final metrics: acc1, acc5, eval_samples

    Args:
        trainable_module: The trainable module from TrainingTask (or EMA version)
        topk: Tuple of k values for top-k accuracy (default: (1, 5))
        store_predictions: What to store for advanced metrics:
            - False: Don't store (default, memory efficient)
            - 'argmax': Store predicted class indices (for F1, confusion matrix)
            - 'logits': Store full logits (for calibration, soft metrics)

    Example:
        >>> eval_task = ClassificationEvalTask(model)
        >>> eval_task.reset()
        >>> for input, target in loader:
        >>>     result = eval_task(input, target)
        >>>     print(f"Running acc1: {result['acc1'].item():.2f}%")
        >>> metrics = eval_task.compute_metrics()
        >>> print(f"Final acc1: {metrics['acc1']:.2f}%")
    """

    def __init__(
            self,
            trainable_module: nn.Module,
            topk: Tuple[int, ...] = (1, 5),
            store_predictions: Union[bool, str] = False,
    ):
        self.topk = topk
        self.store_predictions = store_predictions
        super().__init__(trainable_module)
        self.reset()

    def reset(self) -> None:
        """Reset accumulated metrics."""
        self._correct = {k: 0 for k in self.topk}
        self._total = 0

        if self.store_predictions:
            self._predictions: List[torch.Tensor] = []
            self._targets: List[torch.Tensor] = []

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with metric accumulation.

        Args:
            input: Input tensor [B, C, H, W]
            target: Target labels [B]

        Returns:
            Dictionary containing:
                - 'output': Model logits [B, num_classes]
                - 'acc1': Running top-1 accuracy (percentage)
                - 'acc5': Running top-5 accuracy (percentage, if in topk)
        """
        with torch.no_grad():
            output = self.trainable_module(input)
            batch_size = target.size(0)

            # Compute batch accuracy
            maxk = min(max(self.topk), output.size(1))
            _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            # Accumulate correct counts
            for k in self.topk:
                if k <= maxk:
                    correct_k = correct[:k].reshape(-1).float().sum().item()
                    self._correct[k] += correct_k
            self._total += batch_size

            # Store predictions if requested
            if self.store_predictions == 'logits':
                self._predictions.append(output.cpu())
                self._targets.append(target.cpu())
            elif self.store_predictions == 'argmax':
                self._predictions.append(output.argmax(dim=1).cpu())
                self._targets.append(target.cpu())

            # Return running metrics
            result = {'output': output}
            for k in self.topk:
                key = f'acc{k}'
                if self._total > 0:
                    result[key] = torch.tensor(100.0 * self._correct[k] / self._total)
                else:
                    result[key] = torch.tensor(0.0)

            return result

    def compute_metrics(self) -> Dict[str, float]:
        """Compute final accuracy metrics.

        Returns:
            Dictionary with acc1, acc5 (if in topk), and sample count
        """
        if self._total == 0:
            metrics = {f'acc{k}': 0.0 for k in self.topk}
            metrics['eval_samples'] = 0.0
            return metrics

        metrics = {}
        for k in self.topk:
            metrics[f'acc{k}'] = 100.0 * self._correct[k] / self._total
        metrics['eval_samples'] = float(self._total)

        return metrics

    def get_predictions(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get all stored predictions and targets.

        Only available if store_predictions was set to 'argmax' or 'logits'.

        Returns:
            Tuple of (predictions, targets)
            - If store_predictions='logits': predictions are [N, num_classes]
            - If store_predictions='argmax': predictions are [N]

        Raises:
            RuntimeError: If store_predictions was False or no data accumulated
        """
        if not self.store_predictions:
            raise RuntimeError("Predictions not stored. Set store_predictions='argmax' or 'logits'.")
        if not self._predictions:
            raise RuntimeError("No predictions accumulated. Run forward() first.")

        return torch.cat(self._predictions, dim=0), torch.cat(self._targets, dim=0)


class SSLEvalTask(EvalTask):
    """Evaluation for SSL models with feature extraction and optional KNN.

    Extracts and accumulates features from the core model for downstream
    evaluation like KNN classification or linear probing.

    Always accumulates features internally. If gallery is set via set_gallery(),
    computes KNN accuracy in compute_metrics().

    Running metrics: num_samples (KNN is expensive per-batch)
    Final metrics: knn_acc (if gallery set), eval_samples

    Args:
        trainable_module: The trainable module from TrainingTask (or EMA version)
        pool: Pooling strategy for features ('avg', 'last', 'cls', None)
        knn_k: Number of neighbors for KNN evaluation (default: 20)
        knn_temperature: Temperature for weighted KNN voting (default: 0.07)

    Example:
        >>> eval_task = SSLEvalTask(trainable_module, pool='avg')
        >>> eval_task.reset()
        >>> for input, target in loader:
        >>>     result = eval_task(input, target)
        >>> metrics = eval_task.compute_metrics()
        >>> print(f"Samples: {metrics['eval_samples']}")
    """

    def __init__(
            self,
            trainable_module: nn.Module,
            pool: Optional[str] = 'avg',
            knn_k: int = 20,
            knn_temperature: float = 0.07,
    ):
        self.pool = pool
        self.knn_k = knn_k
        self.knn_temperature = knn_temperature
        super().__init__(trainable_module)
        self.reset()

    def reset(self) -> None:
        """Reset accumulated features and targets."""
        self._features: List[torch.Tensor] = []
        self._targets: List[torch.Tensor] = []
        self._total = 0
        self._gallery_features: Optional[torch.Tensor] = None
        self._gallery_targets: Optional[torch.Tensor] = None

    def set_gallery(self, features: torch.Tensor, targets: torch.Tensor) -> None:
        """Set gallery (reference) features for KNN evaluation.

        Call this AFTER extracting features from training set,
        BEFORE running eval on validation set.

        Args:
            features: Gallery features [N_gallery, D] (should be L2 normalized)
            targets: Gallery labels [N_gallery]
        """
        self._gallery_features = features
        self._gallery_targets = targets

    def _pool_features(self, features: torch.Tensor) -> torch.Tensor:
        """Apply pooling strategy to features.

        Args:
            features: Feature tensor, typically [B, N, D] for ViT or [B, D] for CNN

        Returns:
            Pooled features [B, D]
        """
        if features.dim() == 2:
            return features

        if features.dim() != 3:
            raise ValueError(f"Expected 2D or 3D features, got shape {features.shape}")

        if self.pool == 'avg':
            return features.mean(dim=1)
        elif self.pool == 'last':
            return features[:, -1]
        elif self.pool == 'cls':
            return features[:, 0]
        elif self.pool is None:
            return features
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pool}")

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Extract features and accumulate for KNN evaluation.

        Args:
            input: Input tensor [B, C, H, W] or [B, V, C, H, W] for multi-view
            target: Target labels [B]

        Returns:
            Dictionary containing:
                - 'features': Extracted features [B, D]
                - 'target': Target labels [B]
                - 'num_samples': Running total of accumulated samples
        """
        with torch.no_grad():
            # Handle multi-view input by taking first view
            if input.dim() == 5:
                input = input[:, 0]

            # Get features from core model
            features = self.model.forward_features(input)

            # Apply pooling
            features = self._pool_features(features)

            # L2 normalize features for KNN
            features = F.normalize(features, dim=1)

            # Accumulate (store on CPU to save GPU memory)
            self._features.append(features.cpu())
            self._targets.append(target.cpu())
            self._total += target.size(0)

            return {
                'features': features,
                'target': target,
                'num_samples': torch.tensor(self._total),
            }

    def compute_metrics(self) -> Dict[str, float]:
        """Compute KNN accuracy from accumulated features.

        If gallery was set via set_gallery(), computes KNN accuracy.
        Otherwise just returns sample count.

        Returns:
            Dictionary with knn_acc (if gallery set) and eval_samples
        """
        metrics = {'eval_samples': float(self._total)}

        if self._total == 0:
            return metrics

        if self._gallery_features is not None:
            query_features = torch.cat(self._features, dim=0)
            query_targets = torch.cat(self._targets, dim=0)
            metrics['knn_acc'] = self._compute_knn(
                self._gallery_features,
                self._gallery_targets,
                query_features,
                query_targets,
            )

        return metrics

    def _compute_knn(
            self,
            gallery_features: torch.Tensor,
            gallery_targets: torch.Tensor,
            query_features: torch.Tensor,
            query_targets: torch.Tensor,
    ) -> float:
        """Compute KNN accuracy.

        Uses weighted voting with temperature scaling.

        Args:
            gallery_features: Gallery features [N_gallery, D]
            gallery_targets: Gallery labels [N_gallery]
            query_features: Query features [N_query, D]
            query_targets: Query labels [N_query]

        Returns:
            KNN accuracy as percentage
        """
        num_query = query_features.size(0)
        num_classes = int(max(gallery_targets.max().item(), query_targets.max().item())) + 1
        k = min(self.knn_k, gallery_features.size(0))

        # Process in chunks to avoid OOM
        chunk_size = 256
        correct = 0

        for i in range(0, num_query, chunk_size):
            chunk_features = query_features[i:i + chunk_size]
            chunk_targets = query_targets[i:i + chunk_size]

            # Compute similarities [chunk_size, N_gallery]
            sim = chunk_features @ gallery_features.t()

            # Get top-k neighbors
            sim_topk, indices_topk = sim.topk(k, dim=1)

            # Get neighbor labels
            neighbor_labels = gallery_targets[indices_topk]

            # Weighted voting with temperature
            weights = (sim_topk / self.knn_temperature).softmax(dim=1)

            # Aggregate votes per class
            votes = torch.zeros(chunk_features.size(0), num_classes)
            for c in range(num_classes):
                mask = (neighbor_labels == c).float()
                votes[:, c] = (weights * mask).sum(dim=1)

            # Predict class with most votes
            predictions = votes.argmax(dim=1)
            correct += (predictions == chunk_targets).sum().item()

        return 100.0 * correct / num_query

    def get_features(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get all accumulated features and targets.

        Useful for extracting gallery features or for linear probe training.

        Returns:
            Tuple of (features [N, D], targets [N])

        Raises:
            RuntimeError: If no features have been accumulated
        """
        if not self._features:
            raise RuntimeError("No features accumulated. Run forward() first.")

        return torch.cat(self._features, dim=0), torch.cat(self._targets, dim=0)
