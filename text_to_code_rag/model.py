import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModel


class BiEncoderModel(pl.LightningModule):
    """Bi-encoder model for text-to-code semantic search"""

    def __init__(
        self,
        text_model_name: str = "distilbert-base-uncased",
        code_model_name: str = "microsoft/codebert-base",
        embedding_dim: int = 768,
        learning_rate: float = 2e-5,
        temperature: float = 0.07,
        recall_k_values: list[int] = None,
        freeze_encoders: bool = False,
    ):
        """Initialize bi-encoder model.

        Args:
            text_model_name: HuggingFace model name for text encoder
            code_model_name: HuggingFace model name for code encoder
            embedding_dim: Dimension of embeddings
            learning_rate: Learning rate for optimizer
            temperature: Temperature for contrastive loss
            recall_k_values: List of K values for Recall@K metric
            freeze_encoders: If True, freeze encoders and train only projection layers
        """
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.temperature = temperature
        self.recall_k_values = recall_k_values or [1, 5, 10]
        self.freeze_encoders = freeze_encoders

        self.text_encoder = AutoModel.from_pretrained(text_model_name, use_safetensors=True)
        self.code_encoder = AutoModel.from_pretrained(code_model_name, use_safetensors=True)

        self.text_projection = nn.Linear(self.text_encoder.config.hidden_size, embedding_dim)
        self.code_projection = nn.Linear(self.code_encoder.config.hidden_size, embedding_dim)

        if self.freeze_encoders:
            self._freeze_encoders()

    def _freeze_encoders(self):
        """Freeze text and code encoders, keeping only projection layers trainable."""
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.code_encoder.parameters():
            param.requires_grad = False

        print("Encoders frozen. Only projection layers will be trained.")

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode text descriptions.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            Text embeddings
        """
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)

        pooled_output = outputs.last_hidden_state[:, 0, :]
        embeddings = self.text_projection(pooled_output)

        return F.normalize(embeddings, p=2, dim=1)

    def encode_code(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode code snippets.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            Code embeddings
        """
        outputs = self.code_encoder(input_ids=input_ids, attention_mask=attention_mask)

        pooled_output = outputs.last_hidden_state[:, 0, :]
        embeddings = self.code_projection(pooled_output)

        return F.normalize(embeddings, p=2, dim=1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            batch: Batch dictionary with text and code inputs

        Returns:
            Dictionary with embeddings
        """
        text_embeddings = self.encode_text(batch["text_input_ids"], batch["text_attention_mask"])
        code_embeddings = self.encode_code(batch["code_input_ids"], batch["code_attention_mask"])

        return {"text_embeddings": text_embeddings, "code_embeddings": code_embeddings}

    def compute_contrastive_loss(
        self, text_embeddings: torch.Tensor, code_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Compute InfoNCE contrastive loss.

        Args:
            text_embeddings: Text embeddings [batch_size, embedding_dim]
            code_embeddings: Code embeddings [batch_size, embedding_dim]

        Returns:
            Contrastive loss value
        """
        similarity_matrix = torch.matmul(text_embeddings, code_embeddings.T) / self.temperature

        batch_size = similarity_matrix.size(0)
        labels = torch.arange(batch_size, device=similarity_matrix.device)

        loss_text_to_code = F.cross_entropy(similarity_matrix, labels)
        loss_code_to_text = F.cross_entropy(similarity_matrix.T, labels)

        return (loss_text_to_code + loss_code_to_text) / 2

    def compute_recall_at_k(
        self, text_embeddings: torch.Tensor, code_embeddings: torch.Tensor
    ) -> dict[str, float]:
        """Compute Recall@K metrics.

        Args:
            text_embeddings: Text embeddings
            code_embeddings: Code embeddings

        Returns:
            Dictionary with Recall@K for different K values
        """
        similarity_matrix = torch.matmul(text_embeddings, code_embeddings.T)
        batch_size = similarity_matrix.size(0)

        metrics = {}
        for k in self.recall_k_values:
            top_k_indices = torch.topk(similarity_matrix, k=min(k, batch_size), dim=1).indices

            correct_indices = torch.arange(batch_size, device=similarity_matrix.device).unsqueeze(1)
            hits = (top_k_indices == correct_indices).any(dim=1).float()

            recall = hits.mean().item()
            metrics[f"recall_at_{k}"] = recall

        return metrics

    def compute_mrr(self, text_embeddings: torch.Tensor, code_embeddings: torch.Tensor) -> float:
        """Compute Mean Reciprocal Rank.

        Args:
            text_embeddings: Text embeddings
            code_embeddings: Code embeddings

        Returns:
            MRR value
        """
        similarity_matrix = torch.matmul(text_embeddings, code_embeddings.T)
        batch_size = similarity_matrix.size(0)

        sorted_indices = torch.argsort(similarity_matrix, dim=1, descending=True)
        correct_indices = torch.arange(batch_size, device=similarity_matrix.device)

        ranks = []
        for i in range(batch_size):
            rank = (sorted_indices[i] == correct_indices[i]).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(1.0 / rank)

        return sum(ranks) / len(ranks)

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step.

        Args:
            batch: Batch dictionary
            batch_idx: Batch index

        Returns:
            Loss value
        """
        outputs = self.forward(batch)
        loss = self.compute_contrastive_loss(outputs["text_embeddings"], outputs["code_embeddings"])

        batch_size = batch["text_input_ids"].size(0)
        self.log(
            "train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size
        )
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step.

        Args:
            batch: Batch dictionary
            batch_idx: Batch index

        Returns:
            Loss value
        """
        outputs = self.forward(batch)
        loss = self.compute_contrastive_loss(outputs["text_embeddings"], outputs["code_embeddings"])

        recall_metrics = self.compute_recall_at_k(
            outputs["text_embeddings"], outputs["code_embeddings"]
        )
        mrr = self.compute_mrr(outputs["text_embeddings"], outputs["code_embeddings"])

        batch_size = batch["text_input_ids"].size(0)
        self.log(
            "val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size
        )
        for metric_name, metric_value in recall_metrics.items():
            self.log(
                f"val_{metric_name}",
                metric_value,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
        self.log("val_mrr", mrr, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size)

        return loss

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step.

        Args:
            batch: Batch dictionary
            batch_idx: Batch index

        Returns:
            Loss value
        """
        outputs = self.forward(batch)
        loss = self.compute_contrastive_loss(outputs["text_embeddings"], outputs["code_embeddings"])

        recall_metrics = self.compute_recall_at_k(
            outputs["text_embeddings"], outputs["code_embeddings"]
        )
        mrr = self.compute_mrr(outputs["text_embeddings"], outputs["code_embeddings"])

        batch_size = batch["text_input_ids"].size(0)
        self.log("test_loss", loss, on_step=False, on_epoch=True, batch_size=batch_size)
        for metric_name, metric_value in recall_metrics.items():
            self.log(
                f"test_{metric_name}",
                metric_value,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
        self.log("test_mrr", mrr, on_step=False, on_epoch=True, batch_size=batch_size)

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer.

        Returns:
            AdamW optimizer
        """
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
