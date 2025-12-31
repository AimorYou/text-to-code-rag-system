import os
import subprocess

import pytorch_lightning as pl
from hydra import compose, initialize
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from text_to_code_rag.data_loader import CodeSearchDataset
from text_to_code_rag.data_manager import ensure_data_available
from text_to_code_rag.model import BiEncoderModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_git_commit_id() -> str:
    """Get current git commit ID.

    Returns:
        Git commit hash or empty string if not available
    """
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
    except Exception:
        return ""


def train(config_name: str = "train", config_path: str | None = None):
    """Train the bi-encoder model.

    Args:
        config_name: Name of the config file (without .yaml)
        config_path: Path to config directory (default: ../configs)
    """
    if config_path is None:
        config_path = "../configs"

    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=config_name)

    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    print("Setting up data with DVC...")
    data_path = ensure_data_available(
        dataset_name=cfg.data.dataset_name,
        data_dir=cfg.data.data_dir,
        use_dvc=cfg.data.use_dvc,
    )
    print(f"Data available at: {data_path}")

    pl.seed_everything(cfg.training.seed)

    text_tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_model_name)
    code_tokenizer = AutoTokenizer.from_pretrained(cfg.model.code_model_name)

    train_dataset = CodeSearchDataset(
        split="train",
        text_tokenizer=text_tokenizer,
        code_tokenizer=code_tokenizer,
        max_text_length=cfg.data.max_text_length,
        max_code_length=cfg.data.max_code_length,
        subset_size=cfg.data.train_subset_size,
        data_path=data_path,
    )

    val_dataset = CodeSearchDataset(
        split="train",
        text_tokenizer=text_tokenizer,
        code_tokenizer=code_tokenizer,
        max_text_length=cfg.data.max_text_length,
        max_code_length=cfg.data.max_code_length,
        subset_size=cfg.data.val_subset_size,
        data_path=data_path,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=False,
    )

    model = BiEncoderModel(
        text_model_name=cfg.model.text_model_name,
        code_model_name=cfg.model.code_model_name,
        embedding_dim=cfg.model.embedding_dim,
        learning_rate=cfg.model.learning_rate,
        temperature=cfg.model.temperature,
        recall_k_values=cfg.model.recall_k_values,
        freeze_encoders=cfg.model.freeze_encoders,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.training.checkpoint_dir,
        filename="bi-encoder-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=cfg.training.early_stopping_patience, mode="min"
    )

    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.logging.experiment_name, tracking_uri=cfg.logging.tracking_uri
    )

    params = OmegaConf.to_container(cfg, resolve=True)
    mlflow_logger.log_hyperparams(params)

    git_commit = get_git_commit_id()
    if git_commit:
        mlflow_logger.experiment.set_tag(mlflow_logger.run_id, "git_commit_id", git_commit)

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        logger=mlflow_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=cfg.training.log_every_n_steps,
        deterministic=False,
    )

    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)

    print("Training completed!")
    print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    train()
