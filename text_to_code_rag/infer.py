from pathlib import Path

import torch
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from text_to_code_rag.data_manager import ensure_data_available
from text_to_code_rag.model import BiEncoderModel


def load_model(checkpoint_path: Path, cfg: DictConfig) -> BiEncoderModel:
    """Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        cfg: Hydra configuration

    Returns:
        Loaded model
    """
    model = BiEncoderModel.load_from_checkpoint(
        checkpoint_path,
        text_model_name=cfg.model.text_model_name,
        code_model_name=cfg.model.code_model_name,
        embedding_dim=cfg.model.embedding_dim,
        learning_rate=cfg.model.learning_rate,
        temperature=cfg.model.temperature,
        recall_k_values=cfg.model.recall_k_values,
        weights_only=False,
    )
    model.eval()
    return model


def encode_query(
    query: str, model: BiEncoderModel, tokenizer: AutoTokenizer, max_length: int = 128
) -> torch.Tensor:
    """Encode text query into embedding.

    Args:
        query: Text query
        model: Trained bi-encoder model
        tokenizer: Text tokenizer
        max_length: Maximum token length

    Returns:
        Query embedding
    """
    encoded = tokenizer(
        query,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    device = next(model.parameters()).device
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        embedding = model.encode_text(input_ids, attention_mask)

    return embedding


def search_similar_code(
    query_embedding: torch.Tensor,
    code_embeddings: torch.Tensor,
    code_snippets: list[str],
    top_k: int = 5,
) -> list[dict]:
    """Search for similar code snippets.

    Args:
        query_embedding: Query embedding [1, embedding_dim]
        code_embeddings: Code embeddings [num_codes, embedding_dim]
        code_snippets: List of code snippets
        top_k: Number of results to return

    Returns:
        List of dictionaries with code and similarity scores
    """
    similarities = torch.matmul(query_embedding, code_embeddings.T).squeeze(0)

    top_k_values, top_k_indices = torch.topk(similarities, k=min(top_k, len(code_snippets)))

    results = []
    for idx, similarity in zip(top_k_indices.tolist(), top_k_values.tolist(), strict=False):
        results.append({"code": code_snippets[idx], "similarity": similarity})

    return results


def infer(
    query: str,
    checkpoint_path: Path | str,
    config_name: str = "train",
    config_path: str | None = None,
    top_k: int = 5,
):
    """Run inference on a text query.

    Args:
        query: Text query describing desired code
        checkpoint_path: Path to model checkpoint
        config_name: Name of config file
        config_path: Path to config directory
        top_k: Number of results to return
    """
    checkpoint_path = Path(checkpoint_path)

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

    print(f"Loading model from {checkpoint_path}...")
    model = load_model(checkpoint_path, cfg)

    text_tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_model_name)

    print(f"Encoding query: {query}")
    query_embedding = encode_query(query, model, text_tokenizer, cfg.data.max_text_length)

    dummy_codes = [
        "def sort_list(lst):\n    return sorted(lst)",
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr",
        "def parse_sql(query):\n    return sqlparse.parse(query)",
    ]

    code_tokenizer = AutoTokenizer.from_pretrained(cfg.model.code_model_name)

    device = next(model.parameters()).device

    code_embeddings = []
    for code in dummy_codes:
        encoded = code_tokenizer(
            code,
            max_length=cfg.data.max_code_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.no_grad():
            embedding = model.encode_code(input_ids, attention_mask)
        code_embeddings.append(embedding)

    code_embeddings = torch.cat(code_embeddings, dim=0)

    results = search_similar_code(query_embedding, code_embeddings, dummy_codes, top_k)

    print(f"\nTop {top_k} results for query: '{query}'")
    print("=" * 80)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Similarity: {result['similarity']:.4f}")
        print(f"Code:\n{result['code']}")
        print("-" * 80)


if __name__ == "__main__":
    infer(
        query="List sorting function",
        checkpoint_path="checkpoints/bi-encoder-epoch=02-val_loss=1.32.ckpt",
        top_k=3,
    )
