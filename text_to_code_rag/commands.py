import fire

from text_to_code_rag.infer import infer as run_infer
from text_to_code_rag.train import train as run_train


class Commands:
    """CLI commands for text-to-code RAG system."""

    @staticmethod
    def train(config_name: str = "train", config_path: str | None = None):
        """Train the bi-encoder model.

        Args:
            config_name: Name of the config file (without .yaml)
            config_path: Path to config directory (default: ../configs)

        Examples:
            # Train with default config
            text-to-code-rag train

            # Train all params
            text-to-code-rag train --config_name=train model=base_model

            # Train with custom config path
            text-to-code-rag train --config_path=/path/to/configs
        """
        run_train(config_name=config_name, config_path=config_path)

    @staticmethod
    def infer(
        query: str = "function to sort a list",
        checkpoint_path: str = "checkpoints/bi-encoder-epoch=02-val_loss=1.32.ckpt",
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

        Examples:
            # Run inference with default query
            text-to-code-rag infer

            # Custom query
            text-to-code-rag infer --query="function to parse JSON"

            # Custom checkpoint
            text-to-code-rag infer --query="sort algorithm" --checkpoint_path=checkpoints/model.ckpt
        """
        run_infer(
            query=query,
            checkpoint_path=checkpoint_path,
            config_name=config_name,
            config_path=config_path,
            top_k=top_k,
        )


def main():
    """Entry point for the CLI."""
    fire.Fire(Commands)


if __name__ == "__main__":
    main()
