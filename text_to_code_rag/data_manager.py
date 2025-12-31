import subprocess
from pathlib import Path

from datasets import load_dataset


def download_data(
    dataset_name: str = "espejelomar/code_search_net_python_10000_examples",
    data_dir: Path | str = "data/raw",
    cache_dir: Path | str | None = None,
) -> Path:
    """Download dataset from HuggingFace and save locally.

    Args:
        dataset_name: HuggingFace dataset name
        data_dir: Directory to save the dataset
        cache_dir: Optional cache directory for HuggingFace datasets

    Returns:
        Path to downloaded data directory
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset: {dataset_name}")

    dataset = load_dataset(dataset_name, cache_dir=cache_dir)

    output_path = data_dir / "code_search_net"
    print(f"Saving dataset to: {output_path}")

    for split_name, split_data in dataset.items():
        split_path = output_path / split_name
        split_path.mkdir(parents=True, exist_ok=True)
        split_data.to_parquet(str(split_path / "data.parquet"))

    print(f"Dataset downloaded and saved to {output_path}")
    return output_path


def init_dvc(repo_dir: Path | str = ".") -> bool:
    """Initialize DVC in the repository if not already initialized.

    Args:
        repo_dir: Repository directory

    Returns:
        True if DVC was initialized or already exists
    """
    repo_dir = Path(repo_dir)
    dvc_dir = repo_dir / ".dvc"

    if dvc_dir.exists():
        print("DVC already initialized")
        return True

    try:
        subprocess.run(
            ["dvc", "init"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        print("DVC initialized successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to initialize DVC: {e.stderr}")
        return False


def add_to_dvc(file_path: Path | str, repo_dir: Path | str = ".") -> bool:
    """Add file or directory to DVC tracking.

    Args:
        file_path: Path to file or directory to track
        repo_dir: Repository directory

    Returns:
        True if successfully added to DVC
    """
    file_path = Path(file_path)
    repo_dir = Path(repo_dir)

    if not file_path.exists():
        print(f"Path does not exist: {file_path}")
        return False

    dvc_file = Path(str(file_path) + ".dvc")
    if dvc_file.exists():
        print(f"Already tracked by DVC: {file_path}")
        return True

    try:
        subprocess.run(
            ["dvc", "add", str(file_path)],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"Added to DVC: {file_path}")

        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to add to DVC: {e.stderr}")
        return False


def setup_local_remote(
    remote_name: str = "local_storage",
    remote_path: Path | str = "dvc_storage",
    repo_dir: Path | str = ".",
) -> bool:
    """Setup local DVC remote storage.

    Args:
        remote_name: Name for the remote
        remote_path: Path to local storage directory
        repo_dir: Repository directory

    Returns:
        True if remote was configured successfully
    """
    repo_dir = Path(repo_dir)
    remote_path = Path(remote_path).absolute()
    remote_path.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            ["dvc", "remote", "list"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        )

        if remote_name in result.stdout:
            print(f"DVC remote '{remote_name}' already configured")
            return True

        subprocess.run(
            ["dvc", "remote", "add", "-d", remote_name, str(remote_path)],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"DVC remote '{remote_name}' configured at: {remote_path}")

        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to configure DVC remote: {e.stderr}")
        return False


def push_to_dvc(repo_dir: Path | str = ".") -> bool:
    """Push data to DVC remote storage.

    Args:
        repo_dir: Repository directory

    Returns:
        True if push was successful
    """
    repo_dir = Path(repo_dir)

    try:
        subprocess.run(
            ["dvc", "push"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        print("Data pushed to DVC remote")

        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to push to DVC: {e.stderr}")
        return False


def pull_from_dvc(repo_dir: Path | str = ".") -> bool:
    """Pull data from DVC remote storage.

    Args:
        repo_dir: Repository directory

    Returns:
        True if pull was successful
    """
    repo_dir = Path(repo_dir)

    try:
        result = subprocess.run(
            ["dvc", "pull"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        if result.stdout:
            print(result.stdout)
        print("Data pulled from DVC remote")

        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to pull from DVC: {e.stderr}")
        return False


def ensure_data_available(
    dataset_name: str = "espejelomar/code_search_net_python_10000_examples",
    data_dir: Path | str = "data/raw",
    repo_dir: Path | str = ".",
    use_dvc: bool = True,
) -> Path:
    """Ensure data is available, using DVC if enabled.

    This function:
    1. Tries to pull data from DVC remote
    2. If that fails, downloads data from source
    3. Adds data to DVC tracking
    4. Pushes to DVC remote

    Args:
        dataset_name: HuggingFace dataset name
        data_dir: Directory for the dataset
        repo_dir: Repository directory
        use_dvc: Whether to use DVC for data management

    Returns:
        Path to the data directory
    """
    data_dir = Path(data_dir)
    repo_dir = Path(repo_dir)
    dataset_path = data_dir / "code_search_net"

    if use_dvc:
        init_dvc(repo_dir)

        setup_local_remote(repo_dir=repo_dir)

        if dataset_path.exists():
            print(f"Data directory already exists: {dataset_path}")
            return dataset_path

        if pull_from_dvc(repo_dir):
            if dataset_path.exists():
                print(f"Data successfully pulled from DVC: {dataset_path}")
                return dataset_path

    print("Downloading fresh data from source...")
    dataset_path = download_data(dataset_name, data_dir)

    if use_dvc:
        print("Adding data to DVC...")
        add_to_dvc(data_dir, repo_dir)

        print("Pushing data to DVC remote...")
        push_to_dvc(repo_dir)

    return dataset_path


def get_data_path(
    data_dir: Path | str = "data/raw",
    dataset_name: str = "code_search_net",
) -> Path:
    """Get the path to the dataset.

    Args:
        data_dir: Data directory
        dataset_name: Dataset name

    Returns:
        Path to the dataset
    """
    return Path(data_dir) / dataset_name
