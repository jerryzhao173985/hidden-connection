#!/usr/bin/env python3
"""
Hidden Connections - ML Pipeline
Processes participant responses into semantic embeddings for visualization.
Outputs point data with nearest neighbor links for constellation visualization.
"""

import json
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from umap import UMAP

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Question field mappings for display labels
QUESTION_LABELS = {
    "q1_safe_place": "Q1 (safe place)",
    "q2_stress": "Q2 (stress)",
    "q3_understood": "Q3 (understood)",
    "q4_free_day": "Q4 (free day)",
    "q5_one_word": "Q5 (one word)",
}

TEXT_FIELDS = list(QUESTION_LABELS.keys())

REQUIRED_COLUMNS = [
    "id",
    *TEXT_FIELDS,
]


class PipelineError(Exception):
    """Base exception for pipeline failures."""
    pass


class ConfigError(PipelineError):
    """Configuration validation error."""
    pass


class DataValidationError(PipelineError):
    """Data validation error."""
    pass


def validate_path(base: Path, user_path: str, must_exist: bool = True) -> Path:
    """Validate path to prevent directory traversal attacks."""
    try:
        full_path = (base / user_path).resolve()
        base_resolved = base.resolve()
        if not (
            full_path.is_relative_to(base_resolved)
            or full_path.is_relative_to(base_resolved.parent)
        ):
            raise ConfigError(f"Path '{user_path}' is outside allowed directory")
        if must_exist and not full_path.exists():
            raise ConfigError(f"Path does not exist: {full_path}")
        return full_path
    except ValueError as e:
        raise ConfigError(f"Invalid path '{user_path}': {e}")


def load_config(config_path: Path) -> dict:
    """Load and validate configuration from YAML file."""
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in config: {e}")

    if not isinstance(config, dict):
        raise ConfigError("Config must be a dictionary")

    required_keys = ["embedding_model", "umap", "clustering", "text_processing", "paths"]
    for key in required_keys:
        if key not in config:
            raise ConfigError(f"Missing required config key: {key}")

    max_chars = config.get("text_processing", {}).get("max_chars_per_question")
    if not isinstance(max_chars, int) or max_chars < 1:
        raise ConfigError(f"max_chars_per_question must be positive integer, got {max_chars}")

    n_neighbors = config.get("umap", {}).get("n_neighbors")
    if not isinstance(n_neighbors, int) or n_neighbors < 2:
        raise ConfigError(f"umap.n_neighbors must be >= 2, got {n_neighbors}")

    n_clusters = config.get("clustering", {}).get("n_clusters")
    if not isinstance(n_clusters, int) or n_clusters < 2:
        raise ConfigError(f"clustering.n_clusters must be >= 2, got {n_clusters}")

    return config


def validate_dataframe(df: pd.DataFrame) -> None:
    """Validate that required columns exist."""
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")


def validate_data_quality(df: pd.DataFrame) -> None:
    """Validate data content quality."""
    errors = []

    if len(df) == 0:
        raise DataValidationError("Input CSV is empty")

    if df["id"].duplicated().any():
        dup_count = df["id"].duplicated().sum()
        errors.append(f"Found {dup_count} duplicate participant IDs")

    for field in TEXT_FIELDS:
        empty_count = df[field].isna().sum() + (df[field] == "").sum()
        if empty_count > len(df) * 0.5:
            errors.append(f"Field {field} has {empty_count}/{len(df)} empty values")

    if errors:
        raise DataValidationError("\n".join(errors))


def concatenate_responses(row: pd.Series, max_chars: int) -> str:
    """Concatenate text fields into a single string with labels."""
    parts = []
    for field, label in QUESTION_LABELS.items():
        text = str(row.get(field, "")).strip()
        if text and text.lower() != "nan":
            if len(text) > max_chars:
                text = text[:max_chars] + "..."
            parts.append(f"{label}: {text}")
    return "\n".join(parts)


def normalize_coordinates(coords: np.ndarray) -> np.ndarray:
    """Normalize 2D coordinates to [-1, 1] range."""
    min_vals = coords.min(axis=0)
    max_vals = coords.max(axis=0)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1
    normalized = (coords - min_vals) / ranges
    return normalized * 2 - 1


def compute_neighbor_links(embeddings: np.ndarray, clusters: np.ndarray, k: int = 3) -> list[tuple[int, int, float]]:
    """
    Compute k-nearest neighbor links within each cluster.
    Returns list of (source_idx, target_idx, similarity) tuples.
    """
    links = []
    unique_clusters = np.unique(clusters)

    for cluster_id in unique_clusters:
        # Get indices of points in this cluster
        cluster_mask = clusters == cluster_id
        cluster_indices = np.where(cluster_mask)[0]

        if len(cluster_indices) < 2:
            continue

        # Get embeddings for this cluster
        cluster_embeddings = embeddings[cluster_mask]

        # Compute k-nearest neighbors within cluster
        n_neighbors = min(k + 1, len(cluster_indices))  # +1 because point is its own neighbor
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
        nn.fit(cluster_embeddings)
        distances, indices = nn.kneighbors(cluster_embeddings)

        # Convert to global indices and create links
        for local_idx, (dists, neighbors) in enumerate(zip(distances, indices)):
            global_source = cluster_indices[local_idx]
            for dist, local_neighbor in zip(dists[1:], neighbors[1:]):  # Skip self (first neighbor)
                global_target = cluster_indices[local_neighbor]
                # Only add link once (avoid duplicates)
                if global_source < global_target:
                    similarity = 1.0 - dist  # Convert distance to similarity
                    links.append((int(global_source), int(global_target), float(similarity)))

    return links


def process_data(config: dict, base_dir: Path) -> dict:
    """Main processing pipeline. Returns dict with points and links."""
    input_path = validate_path(base_dir, config["paths"]["input"])

    logger.info(f"Loading data from {input_path}")
    try:
        df = pd.read_csv(input_path)
    except pd.errors.EmptyDataError:
        raise DataValidationError(f"CSV file is empty: {input_path}")
    except pd.errors.ParserError as e:
        raise DataValidationError(f"Failed to parse CSV: {e}")

    validate_dataframe(df)
    validate_data_quality(df)
    logger.info(f"Loaded {len(df)} participants")

    n_neighbors = config["umap"]["n_neighbors"]
    if len(df) < n_neighbors:
        raise DataValidationError(
            f"Dataset has {len(df)} samples but UMAP requires >= {n_neighbors} samples"
        )

    max_chars = config["text_processing"]["max_chars_per_question"]

    logger.info("Concatenating text responses...")
    texts = df.apply(lambda row: concatenate_responses(row, max_chars), axis=1).tolist()

    model_name = config["embedding_model"]
    logger.info(f"Loading embedding model: {model_name}")
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        raise PipelineError(f"Failed to load embedding model '{model_name}': {e}")

    logger.info("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)
    logger.info(f"Embedding shape: {embeddings.shape}")

    logger.info("Applying UMAP...")
    umap_config = config["umap"]
    reducer = UMAP(
        n_neighbors=umap_config["n_neighbors"],
        min_dist=umap_config["min_dist"],
        metric=umap_config["metric"],
        random_state=umap_config.get("random_state", 42),
        n_components=2,
    )
    coords_2d = reducer.fit_transform(embeddings)
    coords_2d = normalize_coordinates(coords_2d)

    logger.info("Clustering...")
    cluster_config = config["clustering"]
    kmeans = KMeans(
        n_clusters=cluster_config["n_clusters"],
        random_state=cluster_config.get("random_state", 42),
        n_init=10,
    )
    clusters = kmeans.fit_predict(embeddings)

    # Compute nearest neighbor links within clusters
    logger.info("Computing neighbor links...")
    k_links = config.get("visualization", {}).get("k_neighbors", 3)
    links = compute_neighbor_links(embeddings, clusters, k=k_links)
    logger.info(f"Created {len(links)} constellation links")

    # Build output records
    logger.info("Building output records...")
    points = []
    for i, row in enumerate(df.itertuples()):
        # getattr with default handles missing attribute; explicit check for empty/nan
        nickname = getattr(row, "nickname", "")
        if pd.isna(nickname) or not str(nickname).strip():
            nickname = "anonymous"

        point = {
            "id": str(row.id),
            "text": texts[i],
            "x": float(coords_2d[i, 0]),
            "y": float(coords_2d[i, 1]),
            "cluster": int(clusters[i]),
            "nickname": str(nickname),
        }
        points.append(point)

    # Format links as list of [source, target, strength]
    formatted_links = [[s, t, round(w, 3)] for s, t, w in links]

    return {
        "points": points,
        "links": formatted_links,
        "clusters": cluster_config["n_clusters"],
    }


def main():
    parser = argparse.ArgumentParser(description="Process Hidden Connections data")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        base_dir = Path(__file__).parent
        data = process_data(config, base_dir)

        output_path = validate_path(base_dir, config["paths"]["output"], must_exist=False)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Writing {len(data['points'])} points and {len(data['links'])} links to {output_path}")
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        # Create symlink in web directory
        web_dir = base_dir.parent / "web"
        web_dir.mkdir(parents=True, exist_ok=True)
        web_output = web_dir / "points.json"

        if web_output.exists() or web_output.is_symlink():
            web_output.unlink()
        web_output.symlink_to(output_path.resolve())
        logger.info(f"Created symlink: {web_output} -> {output_path}")

        logger.info("Done!")

    except PipelineError as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
