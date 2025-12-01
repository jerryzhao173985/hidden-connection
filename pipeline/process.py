#!/usr/bin/env python3
"""
Hidden Connections - ML Pipeline
Processes participant responses into semantic embeddings for visualization.

Supports two embedding backends:
- OpenAI API (text-embedding-3-large with Matryoshka dimensions)
- BGE local fallback (BAAI/bge-large-en-v1.5)

Outputs point data with nearest neighbor links for constellation visualization.
Supports multiple views: combined (all questions) and individual question views.
"""

import json
import argparse
import logging
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
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


# =============================================================================
# EMBEDDING BACKENDS
# =============================================================================

class EmbeddingBackend(ABC):
    """Abstract base class for embedding backends."""

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts to embeddings. Returns (n_texts, dim) array."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        pass


class OpenAIBackend(EmbeddingBackend):
    """OpenAI API embedding backend using text-embedding-3-large."""

    def __init__(self, model: str = "text-embedding-3-large", dimensions: int = 256):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self._dimensions = dimensions
        logger.info(f"Initialized OpenAI backend: {model} (dim={dimensions})")

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts using OpenAI embeddings API."""
        # OpenAI API accepts list of strings
        # Process in batches to avoid rate limits
        batch_size = 100
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
                dimensions=self._dimensions,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings, dtype=np.float32)

    @property
    def dimension(self) -> int:
        return self._dimensions


class BGEBackend(EmbeddingBackend):
    """BAAI/BGE local embedding backend (fallback)."""

    def __init__(
        self,
        model: str = "BAAI/bge-large-en-v1.5",
        instruction: str = None,
        device: str = "auto",
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")

        self.device = self._resolve_device(device)
        self.model = SentenceTransformer(model, device=self.device)
        self.instruction = instruction
        self._dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Initialized BGE backend: {model} (dim={self._dimension}, device={self.device})")

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts using local BGE model."""
        # BGE models work best with instruction prefix
        if self.instruction:
            texts = [f"{self.instruction} {t}" for t in texts]

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return np.array(embeddings, dtype=np.float32)

    @property
    def dimension(self) -> int:
        return self._dimension


def create_embedding_backend(config: dict) -> EmbeddingBackend:
    """Factory function to create the appropriate embedding backend."""
    embedding_config = config.get("embedding", {})
    backend_type = embedding_config.get("backend", "openai")

    if backend_type == "openai":
        openai_config = embedding_config.get("openai", {})
        return OpenAIBackend(
            model=openai_config.get("model", "text-embedding-3-large"),
            dimensions=openai_config.get("dimensions", 256),
        )
    elif backend_type == "bge":
        bge_config = embedding_config.get("bge", {})
        return BGEBackend(
            model=bge_config.get("model", "BAAI/bge-large-en-v1.5"),
            instruction=bge_config.get("instruction"),
            device=bge_config.get("device", "auto"),
        )
    else:
        raise ValueError(f"Unknown embedding backend: {backend_type}")


# =============================================================================
# CONFIGURATION
# =============================================================================

# Question field mappings for display labels
QUESTION_LABELS = {
    "q1_safe_place": "Q1 (safe place)",
    "q2_stress": "Q2 (stress)",
    "q3_understood": "Q3 (understood)",
    "q4_free_day": "Q4 (free day)",
    "q5_one_word": "Q5 (one word)",
}

# View definitions for multi-view visualization
VIEW_CONFIG = {
    "combined": {
        "label": "All Responses",
        "shortLabel": "All",
        "fields": list(QUESTION_LABELS.keys()),
        "description": "Semantic similarity across all responses",
    },
    "q1_safe_place": {
        "label": "Safe Place",
        "shortLabel": "Safe",
        "fields": ["q1_safe_place"],
        "description": "Where do you feel safe?",
    },
    "q2_stress": {
        "label": "Stress Response",
        "shortLabel": "Stress",
        "fields": ["q2_stress"],
        "description": "How do you cope with stress?",
    },
    "q3_understood": {
        "label": "Feeling Understood",
        "shortLabel": "Understood",
        "fields": ["q3_understood"],
        "description": "When did you feel seen?",
    },
    "q4_free_day": {
        "label": "Free Day",
        "shortLabel": "Free",
        "fields": ["q4_free_day"],
        "description": "How would you spend a free day?",
    },
    "q5_one_word": {
        "label": "One Word",
        "shortLabel": "Word",
        "fields": ["q5_one_word"],
        "description": "One word to describe yourself",
    },
}

TEXT_FIELDS = list(QUESTION_LABELS.keys())

REQUIRED_COLUMNS = [
    "id",
    *TEXT_FIELDS,
]


# =============================================================================
# EXCEPTIONS
# =============================================================================

class PipelineError(Exception):
    """Base exception for pipeline failures."""
    pass


class ConfigError(PipelineError):
    """Configuration validation error."""
    pass


class DataValidationError(PipelineError):
    """Data validation error."""
    pass


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

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

    # Required keys for new config structure
    required_keys = ["embedding", "umap", "clustering", "text_processing", "paths"]
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


def concatenate_responses(row: pd.Series, max_chars: int, fields: list[str] | None = None) -> str:
    """Concatenate text fields into a single string with labels."""
    if fields is None:
        fields = list(QUESTION_LABELS.keys())

    parts = []
    for field in fields:
        if field not in QUESTION_LABELS:
            continue
        label = QUESTION_LABELS[field]
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


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_view(
    df: pd.DataFrame,
    view_id: str,
    view_config: dict,
    embedding_backend: EmbeddingBackend,
    umap_config: dict,
    cluster_config: dict,
    max_chars: int,
    k_links: int,
) -> dict:
    """Process a single view (combined or individual question)."""
    fields = view_config["fields"]
    logger.info(f"Processing view '{view_id}' with fields: {fields}")

    # Generate text for this view
    all_texts = df.apply(
        lambda row: concatenate_responses(row, max_chars, fields), axis=1
    ).tolist()

    # Identify valid (non-empty) entries
    valid_mask = [bool(t.strip()) for t in all_texts]
    valid_indices = [i for i, valid in enumerate(valid_mask) if valid]
    excluded_indices = [i for i, valid in enumerate(valid_mask) if not valid]

    if excluded_indices:
        logger.warning(f"  Excluding {len(excluded_indices)} participants with missing responses: indices {excluded_indices}")

    # Filter to only valid texts for processing
    texts = [all_texts[i] for i in valid_indices]

    if len(texts) == 0:
        raise DataValidationError(f"No valid responses for view '{view_id}'")

    # Adjust UMAP n_neighbors if we have fewer valid samples
    effective_n_neighbors = min(umap_config["n_neighbors"], len(texts) - 1)
    if effective_n_neighbors < umap_config["n_neighbors"]:
        logger.warning(f"  Reduced n_neighbors from {umap_config['n_neighbors']} to {effective_n_neighbors} due to sample size")

    # Generate embeddings using the configured backend
    logger.info(f"  Generating embeddings for {len(texts)} texts...")
    embeddings = embedding_backend.encode(texts)
    logger.info(f"  Embeddings shape: {embeddings.shape}")

    # Apply UMAP
    logger.info(f"  Applying UMAP...")
    reducer = UMAP(
        n_neighbors=effective_n_neighbors,
        min_dist=umap_config["min_dist"],
        metric=umap_config["metric"],
        random_state=umap_config.get("random_state", 42),
        n_components=2,
    )
    coords_2d = reducer.fit_transform(embeddings)
    coords_2d = normalize_coordinates(coords_2d)

    # Clustering
    logger.info(f"  Clustering...")
    kmeans = KMeans(
        n_clusters=cluster_config["n_clusters"],
        random_state=cluster_config.get("random_state", 42),
        n_init=10,
    )
    clusters = kmeans.fit_predict(embeddings)

    # Compute nearest neighbor links
    links = compute_neighbor_links(embeddings, clusters, k=k_links)
    formatted_links = [[s, t, round(w, 3)] for s, t, w in links]

    logger.info(f"  View '{view_id}': {len(formatted_links)} links")

    return {
        "texts": texts,
        "coords": coords_2d,
        "clusters": clusters,
        "links": formatted_links,
        "embeddings": embeddings,  # Include for frontend KNN/estimatePosition
        "valid_indices": valid_indices,  # Map from result index to original df index
    }


def process_data(config: dict, base_dir: Path) -> dict:
    """Main processing pipeline. Returns dict with multi-view points and links."""
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
    k_links = config.get("visualization", {}).get("k_neighbors", 3)
    umap_config = config["umap"]
    cluster_config = config["clustering"]

    # Create embedding backend based on config
    logger.info("Initializing embedding backend...")
    embedding_backend = create_embedding_backend(config)

    # Process each view
    logger.info(f"Processing {len(VIEW_CONFIG)} views...")
    view_results = {}
    for view_id, view_cfg in VIEW_CONFIG.items():
        view_results[view_id] = process_view(
            df, view_id, view_cfg, embedding_backend, umap_config, cluster_config, max_chars, k_links
        )

    # Build output structure
    logger.info("Building output records...")

    # Views metadata
    views_meta = {}
    for view_id, view_cfg in VIEW_CONFIG.items():
        views_meta[view_id] = {
            "label": view_cfg["label"],
            "shortLabel": view_cfg["shortLabel"],
            "description": view_cfg["description"],
        }

    # Build points with per-view data
    points = []
    for i, row in enumerate(df.itertuples()):
        nickname = getattr(row, "nickname", "")
        if pd.isna(nickname) or not str(nickname).strip():
            nickname = "anonymous"

        # Collect raw responses
        responses = {}
        for field in TEXT_FIELDS:
            val = getattr(row, field, "")
            if pd.isna(val):
                val = ""
            responses[field] = str(val).strip()

        # Build per-view data (handle excluded points per view)
        views_data = {}
        for view_id, result in view_results.items():
            valid_indices = result["valid_indices"]
            if i in valid_indices:
                # Find position in the filtered results array
                result_idx = valid_indices.index(i)
                views_data[view_id] = {
                    "x": float(result["coords"][result_idx, 0]),
                    "y": float(result["coords"][result_idx, 1]),
                    "cluster": int(result["clusters"][result_idx]),
                    "text": result["texts"][result_idx],
                }
            else:
                # Point excluded from this view due to missing response
                views_data[view_id] = None

        # Include embedding from combined view for frontend KNN
        combined_valid = view_results["combined"]["valid_indices"]
        if i in combined_valid:
            result_idx = combined_valid.index(i)
            combined_embedding = view_results["combined"]["embeddings"][result_idx].tolist()
        else:
            combined_embedding = None  # No embedding if excluded from combined view

        point = {
            "id": str(row.id),
            "nickname": str(nickname),
            "responses": responses,
            "views": views_data,
            "embedding": combined_embedding,  # 256-dim vector for frontend similarity search
        }
        points.append(point)

    # Collect links per view (convert filtered indices back to original df indices)
    links = {}
    for view_id, result in view_results.items():
        valid_indices = result["valid_indices"]
        # Links are [source_idx, target_idx, weight] in filtered space
        # Convert to original df indices
        converted_links = []
        for s, t, w in result["links"]:
            orig_s = valid_indices[s]
            orig_t = valid_indices[t]
            converted_links.append([orig_s, orig_t, w])
        links[view_id] = converted_links

    return {
        "views": views_meta,
        "points": points,
        "links": links,
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
    parser.add_argument(
        "--backend",
        choices=["openai", "bge"],
        help="Override embedding backend (default: from config)",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)

        # Override backend if specified
        if args.backend:
            config["embedding"]["backend"] = args.backend
            logger.info(f"Using backend override: {args.backend}")

        base_dir = Path(__file__).parent
        data = process_data(config, base_dir)

        output_path = validate_path(base_dir, config["paths"]["output"], must_exist=False)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Writing {len(data['points'])} points to {output_path}")
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        # Copy to web directory (not symlink for deployment compatibility)
        web_dir = base_dir.parent / "web"
        web_dir.mkdir(parents=True, exist_ok=True)
        web_output = web_dir / "points.json"

        with open(web_output, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Copied to web directory: {web_output}")

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
