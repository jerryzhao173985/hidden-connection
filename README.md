# Hidden Connections

An interactive web-based digital artwork that visualizes the latent semantic relationships between anonymous participants based on their responses to psychological-style questions.

## Overview

Hidden Connections embeds free-text responses into a high-dimensional latent space using sentence transformers, projects to 2D via UMAP, and renders an interactive star map. Two viewing modes expose different interpretive lenses:
- **Model's view**: Coloring by unsupervised clustering
- **Self view**: Coloring by participants' self-reported social energy

## Project Structure

```
hidden-connections/
├── data/
│   └── responses_projective.csv      # Input data
├── processed/
│   └── points.json                   # ML pipeline output
├── pipeline/
│   ├── process.py                    # Main ML script
│   ├── config.yaml                   # Pipeline configuration
│   └── requirements.txt              # Python dependencies
├── web/
│   ├── index.html                    # Entry point
│   ├── style.css                     # Styles
│   ├── main.js                       # Visualization logic
│   └── points.json                   # Data for frontend
└── README.md
```

## Setup

### Prerequisites

- Python 3.10+
- pip or uv for package management

### Install Dependencies

```bash
cd pipeline
pip install -r requirements.txt
```

Or with uv:

```bash
cd pipeline
uv pip install -r requirements.txt
```

### Run the ML Pipeline

```bash
cd pipeline
python process.py
```

This will:
1. Load the CSV data
2. Generate sentence embeddings
3. Apply UMAP for 2D projection
4. Cluster participants using KMeans
5. Output `processed/points.json` and `web/points.json`

### Serve the Frontend

From the `web` directory:

```bash
cd web
python -m http.server 8000
```

Then open http://localhost:8000 in your browser.

## Configuration

Edit `pipeline/config.yaml` to adjust:

- **embedding_model**: Sentence transformer model (default: `all-MiniLM-L6-v2`)
- **umap**: UMAP parameters (n_neighbors, min_dist, metric)
- **clustering**: Number of clusters
- **paths**: Input/output file locations

## Data Format

### Input CSV

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique participant identifier |
| `q1_safe_place` | string | Response to "Describe a place where you feel very safe" |
| `q2_stress` | string | Response to stress coping question |
| `q3_understood` | string | Response to being understood question |
| `q4_free_day` | string | Response to free day question |
| `q5_one_word` | string | One-word self description |
| `q6_decision_style` | enum | `Mostly rational` / `Mostly emotional` / `Depends` |
| `q7_social_energy` | enum | `Energised` / `Drained` / `Depends` |
| `q8_region` | string | Geographic region |
| `nickname` | string | Optional display name |

### Output JSON

Each point contains:
- `id`: Participant identifier
- `text`: Concatenated responses with labels
- `x`, `y`: 2D coordinates (normalized to [-1, 1])
- `cluster`: Cluster assignment (0 to k-1)
- `decision_style`, `social_energy`, `region`: Self-reported attributes
- `nickname`: Display name or "anonymous"

## Interaction

- **Hover** over points to see participant responses and metadata
- Click **Model's view** to color by algorithmic clustering
- Click **Self view** to color by self-reported social energy

## Color Palettes

### Model's View (Clusters)
- Cluster 0: Teal (#4ECDC4)
- Cluster 1: Coral (#FF6B6B)
- Cluster 2: Lavender (#C9B1FF)
- Cluster 3: Warm Yellow (#FFE66D)
- Cluster 4: Mint (#95E1D3)

### Self View (Social Energy)
- Energised: Bright Gold (#FFD93D)
- Drained: Deep Violet (#6C5CE7)
- Depends: Soft Green (#A8E6CF)
