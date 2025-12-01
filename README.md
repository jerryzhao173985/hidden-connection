# Hidden Connections

Interactive constellation visualization of semantic relationships between anonymous survey responses. Stars represent participants, connections reveal shared meaning.

**Live Demo**: https://jerryzhao173985.github.io/hidden-connection/

## Features

### Multi-View Exploration
- **Combined View**: All 5 questions merged into single embedding space
- **Per-Question Views**: See how people cluster differently based on:
  - Q1: Safe Place
  - Q2: Stress Response
  - Q3: Feeling Understood
  - Q4: Free Day
  - Q5: One Word

Switch views to discover: people similar in stress coping may be completely different in ideal free day.

### Visual Effects
- **3D Galaxy Motion**: Stars gently orbit, drift, and oscillate in depth
- **Background Parallax**: Distant stars move slower for depth perception
- **Breathing/Twinkling**: Organic pulsation and shimmer animations
- **Link Strength**: Stronger semantic connections appear thicker with soft glow
- **Cluster Brightness**: Selecting a star highlights entire cluster

### Interaction
- **Hover**: Preview star info, see neighbor connections
- **Click**: Lock selection, explore connections
- **Keyboard**:
  - `←` `→` Navigate between views
  - `1-6` Jump to specific view
  - `Esc` Clear selection

## Architecture

```
hidden-connections/
├── data/
│   └── responses_projective.csv    # Survey responses
├── pipeline/
│   ├── process.py                  # ML pipeline (embeddings, UMAP, clustering)
│   ├── config.yaml                 # Model & parameter config
│   └── requirements.txt
├── processed/
│   └── points.json                 # Pipeline output
├── web/
│   ├── index.html
│   ├── style.css
│   ├── main.js                     # Visualization (~1400 lines)
│   └── points.json                 # Data for frontend
├── server/
│   ├── server.js                   # Express API for live embeddings
│   ├── package.json
│   └── .env.example
└── .github/
    └── workflows/
        └── deploy.yml              # GitHub Pages + optional embedding regen
```

## ML Pipeline

### Embedding Backends

Two backends supported, configured in `pipeline/config.yaml`:

| Backend | Model | Dimensions | Use Case |
|---------|-------|------------|----------|
| **OpenAI** (default) | text-embedding-3-large | 256 (Matryoshka) | Best quality |
| **BGE** (fallback) | BAAI/bge-large-en-v1.5 | 1024 | Local, no API key |

OpenAI's Matryoshka representation: 256 dims still outperforms ada-002 at 1536 dims.

### Pipeline Flow

```
Survey Text → Embeddings → UMAP (2D) → KMeans Clustering → Neighbor Links → JSON
                 │
         OpenAI API or BGE local
```

Each view (combined + 5 questions) processed independently with separate:
- Embeddings
- UMAP projection
- Cluster assignments
- Nearest neighbor links

### Run Pipeline

```bash
cd pipeline
pip install -r requirements.txt

# With OpenAI (requires OPENAI_API_KEY env var)
export OPENAI_API_KEY=sk-...
python process.py

# With local BGE model
python process.py --backend bge
```

## Web Visualization

### Local Development

```bash
cd web
python -m http.server 9999
# Open http://localhost:9999
```

### Key Rendering Systems

| System | Description |
|--------|-------------|
| **Galaxy Motion** | Orbital drift, depth oscillation, global rotation, noise-based wandering |
| **Brightness Manager** | Smooth transitions, cluster-aware dimming hierarchy |
| **Link Renderer** | Strength-based width/opacity, soft glow on strong connections |
| **View Transitions** | Animated star movement when switching views |
| **Bounds Manager** | UMAP clamping + margin enforcement for clickability |

### CONFIG Object

All visual parameters tunable in `web/main.js`:

```javascript
CONFIG = {
  star: { baseRadius, hoverRadius, glowRadius, breathSpeed, ... },
  link: { baseAlpha, hoverAlpha, baseWidth, maxWidth, glowEnabled, ... },
  galaxy: { orbitSpeed, orbitRadius, depthRange, rotationSpeed, ... },
  ...
}
```

## API Server (Optional)

Express server for live embedding generation:

```bash
cd server
npm install
OPENAI_API_KEY=sk-... node server.js
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/embed` | Single text → 256-dim embedding |
| POST | `/api/embed-batch` | Up to 100 texts → embeddings |
| GET | `/api/health` | Health check |
| GET | `/api/config` | Current model config |

Enables future "add yourself" feature.

## Deployment

### GitHub Pages (Automatic)

Push to `main` → GitHub Actions deploys `web/` to Pages.

### Manual Embedding Regeneration

1. Add `OPENAI_API_KEY` to repo secrets
2. Go to Actions → "Deploy to GitHub Pages"
3. Click "Run workflow" with `regenerate_embeddings: true`

## Data Format

### Input CSV

| Field | Description |
|-------|-------------|
| `id` | Unique participant ID |
| `q1_safe_place` | "Describe a place where you feel safe" |
| `q2_stress` | Stress coping response |
| `q3_understood` | Feeling understood response |
| `q4_free_day` | Ideal free day response |
| `q5_one_word` | One-word self description |
| `q6_decision_style` | Rational / Emotional / Depends |
| `q7_social_energy` | Energised / Drained / Depends |
| `q8_region` | Geographic region |
| `nickname` | Display name (optional) |

### Output JSON

```json
{
  "views": {
    "combined": { "label": "All Questions", "description": "..." },
    "q1_safe_place": { "label": "Safe Place", ... }
  },
  "points": [{
    "id": "p_001",
    "nickname": "stargazer",
    "responses": { "q1_safe_place": "...", ... },
    "views": {
      "combined": { "x": 0.24, "y": -0.31, "cluster": 2, "text": "..." },
      "q1_safe_place": { "x": 0.67, "y": 0.12, "cluster": 0, "text": "..." }
    },
    "embedding": [0.023, -0.041, ...]  // 256-dim for frontend similarity
  }],
  "links": {
    "combined": [[0, 5, 0.847], [1, 3, 0.792]],  // [source, target, cosine_sim]
    "q1_safe_place": [...]
  }
}
```

## Development Notes

### Bug Fixes Implemented
- UMAP coordinates clamped to [-1, 1] (can exceed bounds)
- Final bounds check after galaxy motion (40px margin)
- Animation phases preserved across window resize
- Symlink replaced with actual file for GitHub Pages deployment
- Redundant `computeScreenPositions()` call eliminated during transitions

### Performance Optimizations
- Active point calculated once outside render loop
- Cluster centers computed only when needed
- Depth-based sorting for proper 3D layering

### Visual Consistency Fixes
- Single active star at a time (selected takes priority over hovered)
- Cluster brightness hierarchy: active (1.0) → neighbors (0.9) → same cluster (0.7) → others (0.25)

## Color Palette

```
Cluster 0: Teal      #4ECDC4
Cluster 1: Coral     #FF6B6B
Cluster 2: Lavender  #C9B1FF
Cluster 3: Gold      #FFE66D
Cluster 4: Mint      #95E1D3
```

## Tech Stack

- **Frontend**: Vanilla JS, Canvas API, simplex-noise
- **Pipeline**: Python, sentence-transformers, OpenAI API, UMAP, scikit-learn
- **Server**: Node.js, Express, OpenAI SDK
- **Deploy**: GitHub Actions, GitHub Pages

## License

MIT
