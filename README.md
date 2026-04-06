# Music-Recommendation-System-Spotify

[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org/)

A production-grade music recommendation system combining deep learning, graph neural networks, and reinforcement learning for personalized music discovery.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Tech Stack](#tech-stack)
5. [Installation](#installation)
6. [Data Preparation](#data-preparation)
7. [Model Training](#model-training)
8. [Running the API](#running-the-api)
9. [API Endpoints](#api-endpoints)
10. [Project Structure](#project-structure)
11. [Future Scope](#future-scope)

---

## Project Overview

Spotify Music Recommender is an end-to-end machine learning system that provides personalized music recommendations using a multi-stage pipeline:

- **Retrieval**: Two-Tower Neural Network + GNN (LightGCN) for candidate generation
- **Ranking**: Multi-Task Learning model predicting user engagement (likes, skips, completions)
- **Re-Ranking**: Multi-Armed Bandit for exploration vs exploitation balance
- **Intent Parsing**: LLM-powered natural language query understanding

The system is designed to handle millions of tracks with sub-10ms latency using FAISS vector search.

---

## Features

### Core ML Features

| Feature | Description |
|---------|-------------|
| **Two-Tower Neural Network** | User and item towers learning embeddings for efficient retrieval |
| **Graph Neural Network (LightGCN)** | Collaborative filtering via co-listening graph propagation |
| **Multi-Task Learning Ranker** | Simultaneously predicts like, skip, completion, and playlist probabilities |
| **Multi-Armed Bandit** | ε-greedy exploration for discovery vs exploitation balance |
| **LLM Intent Parser** | Translates natural language queries into recommendation filters |
| **FAISS Vector Search** | Local approximate nearest neighbor search for fast candidate retrieval |

### System Features

- Cold start handling via onboarding quiz
- Context-aware recommendations
- Semantic search with mood/intent understanding
- RESTful API with FastAPI
- 500K+ track catalog support

---


## Architecture:

![Architecture](https://github.com/user-attachments/assets/d6304164-6145-4bc1-b67c-2c5d7842ae63)

---

## Tech Stack

### Core Technologies

| Category | Technology | Purpose |
|-----------|------------|----------|
| **Language** | Python 3.13 | Core ML and backend |
| **ML Framework** | PyTorch 2.x | Neural network training |
| **Graph ML** | PyTorch Geometric | LightGCN implementation |
| **Vector Store** | FAISS | Approximate nearest neighbor search |
| **API** | FastAPI | REST API backend |
| **Frontend** | React | User interface |

### Key Libraries

```
torch>=2.2.0
torch-geometric>=2.5.0
transformers>=4.40.0
faiss-cpu>=1.7.0
fastapi>=0.111.0
uvicorn>=0.30.0
scikit-learn>=1.5.0
pandas>=2.2.0
numpy>=1.26.0
```

---

## Installation

### Prerequisites

- Python 3.13+
- pip package manager
- (Optional) CUDA for GPU acceleration

### Step 1: Clone and Setup Virtual Environment

```bash
# Navigate to project directory
cd music-recommendation-system-spotify

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Environment Variables (Optional)

Copy `.env.example` to `.env` if you need custom paths:

```bash
copy .env.example .env
```

---

## Data Preparation

### Step 1: Download Dataset

Download the Spotify dataset from Kaggle:
- **Source**: [Spotify Dataset 1921-2020, 160k tracks](https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks)
- **Files needed**:
  - `tracks.csv` → save as `data/raw/tracks.csv`
  - `artists.csv` → save as `data/raw/artists.csv` (if available)

### Step 2: Run Data Processing Scripts

```bash
# 1. Ingest and clean Kaggle data
python -c "import sys; sys.path.insert(0, '.'); from data_ingestion.ingest_kaggle import ingest_kaggle; ingest_kaggle()"

# 2. Build genre vocabulary
python -c "import sys; sys.path.insert(0, '.'); from feature_engineering.genre_encoder import build_genre_vocab; build_genre_vocab()"

# 3. Create audio feature vectors
python -c "import sys; sys.path.insert(0, '.'); from feature_engineering.audio_features import create_audio_vectors; create_audio_vectors()"

# 4. Generate simulated interactions (for training)
python -c "import sys; sys.path.insert(0, '.'); from data_ingestion.generate_simulated_interactions import generate_simulated_data; generate_simulated_data()"
```

---

## Model Training

### Training Pipeline

```bash
# 1. Train Two-Tower Model
python -c "import sys; sys.path.insert(0, '.'); from retrieval.two_tower_train import train_model; train_model()"

# 2. Export item embeddings
python -c "import sys; sys.path.insert(0, '.'); from retrieval.export_embeddings import export_item_embeddings; export_item_embeddings()"

# 3. Build FAISS index
python -c "import sys; sys.path.insert(0, '.'); from vector_store.index_vectors import index_all_tracks; index_all_tracks()"

# 4. Prepare GNN data
python -c "import sys; sys.path.insert(0, '.'); from retrieval.gnn_prepare import prepare_gnn_data; prepare_gnn_data()"

# 5. Train GNN (LightGCN)
python -c "import sys; sys.path.insert(0, '.'); from retrieval.gnn_train import train_gnn; train_gnn()"

# 6. Train Multi-Task Ranker
python -c "import sys; sys.path.insert(0, '.'); from ranking.mtl_train import train_ranker; train_ranker()"
```

### Training Output

After training, the following files are generated in `models/`:
- `two_tower_epoch5.pth` - Two-Tower model weights
- `gnn_lightgcn.pth` - GNN model weights
- `ranking_mtl.pth` - Multi-Task ranker weights

---

## Running the API

### Start the Backend

```bash
# From project root
python -m api.main
```

Or using uvicorn:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service status and track count |
| `/genres` | GET | Get top genres for onboarding |
| `/onboarding` | POST | Create user profile with selected genres |
| `/recommend/{user_id}` | GET | Get personalized recommendations |
| `/search` | GET | Semantic search with LLM intent |

### Example Usage

#### Get Recommendations

```bash
curl "http://localhost:8000/recommend/1?k=10"
```

Response:
```json
{
  "user_id": 1,
  "recommendations": [...],
  "total_served": 10
}
```

#### User Onboarding

```bash
curl -X POST "http://localhost:8000/onboarding?user_id=123" \
  -H "Content-Type: application/json" \
  -d '["pop", "rock", "jazz"]'
```

#### Semantic Search

```bash
curl "http://localhost:8000/search?query=high%20energy%20rock%20for%20workout"
```

---

## Project Structure

```
spotify-music-recommender/
├── api/
│   └── main.py                 # FastAPI application entry point
│
├── data/
│   ├── raw/                    # Raw data (tracks.csv, artists.csv)
│   └── processed/              # Cleaned data, vectors, indices
│
├── data_ingestion/
│   ├── ingest_kaggle.py        # Load and clean Kaggle dataset
│   └── generate_simulated_interactions.py  # Training data generator
│
├── feature_engineering/
│   ├── audio_features.py       # Audio feature vectorization
│   └── genre_encoder.py        # Genre vocabulary builder
│
├── retrieval/
│   ├── two_tower_model.py      # Two-Tower neural network
│   ├── two_tower_train.py     # Training script
│   ├── gnn_model.py            # LightGCN implementation
│   ├── gnn_train.py            # GNN training script
│   ├── candidate_generator.py # Retrieval pipeline
│   ├── onboarding.py           # Cold start handling
│   └── export_embeddings.py    # Item embedding export
│
├── ranking/
│   ├── mtl_model.py            # Multi-Task Learning ranker
│   └── mtl_train.py            # Ranker training script
│
├── reranking/
│   ├── bandit.py               # Multi-Armed Bandit for exploration
│   └── llm_intent_parser.py    # LLM-powered query understanding
│
├── vector_store/
│   ├── faiss_manager.py        # FAISS index management
│   └── index_vectors.py        # Index building script
│
├── models/                     # Saved model checkpoints
│
├── frontend/                   # React UI (optional)
│
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variables template
└── README.md                   # This file
```

---

## Future Scope

### Phase 2 Improvements

1. **Pinecone Migration** - Scale to managed cloud vector database
2. **Spotipy Integration** - Real-time Spotify API data sync
3. **Live Audio Analysis** - Extract vibes from raw audio signals
4. **Kafka Streaming** - Real-time signal processing for user events
5. **Feast Feature Store** - Online feature serving for production


