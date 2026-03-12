# Sentiment Analysis Service

A FastAPI microservice that classifies text as **positive**, **neutral**, or **negative** using a fine-tuned DistilBERT model. The service is designed to consume text data from the ms-monolith application and return sentiment predictions via a REST API.

## Project Structure

```
sentiment-analysis-service/
├── main.py                  <- FastAPI application entry point
├── Makefile                 <- Commands for running the pipeline and server
├── config.yaml              <- Configuration for the legacy ML pipeline
├── requirements.txt         <- Project dependencies
├── src/
│   ├── api/                 <- API routes and Pydantic request/response models
│   ├── core/                <- Application settings (pydantic-settings)
│   ├── dataclasses.py       <- Internal domain objects
│   ├── services/            <- Business logic orchestration layer
│   ├── model/               <- BERT model wrapper + legacy ML model
│   ├── data/                <- Data loading and I/O utilities
│   ├── preprocess/          <- Feature engineering and preprocessing
│   ├── evaluate/            <- Model evaluation and metrics
│   ├── visualization/       <- Plotting and reporting
│   └── common/              <- Shared utilities (config loading, serialization)
└── notebooks/               <- Jupyter notebooks for EDA
```

## Getting Started

### 1. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the BERT model

Pre-caches the DistilBERT weights locally so the first API request doesn't block on a download (~250MB):

```bash
make download-model
```

### 4. Start the API server

```bash
make serve
```

The server starts on `http://localhost:8000`. Interactive API docs are available at `http://localhost:8000/docs`.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/sentiment` | Classify a single text |
| `POST` | `/api/v1/sentiment/batch` | Classify multiple texts |
| `GET` | `/api/v1/model/info` | Model metadata and supported labels |

### Example request

```bash
curl -X POST http://localhost:8000/api/v1/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "The delivery was fast and the product works great!"}'
```

### Example response

```json
{
  "text": "The delivery was fast and the product works great!",
  "label": "positive",
  "score": 0.9871,
  "scores": {
    "positive": 0.9871,
    "neutral": 0.0098,
    "negative": 0.0031
  },
  "model_name": "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
}
```

## Model

**`lxyuan/distilbert-base-multilingual-cased-sentiments-student`** — a DistilBERT model (~66M parameters) fine-tuned for 3-class sentiment classification via knowledge distillation. Supports multilingual input and achieves ~88% accuracy. Model weights are cached in `models/bert_cache/` after the first download.

## Makefile Commands

| Command | Description |
|---------|-------------|
| `make serve` | Start the FastAPI server |
| `make download-model` | Pre-download BERT model weights |
| `make download` | Download legacy pipeline data |
| `make preprocess` | Run preprocessing pipeline |
| `make train` | Train the legacy ML model |
| `make test` | Run predictions with the legacy model |
| `make evaluate` | Evaluate legacy model performance |
| `make visualize` | Generate visualizations |
| `make clean` | Remove compiled Python files |
| `make clean-all` | Remove all data, models, and results |

## Configuration

API settings are managed via `src/core/config.py` using pydantic-settings. All values can be overridden with environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PROJECT_NAME` | `Sentiment Analysis Service` | Service name |
| `DEBUG` | `true` | Debug mode |
| `MODEL_NAME` | `lxyuan/distilbert-base-multilingual-cased-sentiments-student` | HuggingFace model ID |
| `MODEL_CACHE_DIR` | `models/bert_cache` | Local model cache path |
| `MAX_INPUT_LENGTH` | `512` | Max token length for inference |
