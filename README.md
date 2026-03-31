# Momentum

An AI-powered news aggregator that collects, scores, and organizes the most relevant artificial intelligence news from top sources — all running locally on your CPU, no cloud APIs or GPU required.

![Momentum Screenshot](https://img.shields.io/badge/status-active-brightgreen) ![Python](https://img.shields.io/badge/python-3.10+-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.135-009688)

## What it does

Momentum pulls news from sources like OpenAI, Anthropic, HuggingFace, NVIDIA, and The Decoder, then uses a local sentence-transformer model to score each article by relevance, filter out noise, and categorize everything automatically. The result is a clean, ranked feed you can check in 30 seconds to know exactly what happened in AI.

The scoring system combines semantic similarity against curated AI-related queries, recency bonuses, title quality heuristics, and source reputation weights. Posts are deduplicated both by URL fingerprinting and semantic similarity to avoid near-duplicate stories from different outlets. A noise filter removes podcasts, job listings, sponsored content, and other low-value entries before they ever reach the feed.

Articles are organized into time buckets — today, this week, this month, and all time — and sorted by score within each bucket. The frontend displays a featured hero card for the top story and a responsive grid for everything else.

## Tech Stack

The backend is built with **FastAPI** and uses **BeautifulSoup** for HTML parsing and **Requests** for fetching data. Classification and scoring run on **sentence-transformers** using the `multi-qa-mpnet-base-dot-v1` model, which provides strong semantic understanding while remaining light enough for CPU inference. The model is downloaded once from HuggingFace Hub and cached locally, so subsequent runs work fully offline.

The frontend is a single `index.html` file with no frameworks — just vanilla HTML, CSS, and JavaScript. It communicates with the FastAPI backend and renders everything client-side.

Data persistence is handled through a simple JSON file. Old posts beyond 60 days are automatically purged to keep the file size manageable.

## Categories

Posts are automatically classified into six categories: **Research** (papers, architectures, benchmarks), **Product** (launches, APIs, feature updates), **Business** (funding, M&A, market analysis), **Policy** (regulation, safety, ethics), **Infrastructure** (hardware, open-source models, cloud), and **Applications** (AI applied to healthcare, robotics, finance, etc.).

## Getting Started

Clone the repository and install the dependencies:

```bash
git clone https://github.com/Dipss4/Momentum.git
cd Momentum
pip install -r requirements.txt
```

Run the server:

```bash
uvicorn main:app --reload
```

On first launch, the sentence-transformer model will be downloaded automatically (~420MB). After that, everything runs offline. Open `index.html` in your browser or navigate to `http://127.0.0.1:8000/docs` to explore the API.

## API Endpoints

`GET /news` — Returns scored and categorized news organized by time period. Accepts optional `category` and `limit` query parameters.

`GET /categories` — Lists all available categories with descriptions.

`GET /stats` — Returns dataset statistics including total posts, averages scores, and breakdowns by category and source.

## How Scoring Works

Each post receives a final score calculated as `(semantic_relevance + recency_bonus + title_quality_bonus) × source_weight`. Semantic relevance is the maximum cosine similarity between the post embedding and a set of curated AI-related query embeddings. Recency adds a small bonus for very recent posts. Source weight gives a multiplier to posts from high-signal origins like OpenAI or Anthropic. Posts scoring below 0.28 are discarded entirely.
