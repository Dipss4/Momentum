from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
import os, json, hashlib, re
from sentence_transformers import SentenceTransformer, util

app = FastAPI(title="AI News Aggregator API v2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
JSON_FILE = "news_data.json"
MAX_AGE_DAYS = 60          # purge posts older than this
TOP_N_PER_PERIOD = 60      # max posts returned per period


from huggingface_hub import snapshot_download
from huggingface_hub import login

repo_id = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
model_name = repo_id.split("/")[-1]
local_dir = os.path.join("./model_data", model_name)

model_exists = os.path.exists(os.path.join(local_dir, "config.json"))

if not model_exists:
    print("⬇️ Modelo não encontrado. Baixando...")
    os.makedirs(local_dir, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )

    print("✅ Modelo baixado.")
else:
    print("✅ Modelo já existe. Usando local.")

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

model = SentenceTransformer(local_dir)


model = SentenceTransformer("./model_data/multi-qa-mpnet-base-dot-v1")

# ─────────────────────────────────────────────
# Source weights  (relevance multiplier)
# ─────────────────────────────────────────────
SOURCE_WEIGHTS = {
    "openai.com":        1.60,
    "anthropic.com":     1.60,
    "deepmind.google":   1.55,
    "blogs.nvidia.com":  1.45,
    "huggingface.co":    1.35,
    "mistral.ai":        1.30,
    "the-decoder.com":   1.10,
    "techcrunch.com":    1.05,
    "venturebeat.com":   1.10,
    "arxiv.org":         1.40,
    "technologyreview.com": 1.25,
    "default":           1.00,
}

# ─────────────────────────────────────────────
# Category definitions  (label → queries)
# ─────────────────────────────────────────────
CATEGORIES = {
    "Research": [
        "new AI research paper or academic study published",
        "breakthrough in deep learning or neural network architecture",
        "large language model training or fine-tuning technique",
        "multimodal model vision language audio research",
        "reinforcement learning from human feedback RLHF paper",
        "benchmark evaluation results comparing AI models",
        "arxiv preprint machine learning transformer attention",
    ],
    "Product": [
        "new AI product launch or feature release announced",
        "AI assistant chatbot or copilot product update",
        "API release or developer platform announced",
        "AI model version upgrade GPT Claude Gemini Llama",
        "AI SaaS tool or application released to public",
    ],
    "Business": [
        "AI startup funding round venture capital investment",
        "AI company acquisition merger partnership deal",
        "AI market revenue growth forecast report",
        "AI enterprise adoption deployment at scale",
        "AI company valuation IPO stock market",
    ],
    "Policy": [
        "AI regulation law government policy compliance",
        "AI ethics safety alignment risks concerns",
        "AI governance framework standards international",
        "EU AI Act regulation enforcement rules",
        "AI copyright intellectual property legal ruling",
    ],
    "Infrastructure": [
        "GPU chip hardware accelerator for AI training",
        "AI data center cloud computing infrastructure investment",
        "AI inference optimization quantization efficiency",
        "open source AI model weights released",
        "AI distributed training cluster supercomputer",
    ],
    "Applications": [
        "AI applied to healthcare medicine drug discovery",
        "AI used in robotics autonomous systems embodied",
        "AI application in education creative tools design",
        "AI in finance fraud detection trading",
        "AI transforming scientific research discovery",
    ],
}

# Flat list for relevance scoring (all queries together)
ALL_RELEVANCE_QUERIES = [
    "artificial intelligence machine learning deep learning news",
    "large language model AI breakthrough development",
    "AI technology advancement research product",
    "neural network AI system performance improvement",
    "foundation model training inference deployment",
]
# Add all category queries too
for qs in CATEGORIES.values():
    ALL_RELEVANCE_QUERIES.extend(qs)

# Pre-compute query embeddings once at startup
print("⚙  Pre-computing query embeddings…")
_relevance_embs = model.encode(ALL_RELEVANCE_QUERIES, convert_to_tensor=True)
_category_embs  = {
    cat: model.encode(qs, convert_to_tensor=True)
    for cat, qs in CATEGORIES.items()
}
print("Embeddings ready.")

# ─────────────────────────────────────────────
# Noise / spam filters
# ─────────────────────────────────────────────
NOISE_PATTERNS = [
    r"\b(podcast|episode|interview|recap|roundup|weekly digest)\b",
    r"\b(sponsored|partner content|advertis)\b",
    r"\bjobs?\b.*\bai\b",        # job listing noise
    r"\b(giveaway|contest|discount|coupon)\b",
]
_noise_re = re.compile("|".join(NOISE_PATTERNS), re.IGNORECASE)

MIN_RELEVANCE_SCORE = 0.28   # posts below this are discarded


# ─────────────────────────────────────────────
# Source scrapers
# ─────────────────────────────────────────────
def decode_API():
    try:
        url = "https://the-decoder.com/wp-admin/admin-ajax.php"
        data = {"action":"load_more_posts","page":"1","post_type":"post",
                "is_kipro":"0","nonce":"194feab3b3"}
        r = requests.post(url, data=data, timeout=10)
        soup = BeautifulSoup(r.json()["html"], "html.parser")
        posts = []
        for article in soup.find_all("article"):
            title_tag = article.find("h2")
            if not title_tag: continue
            link_tag  = article.find("a", href=True)
            paragraph = article.find("p")
            date_span = article.find("span", {"alt": "Date of publication"}) or \
                        soup.find("span", {"alt": "Date of publication"})
            date_paper = None
            if date_span:
                try:
                    d = datetime.strptime(date_span.get_text(strip=True), "%b %d, %Y")
                    date_paper = d.replace(hour=10, tzinfo=timezone.utc).isoformat()
                except: pass
            posts.append({
                "title":   title_tag.get_text(strip=True),
                "summary": paragraph.get_text(strip=True) if paragraph else None,
                "link":    link_tag["href"] if link_tag else None,
                "date":    date_paper,
                "site":    "the-decoder.com",
            })
        return posts
    except Exception as e:
        print(f"[decode_API] {e}"); return []


def openai_API():
    try:
        url = "https://openai.com/backend/articles/"
        params = {
            "locale":"en-US",
            "pageQueries":'[{"pageTypes":["Article"],"categories":["company","research","product","engineering","safety","security","ai-adoption"]}]',
            "limit":12,"skip":0,"sort":"new","groupedTags":"","search":""
        }
        headers = {
            "user-agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
            "accept":"application/json","referer":"https://openai.com/news/",
        }
        data = requests.get(url, params=params, headers=headers, timeout=10).json()
        return [
            {"title":   i.get("title"),
             "summary": i.get("seoFields",{}).get("metaDescription"),
             "link":    f"https://openai.com/{i.get('slug')}/",
             "date":    i.get("publicationDate"),
             "site":    "openai.com"}
            for i in data.get("items", [])
        ]
    except Exception as e:
        print(f"[openai_API] {e}"); return []


def hugface_API():
    try:
        data = requests.get("https://huggingface.co/api/blog?p=0", timeout=10).json()
        posts = data.get("allBlogs", []) + data.get("communityBlogPosts", [])
        result = [
            {"title":   p.get("title"),
             "summary": p.get("brief") or p.get("summary"),
             "link":    "https://huggingface.co" + p.get("url",""),
             "date":    p.get("publishedAt"),
             "site":    "huggingface.co"}
            for p in posts
        ]
        return sorted(result, key=lambda x: x["date"] or "", reverse=True)[:20]
    except Exception as e:
        print(f"[hugface_API] {e}"); return []


def nvia_API():
    try:
        url = "https://blogs.nvidia.com/wp-json/nvidia-blog-v6/v2/recent-news-page-load-more"
        data = requests.get(url, params={"posts_per_page":12}, timeout=10).json()
        soup = BeautifulSoup(data["data"]["html"], "html.parser")
        result = []
        for article in soup.find_all("article"):
            t = article.find("h3") and article.find("h3").find("a")
            s = article.find("div", class_="recent-news-post-excerpt")
            d = article.find("time")
            if not t: continue
            result.append({
                "title":   t.text.strip(),
                "summary": s.text.strip() if s else None,
                "link":    t["href"],
                "date":    d["datetime"] if d else None,
                "site":    "blogs.nvidia.com",
            })
        return result
    except Exception as e:
        print(f"[nvia_API] {e}"); return []

# ─────────────────────────────────────────────
# Deduplication (semantic + URL fingerprint)
# ─────────────────────────────────────────────
def url_fingerprint(url: str) -> str:
    """Normalize URL to a stable key (strips tracking params, trailing slashes)."""
    url = re.sub(r"\?.*$", "", url.strip().rstrip("/").lower())
    return hashlib.md5(url.encode()).hexdigest()


def semantic_dedup(new_posts: list, existing_posts: list, threshold: float = 0.92) -> list:
    """Remove posts that are semantically near-duplicate of existing ones."""
    if not existing_posts or not new_posts:
        return new_posts

    existing_texts = [p["title"] + " " + (p.get("summary") or "") for p in existing_posts]
    new_texts      = [p["title"] + " " + (p.get("summary") or "") for p in new_posts]

    ex_embs  = model.encode(existing_texts, convert_to_tensor=True, batch_size=64)
    new_embs = model.encode(new_texts,      convert_to_tensor=True, batch_size=64)

    keep = []
    for i, (post, emb) in enumerate(zip(new_posts, new_embs)):
        sims = util.cos_sim(emb.unsqueeze(0), ex_embs)[0]
        if float(sims.max()) < threshold:
            keep.append(post)
        else:
            print(f"  [dedup] Dropped near-duplicate: {post['title'][:60]}")
    return keep


# ─────────────────────────────────────────────
# Scoring  (multi-factor)
# ─────────────────────────────────────────────
def score_post(post: dict, post_emb) -> dict:
    """
    Final score = (semantic_relevance + recency_bonus + title_quality_bonus) * source_weight

    semantic_relevance : max cosine similarity to any relevance query  [0..1]
    recency_bonus      : +0.15 if today, +0.10 if ≤2d, +0.05 if ≤7d
    title_quality_bonus: small reward for longer, substantive titles
    source_weight      : multiplier from SOURCE_WEIGHTS
    """
    now = datetime.now(timezone.utc)

    # 1. Semantic relevance
    sims = util.cos_sim(post_emb, _relevance_embs)[0]
    relevance = float(sims.max())

    # 2. Recency
    date_str = post.get("date")
    recency  = 0.0
    age_days = None
    if date_str:
        try:
            d = datetime.fromisoformat(date_str)
            if d.tzinfo is None: d = d.replace(tzinfo=timezone.utc)
            age_days = (now - d).total_seconds() / 86400
            if age_days <= 1:   recency = 0.15
            elif age_days <= 2: recency = 0.10
            elif age_days <= 7: recency = 0.05
        except: pass

    # 3. Title quality (length proxy — very short titles are usually low-value)
    title_len  = len((post.get("title") or "").split())
    title_bonus = 0.02 if title_len >= 7 else 0.0

    # 4. Source weight
    source_weight = SOURCE_WEIGHTS.get(post.get("site"), SOURCE_WEIGHTS["default"])

    raw_score = (relevance + recency + title_bonus) * source_weight
    post["score"]    = round(raw_score, 4)
    post["age_days"] = round(age_days, 1) if age_days is not None else None
    return post


def assign_category(post: dict, post_emb) -> dict:
    """Find the best-matching category for a post."""
    best_cat, best_sim = "General", 0.0
    for cat, embs in _category_embs.items():
        sim = float(util.cos_sim(post_emb, embs)[0].max())
        if sim > best_sim:
            best_sim, best_cat = sim, cat
    post["category"]     = best_cat
    post["category_sim"] = round(best_sim, 3)
    return post


def is_noisy(post: dict) -> bool:
    text = (post.get("title") or "") + " " + (post.get("summary") or "")
    return bool(_noise_re.search(text))


def classify_and_score(posts: list) -> list:
    """Full pipeline: embed → filter noise → score → categorize → sort."""
    texts = [p["title"] + " " + (p.get("summary") or "") for p in posts]
    embs  = model.encode(texts, convert_to_tensor=True, batch_size=64)

    results = []
    for post, emb in zip(posts, embs):
        if is_noisy(post):
            print(f"  [noise] Dropped: {post['title'][:60]}")
            continue

        post = score_post(post, emb)
        if post["score"] < MIN_RELEVANCE_SCORE:
            print(f"  [low-score {post['score']:.2f}] Dropped: {post['title'][:60]}")
            continue

        post = assign_category(post, emb)
        results.append(post)

    return sorted(results, key=lambda x: x["score"], reverse=True)


# ─────────────────────────────────────────────
# Date bucketing
# ─────────────────────────────────────────────
def organize_by_date(posts: list) -> dict:
    now = datetime.now(timezone.utc)
    buckets = {"today": [], "week": [], "month": [], "always": []}
    for post in posts:
        date_str = post.get("date")
        try:
            d = datetime.fromisoformat(date_str) if date_str else None
            if d and d.tzinfo is None: d = d.replace(tzinfo=timezone.utc)
        except: d = None

        buckets["always"].append(post)
        if d:
            if d.date() == now.date():
                buckets["today"].append(post)
            if d >= now - timedelta(days=7):
                buckets["week"].append(post)
            if d >= now - timedelta(days=30):
                buckets["month"].append(post)
    return buckets


# ─────────────────────────────────────────────
# JSON persistence
# ─────────────────────────────────────────────
def load_json() -> dict:
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"always": [], "week": [], "month": [], "today": []}


def save_json(data: dict):
    # Purge very old posts from "always" to keep file size sane
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=MAX_AGE_DAYS)
    def is_recent(p):
        try:
            d = datetime.fromisoformat(p.get("date",""))
            if d.tzinfo is None: d = d.replace(tzinfo=timezone.utc)
            return d >= cutoff
        except: return True  # keep if date unknown
    data["always"] = [p for p in data["always"] if is_recent(p)]

    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────
@app.get("/news")
def get_news(
    category: str = Query(None, description="Filter by category (Research, Product, Business, Policy, Infrastructure, Applications)"),
    limit: int = Query(TOP_N_PER_PERIOD, description="Max posts per period"),
):
    existing_data = load_json()

    # Fetch from all sources
    print("🔍 Fetching from all sources…")

    raw_posts = (
        decode_API() +
        openai_API() +
        hugface_API() +
        nvia_API() 
    )
    print(f"   Raw posts fetched: {len(raw_posts)}")

    # Filter out bad records
    raw_posts = [p for p in raw_posts if p.get("title") and p.get("link")]

    # URL-based dedup against existing
    existing_fingerprints = {url_fingerprint(p["link"]) for p in existing_data["always"]}
    url_new = [p for p in raw_posts if url_fingerprint(p["link"]) not in existing_fingerprints]
    print(f"   After URL dedup: {len(url_new)} new posts")

    if not url_new:
        # Return existing data (possibly filtered by category)
        return _build_response(existing_data, category, limit)

    # Semantic dedup against existing
    sem_new = semantic_dedup(url_new, existing_data["always"][-300:])
    print(f"   After semantic dedup: {len(sem_new)} posts")

    if not sem_new:
        return _build_response(existing_data, category, limit)

    # Score + categorize
    print("🧠 Scoring and categorizing…")
    scored = classify_and_score(sem_new)
    print(f"   After filtering: {len(scored)} posts kept")

    # Organize into time buckets
    organized = organize_by_date(scored)

    # Merge into existing
    for k in existing_data:
        existing_data[k] = organized[k] + existing_data[k]  # new first
        # Re-sort always bucket by score descending
        existing_data[k] = sorted(existing_data[k], key=lambda x: x.get("score", 0), reverse=True)

    save_json(existing_data)
    print("✅ Done.")

    return _build_response(existing_data, category, limit)


def _build_response(data: dict, category: str | None, limit: int) -> JSONResponse:
    """Optionally filter by category and apply limit."""
    result = {}
    for period, posts in data.items():
        if category:
            posts = [p for p in posts if p.get("category") == category]
        result[period] = posts[:limit]
    return JSONResponse(content=result)


@app.get("/categories")
def list_categories():
    """Returns available categories and their descriptions."""
    return JSONResponse(content={
        "categories": list(CATEGORIES.keys()),
        "descriptions": {
            "Research":       "Academic papers, model architectures, benchmarks",
            "Product":        "Product launches, API releases, feature updates",
            "Business":       "Funding, M&A, market analysis",
            "Policy":         "Regulation, AI safety, ethics, governance",
            "Infrastructure": "Hardware, open-source models, cloud infra",
            "Applications":   "AI applied to specific industries and domains",
        }
    })


@app.get("/stats")
def get_stats():
    """Returns statistics about the stored news dataset."""
    data = load_json()
    always = data.get("always", [])
    cat_counts = {}
    for p in always:
        c = p.get("category", "Unknown")
        cat_counts[c] = cat_counts.get(c, 0) + 1
    site_counts = {}
    for p in always:
        s = p.get("site", "unknown")
        site_counts[s] = site_counts.get(s, 0) + 1
    avg_score = round(sum(p.get("score",0) for p in always) / max(len(always),1), 3)
    return JSONResponse(content={
        "total_posts": len(always),
        "today": len(data.get("today", [])),
        "week":  len(data.get("week", [])),
        "month": len(data.get("month", [])),
        "avg_score": avg_score,
        "by_category": cat_counts,
        "by_site": site_counts,
    })