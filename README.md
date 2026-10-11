# 🔮 VisualToken Box: Mechanistic Interpretability Powered by Gemma 4

**By Karan Ray** — An interactive, zero-black-box Transformer engine built in pure NumPy, paired with **Gemma 4** (via Gemini API) as an AI Interpretability Co-Pilot to explain attention heatmaps, diagnose hallucinations, and benchmark model behaviors in real time.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![Inference](https://img.shields.io/badge/Inference-Pure%20NumPy-orange.svg)](https://numpy.org)
[![Analysis](https://img.shields.io/badge/Analysis-Gemma%204%20AI-purple.svg)](https://aistudio.google.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

VisualToken Box runs open-weight **SmolLM2-135M** locally without any PyTorch runtime dependency, exposing every matrix multiplication, RoPE rotation, and attention head. By pairing this local inspectable core with **Gemma 4**, it transforms raw numbers into actionable mechanistic insights.

---

## 🧠 Gemma 4: The AI Interpretability Co-Pilot

Rather than treating interpretability data as complex heatmaps that require a PhD to read, VisualToken Box deeply integrates **Gemma 4** into the generation workflow:

### 1. 🔮 Multimodal "Explain This Trace" (`POST /api/explain`)
- **Multimodal Perception**: Captures the live attention heatmap canvas as a high-resolution PNG snapshot alongside structured token metrics (per-step entropy, top-3 candidates, logit distributions).
- **Automated Root-Cause Analysis**: Gemma 4 audits the trace and flags suspicious generation steps (hallucinations, entropy spikes, attention sink traps) with clickable badges that jump straight to that step in the UI.

### 2. 🩺 Prompt & Hallucination Diagnostic (`POST /api/diagnose`)
- **Vulnerability Assessment**: Evaluates user prompts for semantic ambiguity, under-specification, and predictable failure points before or after generation.
- **Confidence Scoring**: Delivers a calibrated 0–100 confidence score estimating model success.
- **One-Click Rephrasing**: Proposes an optimized prompt rephrasing tailored for small language models that can be run with a single click.

### 3. ⚖️ Model Comparison: SmolLM2 vs Gemma 4 (`POST /api/compare`)
- **Real-Time Side-by-Side**: Concurrently runs local SmolLM2 token generation alongside Gemma 4 completion for identical prompts.
- **Divergence Metrics**: Computes lexical overlap percentages and token-level divergence, tracking inference speed (tok/s) against cloud API latency (ms).

### 4. 📥 Research-Grade Report Export
- **Self-Contained Audit**: Generates and downloads a complete, standalone HTML interpretability report with embedded attention heatmaps, per-token entropy tables, Logit Lens milestones, and Gemma 4 diagnostic assessments.

---

## 🧭 Core Engine Features

- 🔥 **Attention Heatmap Matrix**: Inspect token-to-token attention across all 30 layers and 9 attention heads. Toggle first-token attention sinks, normalize rows, or average across heads.
- 📊 **Live Probability Distributions**: Explore the top 200 candidate tokens considered at each step, alongside full-vocabulary entropy and softmax curves.
- 🔀 **What-If Branching**: Click any candidate token in the probability inspector to force its selection, re-route the generation tree, and compare alternative model paths.
- 🔭 **Logit Lens**: Project intermediate hidden states across all 30 transformer blocks through the unembedding head to watch predictions solidify layer by layer.
- ⚡ **KV Cache Diagnostics**: Toggle KV caching dynamically to observe speed-ups, latency deltas, and memory footprint in real time.
- 🌐 **Zero-Build Web UI**: Lightweight FastAPI backend streaming live token mechanics directly to vanilla HTML5, Canvas, and CSS.

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.9+ (Python 3.11 recommended)
- ~270 MB disk space for local model weights
- Google AI Studio API Key ([Get a free key here](https://aistudio.google.com))

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/karanray06/Gemma.git
cd Gemma

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Gemma 4 API Key

Copy the example environment file and add your Gemini API key:

```bash
cp .env.example .env
```

Inside `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 4. Download Model Weights & Launch

```bash
# Download SmolLM2-135M weights (~270MB)
python download_model.py

# Launch the visualizer server
python run_server.py
```

Open **[http://localhost:8000](http://localhost:8000)** in your browser!

---

## 🏗️ Architecture & Gemma Integration

```text
                               ┌────────────────────────────────────────────────────────┐
                               │                    User Interaction                    │
                               │        (Prompt, Hyperparameters, Image Inputs)         │
                               └──────────────────────────┬─────────────────────────────┘
                                                          │
                               ┌──────────────────────────┴─────────────────────────────┐
                               │             FastAPI Backend (glassbox/server)          │
                               └──────────────┬───────────────────────────┬─────────────┘
                                              │                           │
                   WebSocket / HTTP Stream    │                           │  REST Endpoints
                                              ▼                           ▼
            ┌───────────────────────────────────────────┐    ┌───────────────────────────────────────────┐
            │       NumPy SmolLM2-135M Engine           │    │       Gemma 4 Analysis Pipeline           │
            │          (glassbox/model.py)              │    │      (glassbox/gemini_client.py)          │
            ├───────────────────────────────────────────┤    ├───────────────────────────────────────────┤
            │ • RoPE Rotary Embeddings                  │    │ • /api/explain: Multimodal trace + canvas │
            │ • Grouped-Query Attention (9 Q / 3 KV)    │    │ • /api/diagnose: Ambiguity & Failure crit │
            │ • SwiGLU Feed-Forward Blocks              │    │ • /api/compare: Side-by-side completion   │
            │ • KV Caching with Layer Buffers           │    │ • Report Exporter: Self-contained HTML    │
            └─────────────────────┬─────────────────────┘    └─────────────────────┬─────────────────────┘
                                  │                                                │
                                  └───────────────────────┬────────────────────────┘
                                                          │
                                                          ▼
                               ┌────────────────────────────────────────────────────────┐
                               │             Real-Time Canvas UI (glassbox/web)         │
                               │     • Attention Matrix Heatmap     • Probability Bars  │
                               │     • Logit Lens Predictions       • Gemma Slide Panel │
                               └────────────────────────────────────────────────────────┘
```

---

## 📡 API Reference

| Endpoint | Method | Description | Input Payload |
|---|---|---|---|
| `/ws/generate` | `WebSocket` | Real-time token streaming with layer traces | `{ prompt, max_new_tokens, temperature, top_k, top_p, use_cache }` |
| `/api/explain` | `POST` | Gemma 4 multimodal analysis of trace + heatmap | `{ prompt, generated_tokens, top_k_per_step, entropy_per_step, heatmap_b64_png }` |
| `/api/diagnose` | `POST` | Gemma 4 prompt ambiguity & failure critique | `{ prompt, generated_tokens, entropy_per_step, top_k_per_step }` |
| `/api/compare` | `POST` | Generate completion with Gemma 4 for comparison | `{ prompt, temperature, max_tokens }` |
| `/health` | `GET` | Health status and loaded model confirmation | None |

---

## 🧪 Testing

Run smoke tests for all Gemma 4 endpoints and inference modules:

```bash
pytest tests/test_gemma_api.py -v
```

---

## 🗂️ Repository Structure

```text
Gemma/
├── glassbox/
│   ├── model.py            Pure NumPy Transformer forward pass (GQA, RoPE, SwiGLU)
│   ├── ops.py              Numerical primitives (RMSNorm, Softmax, RoPE frequencies)
│   ├── cache.py            KV cache manager for low-latency auto-regression
│   ├── trace.py            Logit Lens unembedding projection & layer extraction
│   ├── trace_payload.py    Compact JSON payload serializer
│   ├── gemini_client.py    Gemma 4 client (explain, diagnose, compare via Gemini API)
│   ├── server/             FastAPI application with WebSocket & REST endpoints
│   └── web/                Vanilla HTML5/Canvas/CSS visualizer dashboard
│       ├── app.js          Main frontend orchestrator & WebSocket handler
│       ├── gemma4.js       Gemma 4 UI module (Explain, Diagnose, Compare, Export)
│       ├── heatmap.js      Hardware-accelerated Attention Heatmap renderer
│       ├── probs.js        SVG probability bars with dynamic temperature reshaping
│       └── lens.js         Logit Lens layer-by-layer prediction tracker
├── tests/
│   └── test_gemma_api.py   Smoke tests for all Gemma 4 API routes
├── download_model.py       Hugging Face snapshot downloader for SmolLM2-135M
├── run_server.py           Local server entry point
├── pyproject.toml          Project metadata and dependencies
└── README.md               Project documentation
```

---

## 📄 License & Credits

- **Author**: Karan Ray ([2006karanray@gmail.com](mailto:2006karanray@gmail.com))
- **Repository**: [github.com/karanray06/Gemma](https://github.com/karanray06/Gemma)
- **License**: [MIT License](LICENSE)
- **Base Architecture**: SmolLM2-135M by HuggingFaceTB · Analysis powered by Google Gemma 4 via Gemini API