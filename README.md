# LLM Prompt Evaluator & Optimizer

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.39+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-000000?style=flat-square&logo=ollama&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?style=flat-square&logo=sqlite&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Privacy](https://img.shields.io/badge/Privacy-100%25_Local-blueviolet?style=flat-square)

**A privacy-first, local evaluation dashboard to scientifically test, benchmark, and auto-optimize your LLM prompts.**

[Features](#-features) · [Demo](#-dashboard-preview) · [Installation](#-installation) · [How It Works](#-how-it-works) · [Tech Stack](#-tech-stack)

</div>

---

## 🎥 Project Demo Video
[Click here to watch the demo](https://drive.google.com/file/d/1_SNuM0YmNr3a5YBYIJ0PFjm0zU6t_uTy/view?usp=drive_link)

## 📌 Overview

The **LLM Prompt Evaluator & Optimizer** is an open-source dashboard built for AI engineers, developers, and researchers who need to **measure** prompt quality — not just guess at it.

Everything runs **100% locally** via [Ollama](https://ollama.com). Your prompts, data, and outputs never leave your machine.

---

## ✨ Features

### 🔬 Single-Shot Evaluation
Test a prompt against an expected output and receive a composite score out of 100 with detailed feedback — semantic similarity, LLM-as-Judge verdict, and length penalty breakdown.

### 🧠 Hallucination Detection
A three-layer detection engine that flags unreliable outputs:
- **Rule-based detection** — catches vague fillers, contradictions, overconfident claims, and numeric inconsistencies
- **Dual embedding comparison** — measures semantic drift from both the expected output and the original prompt
- **Risk classifier** — combines all signals into a final `HIGH / MEDIUM / LOW` verdict

### ⚖️ Multi-Model Comparison
Run the same prompt through `llama3` and `phi3:mini` simultaneously. Get side-by-side score breakdowns, latency tracking, a visual bar chart, and an automatic winner declaration.

### 📊 Bulk CSV Evaluation
Upload a dataset of prompts and expected outputs. The engine streams responses, scores them in real time, and generates category-level analytics (e.g. your model scores 88 on Factual but only 45 on Math).

### 🔁 Consistency Testing
Run the same prompt 5 times and calculate standard deviation to measure how stable or hallucination-prone a model is across repeated calls.

### 📜 Audit History
Every evaluation is automatically saved to a local SQLite database. Track your score trends over time, replay past results, and compare progress across sessions.

### 📄 PDF Report Export
Export any single evaluation as a clean, structured PDF report — prompt, expected output, model response, final score, and hallucination risk — with a single click.


---

## 🖥️ Dashboard Preview

```
┌─────────────────────────────────────────────────────────────────────┐
│  ● LLM Evaluator   Overview  Single Test  Bulk Eval  Model Compare  │
│                                              ● Ollama Connected     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Total Tests    Avg Score    Hallucination Rate    Best Model       │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  1,284   │  │   73.4   │  │     18%      │  │   llama3     │   │
│  │ +38 today│  │ +2.1 ↑   │  │ 12 HIGH today│  │ avg 81.2/100 │   │
│  └──────────┘  └──────────┘  └──────────────┘  └──────────────┘   │
│                                                                     │
│  Category Performance          Recent Evaluations                   │
│  Factual    ████████████ 88    llama3  "What is the capital..." 91  │
│  Reasoning  █████████░░░ 74    phi3    "Explain quantum..."    63   │
│  Creative   ███████░░░░░ 61    llama3  "Write a poem about..." 58   │
│  Code       ██████████░░ 79    phi3    "Solve: 2x + 5 = 13"   41   │
│  Math       █████░░░░░░░ 45                                         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ How It Works

The evaluation engine runs a mathematically weighted pipeline on every response:

```
Raw Prompt ──► Target LLM ──► Output
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
              Semantic AI   LLM Judge   Length Penalty
               Embeddings  (Factual +   (Verbosity
              (40% weight)  Complete +   Control)
                            Clarity)   (20% weight)
                           (40% weight)
                    └───────────┼───────────┘
                                ▼
                      Final Score (0–100)
                                │
                                ▼
                    Hallucination Analysis
                  (Rule-based + Embeddings
                   + Risk Classifier)
```

| Component | Method | Weight |
|-----------|--------|--------|
| Semantic Similarity | SentenceTransformers `all-MiniLM-L6-v2` | 40% |
| LLM-as-Judge | Local Ollama model reviews correctness, completeness, clarity | 40% |
| Length Penalty | Penalises excessive verbosity / hallucinated padding | 20% |

---

## 🚀 Installation

### Prerequisites

1. Install [Python 3.10+](https://www.python.org/downloads/)
2. Install [Ollama](https://ollama.com/) and pull your models:

```bash
ollama pull llama3
ollama pull phi3:mini
```

### Setup

```bash
# Clone the repository
git clone https://github.com/SanaSagar/AI-Internship-training-project.git
cd llm-prompt-evaluator

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

---

## 📁 Project Structure

```
llm-prompt-evaluator/
│
├── app.py                  # Main Streamlit dashboard
│
├── src/
│   ├── evaluator.py        # Core evaluation pipeline (40/40/20 scoring)
│   ├── embeddings.py       # SentenceTransformer semantic similarity
│   ├── hallucination.py    # Three-layer hallucination detection engine
│   ├── model_comparison.py # Multi-model benchmarking engine
│   ├── pdf_report.py       # Single-test PDF export (FPDF2)
│   ├── report.py           # Bulk/history PDF report generation
│   └── utils.py            # SQLite database helpers
│
├── data/
│   └── prompts.csv         # Sample bulk evaluation dataset
│
├── requirements.txt
└── README.md
```

---

## 📦 Tech Stack

| Tool | Purpose |
|------|---------|
| [Streamlit](https://streamlit.io/) | Reactive frontend dashboard |
| [Ollama](https://ollama.com/) | Local LLM hosting (llama3, phi3:mini) |
| [SentenceTransformers](https://sbert.net/) | Semantic similarity via `all-MiniLM-L6-v2` |
| [SQLite](https://www.sqlite.org/) | Persistent local evaluation history |
| [FPDF2](https://pyfpdf.github.io/fpdf2/) | PDF report generation |
| [Pandas](https://pandas.pydata.org/) | CSV processing and analytics |
| [Matplotlib](https://matplotlib.org/) | Category performance charts |

---

## 🔒 Privacy

This project is designed with a **zero-data-leak architecture**:

- All LLM inference runs locally via Ollama
- All embeddings are computed locally via SentenceTransformers
- All evaluation history is stored in a local SQLite file
- No data is sent to any external API or cloud service

---

## 📄 CSV Format for Bulk Evaluation

To use the Bulk Eval feature, upload a `.csv` file with the following columns:

```csv
prompt,expected_output,category
"What is the capital of France?","Paris","Factual"
"Solve: 2x + 5 = 13","x = 4","Math"
"Write a haiku about rain","...","Creative"
```

The `category` column is optional but enables category-level analytics on the dashboard.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue first to discuss what you'd like to change.

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---


<div align="center">
  <sub>Built with Streamlit + Ollama · 100% local · 100% private</sub>
</div>
