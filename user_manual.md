# 🧪 LLM Prompt Evaluator: Complete User Manual

Welcome to your local prompt engineering command center! This tool is designed to remove the guesswork from writing AI prompts by scientifically scoring them, checking for hallucinations, and automatically optimizing them if they fail.

Below is your complete guide on how to navigate the tools, format your inputs, and understand the metrics.

---

## ⚙️ Core Concepts & Global Settings

Before running tests, it's important to understand the global settings available under the **⚙️ Settings** expander found on most pages.

| Setting | What it Does | Recommended Usage |
|---------|-------------|-------------------|
| **Select Model** | Chooses which local Ollama model to run. | Use `llama3` for complex logic and reasoning. Use `phi3:mini` for fast, lightweight testing. |
| **Temperature** | Controls the "creativity" or randomness of the AI. | **0.0 - 0.2**: Factual tasks (coding, math, exact extraction).<br>**0.5 - 0.7**: General tasks (emails, summaries).<br>**0.8 - 1.0**: Creative brainstorming (stories, ideas). |
| **LLM-as-Judge** | A powerful feature where the AI evaluates *its own* answer for factual correctness, completeness, and clarity. | Enable this for highly complex prompts where exact wording doesn't matter, but the *meaning/facts* must be perfectly accurate. *(Note: Adds 3-5s per test)*. |

> [!TIP]
> **What is an Expected Output?**  
> Throughout the app, you will be asked for an "Expected Output". This is the **Ground Truth** or the "Perfect Answer". The system compares the LLM's actual response to this exact text using mathematical semantic embeddings `all-MiniLM-L6-v2` to generate a score.

---

## 🧭 The 6 Dashboard Pages

### 1. 🌐 Overview (The Landing Page)
This is your global analytics dashboard. It pulls data directly from your SQLite database to show you:
- **Metrics**: Total tests run, your average prompt score, and hallucination rates.
- **Score Trend & Leaderboard**: A graph of your recent scores locally, and a ranking of which Model performs best on your machine.
- **Category Performance**: Shows which types of prompts your models excel at (e.g., Coding vs Creative).

### 2. 🔬 Single Test
Use this tab to rapidly test, score, and optimize a single prompt.

**How to use it:**
1. Type a **Prompt** (e.g., *"Explain quantum computing to a 5-year-old using analogy of a playground."*)
2. Type an **Expected Output** (e.g., *"Quantum computing is like being able to play on the swings, the slide, and the seesaw all at the exact same time..."*)
3. Click **Generate & Evaluate**.

**What happens next:**
- You get a final score out of 100.
- A **Hallucination Risk** flag (LOW, MEDIUM, HIGH) warns you if the AI made up facts or flat-out contradicted your prompt.
- **Optimization**: If you score below 90, an **Optimize & Compare** button appears. Click it, and the AI will automatically rewrite your prompt to be better, testing it to prove the score goes up!
- **PDF Export**: You can download a clean PDF report of the evaluation to share with your team.

### 3. 📊 Bulk Eval
Used for testing massive datasets at once to find weak points in your models/prompts.

**How to use it:**
1. Upload a CSV file (or use the default `data/prompts.csv`). 
2. **Crucial CSV Format:** Your CSV *must* contain at least two columns titled exactly: `prompt` and `expected_output`. An optional `category` column is recommended.
3. Click **Run Batch Test**. The app will stream responses, score hundreds of prompts, and provide a downloadable CSV of all the results.

> [!IMPORTANT]
> The Bulk Eval tab includes a powerful **Auto-Optimize Failing Prompts** button. If 10 of your prompts score below 60/100, clicking this button will rewrite and re-test all 10 automatically!

### 4. 🔀 A/B Compare (Prompt Compare)
Have 3 different ideas on how to phrase a prompt but don't know which is best? Put them to the test!

**How to use it:**
1. Enter multiple prompts, **strictly one per line**.
   *Prompt 1: "Summarize this text:"*
   *Prompt 2: "You are an expert editor. Provide a concise summary of the following:"*
   *Prompt 3: "TL;DR this passage:"*
2. Provide the single Expected Output you want them all to match.
3. Click **Compare & Rank**. The engine will crown a statistical `🥇 Best Prompt` and calculate the exact mathematical difference between your phrasing variants.

### 5. ⚖️ Model Compare
Want to know if `phi3` or `llama3` is smarter for your specific use case?

**How to use it:**
1. Select exactly **two models** from the dropdowns (e.g., Model 1: `llama3`, Model 2: `phi3:mini`).
2. Type your prompt and expected output.
3. The app will fire the prompt to both locally running AI brains simultaneously. It will output a side-by-side scorecard showing who generated a better response and who was faster (Latency)!

### 6. 📜 History
A simple ledger of every evaluation you've ever saved to the database. Excellent for retrieving a prompt you wrote three days ago that worked perfectly.

---

## 🧮 How the Evaluation Engine Actually Scores You

The final score you see (e.g., `85 / 100`) is not a random number. It is a strict algorithmic composite calculated internally in `src/evaluator.py`:

- **40% - Semantic Similarity**: We run the LLM output and your Expected Output through a `SentenceTransformer` AI vector model. If the *meaning* matches, the score is high (even if the exact words are different).
- **40% - LLM Judge (If enabled)**: Triggers a secondary invisible prompt asking the AI to grade the primary response strictly on a 1-10 scale for clarity and correctness.
- **20% - Length Penalty**: If you asked for a 2-sentence summary and the AI writes 8 paragraphs, the length penalty aggressively drops the score.

> [!WARNING]  
> **Understanding Hallucination Risks**  
> If the UI flags a `HIGH Risk`, it means the system detected severe contradictions between your prompt instructions and the output. For example, if your prompt said "do not use numbers" and the output contains numbers, or if the semantic similarity to your expected output drops below ~35%. Always double-check HIGH risk outputs.

---

## 🎯 Best Practices for Prompting in this App

To get the most out of the Optimizer and Evaluator, follow these rules:

1. **Be Specific in your Expected Output**: The system is mathematical. If your Expected Output just says "A good summary", the Semantic Embedder will grade poorly. Write exactly what you practically want to see.
2. **Use the Template System**: In the Single Test tab, use the dropdown to load predefined templates. Study how they are structured with clear context and constraints.
3. **Iterate on Failures**: If your score is 45/100, do not just manually rewrite the prompt. Click the `Optimize` button and watch how the AI restructures your prompt. Often, it will add strict boundaries (e.g., *"Respond under 100 words. Do not use filler."*) that skyrocket your score.
