"""
LLM Prompt Evaluator Dashboard
===============================
A Streamlit-based tool for evaluating, comparing, and optimizing LLM prompts.

Features:
1. Individual Test — Test a single prompt with LLM Judge + feedback
2. Batch Evaluation — Bulk test 50+ prompts from CSV with category analysis
3. Prompt Comparison — Compare multiple prompts side-by-side and rank them
4. History — View past evaluations with trend charts
"""

import streamlit as st
import pandas as pd
import numpy as np
from src.llm import get_available_models, generate_response
from src.evaluator import evaluate_response
from src.optimizer import optimize_prompt
from src.utils import init_db, save_to_db, save_results_csv, get_history, clear_history
from src.templates import get_template_names, get_template_by_index
from src.report import generate_pdf_report
import os

# =====================
# Setup
# =====================
DB_PATH = os.path.join(os.path.dirname(__file__), "db", "results.db")
CSV_PATH = os.path.join(os.path.dirname(__file__), "outputs", "results.csv")
init_db(DB_PATH)

st.set_page_config(page_title="LLM Prompt Evaluator", page_icon="🧪", layout="wide")

# Inject Global CSS
st.markdown("""
<style>
  section[data-testid="stSidebar"] { display: none !important; }
  section[data-testid="stSidebarNav"] { display: none !important; }
  .stApp { background-color: #0D1117; }
  .block-container { 
    padding-top: 3rem !important; 
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 100% !important; 
  }
  .stMetric { background: #161B22; border: 1px solid #21262D; 
               border-radius: 10px; padding: 16px; }
  .stButton>button { background: #21262D; color: #E6EDF3; 
                     border: 1px solid #30363D; border-radius: 8px; }
  .stButton>button:hover { background: #30363D; border-color: #4F8EF7; }
  .stTextArea textarea, .stTextInput input { 
    background: #161B22 !important; color: #E6EDF3 !important; 
    border: 1px solid #30363D !important; border-radius: 8px !important; }
  .stSelectbox div, .stMultiSelect div { 
    background: #161B22 !important; color: #E6EDF3 !important; }
  h1, h2, h3 { color: #E6EDF3 !important; }
  p, label { color: #8B949E !important; }

  /* Progress Bar Custom Coloring Hack */
  .score-bar-container { width: 100%; background-color: #30363D; border-radius: 8px; height: 16px; margin-top: 5px; }
  .score-bar-fill { height: 100%; border-radius: 8px; }
  .score-red { background-color: #EF4444; }
  .score-amber { background-color: #F59E0B; }
  .score-green { background-color: #10B981; }

  /* Navbar styling */
  .nav-brand { font-weight: bold; color: #E6EDF3; font-size: 28px; display: flex; align-items: center; padding-left: 10px; margin-top: 5px; }
  .nav-brand-dot { color: #4F8EF7; margin-right: 8px; font-size: 30px; }
  .nav-right { display: flex; align-items: center; justify-content: flex-end; gap: 15px; color: #8B949E; font-size: 14px; margin-top: 10px; padding-right: 10px; }
  .status-dot { height: 10px; width: 10px; border-radius: 50%; display: inline-block; margin-right: 5px; }
  .nav-badge { background: #21262D; border: 1px solid #30363D; padding: 2px 8px; border-radius: 12px; font-size: 12px; }
</style>
""", unsafe_allow_html=True)

def render_score_bar(score, label="Score"):
    color_class = "score-red"
    if score >= 71:
        color_class = "score-green"
    elif score >= 41:
        color_class = "score-amber"
    
    st.markdown(f"**{label}:** {score:.1f}/100")
    st.markdown(f'''
    <div class="score-bar-container">
        <div class="score-bar-fill {color_class}" style="width: {max(0, min(100, score))}%;"></div>
    </div>
    ''', unsafe_allow_html=True)

def render_settings():
    with st.expander("⚙️ Settings", expanded=False):
        c1, c2, c3 = st.columns(3)
        av_models = get_available_models()
        if not av_models:
            st.warning("No Ollama models found. Make sure Ollama is running.")
            s_model = "llama3"
        else:
            s_model = c1.selectbox("Select Model", av_models, key=f"g_model_select_{st.session_state['active_page']}")
        
        temp = c2.slider("Temperature", 0.0, 1.0, 0.7, 0.1, key=f"g_temp_select_{st.session_state['active_page']}")
        u_judge = c3.checkbox("🧑‍⚖️ Enable LLM-as-Judge", value=False, key=f"g_judge_select_{st.session_state['active_page']}")
        
        st.markdown("---")
        with st.expander("ℹ️ What is Temperature?"):
            st.markdown("""
            **Temperature** controls how "creative" vs "focused" the AI is:
            - **Low (0.0 – 0.3):** Very focused and deterministic. Best for factual tasks.
            - **Medium (0.4 – 0.7):** Balanced. Good for most tasks.
            - **High (0.8 – 1.0):** Creative and varied. Best for brainstorming.
            """)
        with st.expander("ℹ️ What is LLM-as-Judge?"):
            st.markdown("""
            When enabled, the AI itself reviews each response and scores it on Correctness, Completeness, and Clarity.
            """)
    return s_model, temp, u_judge

if "active_page" not in st.session_state:
    st.session_state["active_page"] = "Overview"

available_models = get_available_models()
ollama_status_color = "#10B981" if available_models else "#EF4444"
ollama_status_text = "Connected" if available_models else "Disconnected"
active_model = st.session_state.get(f"g_model_select_{st.session_state['active_page']}", "llama3" if available_models else "None")

st.markdown('<div style="background-color: #161B22; border-bottom: 1px solid #21262D; margin: -8px -2rem 20px -2rem; padding: 0 2rem;">', unsafe_allow_html=True)
nav_col1, nav_col2, nav_col3 = st.columns([2, 5, 2])
with nav_col1:
    st.markdown('<div class="nav-brand"><span class="nav-brand-dot">●</span> LLM Evaluator</div>', unsafe_allow_html=True)

pages = ["Overview", "Single Test", "Bulk Eval", "A/B Compare", "Model Compare", "History"]
with nav_col2:
    tab_cols = st.columns(len(pages))
    
    # Global explicit navbar button overrides
    st.markdown('''<style>
    div[data-testid="column"]:nth-child(2) .stButton > button {
        background-color: transparent !important;
        background: transparent !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        border-radius: 0 !important;
        color: #8B949E !important;
        font-size: 13px !important;
        font-weight: 400 !important;
        padding: 14px 16px !important;
        white-space: nowrap !important;
        box-shadow: none !important;
    }
    div[data-testid="column"]:nth-child(2) .stButton > button * {
        white-space: nowrap !important;
        font-size: 13px !important;
    }
    div[data-testid="column"]:nth-child(2) .stButton > button:hover {
        color: #E6EDF3 !important;
        background-color: transparent !important;
        border-bottom: 2px solid #4F8EF7 !important;
    }
    </style>''', unsafe_allow_html=True)

    for i, page in enumerate(pages):
        active_css = ""
        if st.session_state["active_page"] == page:
            active_css = '''
            color: #E6EDF3 !important;
            border-bottom: 2px solid #4F8EF7 !important;
            '''
            
        # Specific active overrides using nth-child targeting
        st.markdown(f'''<style>
        div[data-testid="column"]:nth-child(2) div[data-testid="column"]:nth-child({i+1}) .stButton > button {{
            {active_css}
        }}
        </style>''', unsafe_allow_html=True)
        
        if tab_cols[i].button(page, use_container_width=True, key=f"nav_{page}"):
            st.session_state["active_page"] = page
            st.rerun()

with nav_col3:
    st.markdown(f'''<div class="nav-right">
        <div><span class="status-dot" style="background-color: {ollama_status_color};"></span>Ollama {ollama_status_text}</div>
        <div class="nav-badge">{active_model}</div>
    </div>''', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

def get_halluc_rate(df):
    if df.empty: return 0.0
    if 'semantic_similarity' in df.columns:
        high_risk_df = df[df['semantic_similarity'] < 0.35]
        return (len(high_risk_df) / len(df)) * 100.0
    return 0.0

if st.session_state["active_page"] == "Overview":
    from datetime import datetime
    
    history_arr = get_history(DB_PATH, limit=1000)
    hist_df = pd.DataFrame(history_arr)
    
    if not hist_df.empty:
        if 'timestamp' in hist_df.columns:
            hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp'])
            today_str = datetime.now().strftime("%Y-%m-%d")
            today_runs = hist_df[hist_df['timestamp'].dt.strftime("%Y-%m-%d") == today_str]
            today_count = len(today_runs)
            
            last_7 = hist_df[hist_df['timestamp'] >= (datetime.now() - pd.Timedelta(days=7))]
            old_avg = hist_df[hist_df['timestamp'] < (datetime.now() - pd.Timedelta(days=7))]['score'].mean()
            current_avg = hist_df['score'].mean()
            diff = current_avg - old_avg if not pd.isna(old_avg) else 0.0
        else:
            today_count = 0; diff = 0.0; current_avg = hist_df.get('score', pd.Series([0])).mean()
            today_runs = pd.DataFrame()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Tests", len(hist_df), f"+{today_count} today", delta_color="normal")
        m2.metric("Average Score", f"{current_avg:.1f}", f"{diff:+.1f} vs prev", delta_color="normal")
        
        halluc_rate = get_halluc_rate(hist_df)
        today_high_risk = len(today_runs[today_runs['semantic_similarity'] < 0.35]) if not today_runs.empty and 'semantic_similarity' in today_runs.columns else 0
        m3.metric("Hallucination Rate", f"{halluc_rate:.1f}%", f"{today_high_risk} HIGH risk today", delta_color="inverse")
        
        best_model_name = "None"
        best_model_avg = 0.0
        if 'model_name' in hist_df.columns:
            model_avgs = hist_df.groupby('model_name')['score'].mean()
            if not model_avgs.empty:
                best_model_name = model_avgs.idxmax()
                best_model_avg = model_avgs.max()
        m4.metric("Best Model", best_model_name, f"avg {best_model_avg:.1f} / 100", delta_color="normal")
        
        st.markdown("<br>", unsafe_allow_html=True)
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            st.markdown("### Category Performance")
            from src.utils import read_prompts_csv
            csv_df = read_prompts_csv(CSV_PATH)
            if not csv_df.empty and 'category' in csv_df.columns:
                cat_avgs = csv_df.groupby('category')['score'].mean()
                for cat, val in cat_avgs.items():
                    render_score_bar(val, cat)
            else:
                st.info("No CSV data yet. Run Bulk Eval to populate.")
                
        with r2c2:
            st.markdown("### Recent Evaluations")
            recent = hist_df.head(4)
            for _, r in recent.iterrows():
                with st.container(border=True):
                    val_sc = r.get('semantic_similarity', 1.0)
                    risk_str = "LOW"
                    risk_col = "#10B981"
                    if val_sc < 0.35:
                        risk_str = "HIGH"
                        risk_col = "#EF4444"
                    elif val_sc < 0.5:
                        risk_str = "MEDIUM"
                        risk_col = "#F59E0B"
                        
                    tc1, tc2, tc3 = st.columns([2, 5, 2])
                    tc1.markdown(f"**{r.get('model_name', 'Model')}**")
                    p_text = str(r.get('prompt', ''))
                    tc2.markdown(f"<span style='color: #8B949E'>{p_text[:30]}...</span>", unsafe_allow_html=True)
                    tc3.markdown(f"<span style='background: {risk_col}; padding: 2px 6px; border-radius: 10px; color: white; font-size: 10px;'>{risk_str} Risk</span> <b>{r.get('score', 0):.0f}</b>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        r3c1, r3c2 = st.columns([2, 1])
        with r3c1:
            st.markdown("### Score Trend")
            trend_df = hist_df.head(8)[::-1].reset_index(drop=True)
            if not trend_df.empty:
                st.bar_chart(trend_df['score'])
        with r3c2:
            st.markdown("### Model Leaderboard")
            if 'model_name' in hist_df.columns:
                leaderboard = hist_df.groupby('model_name')['score'].mean().sort_values(ascending=False)
                for i, (m, sc) in enumerate(leaderboard.items()):
                    with st.container(border=True):
                        win_badge = "🏆" if i == 0 else ""
                        st.markdown(f"**{m}** {win_badge} <br><span style='color: #8B949E'>{sc:.1f} avg</span>", unsafe_allow_html=True)
    else:
        st.info("No data available yet. Run a single test to begin logging history.")

elif st.session_state["active_page"] == "Single Test":
    selected_model, temperature, use_judge = render_settings()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input")

        # Template loader
        template_names = get_template_names()
        selected_template_idx = st.selectbox("📋 Load a Prompt Template", 
                                              range(len(template_names)),
                                              format_func=lambda i: template_names[i],
                                              key="template_selector")
        
        # Get values from template (user can still edit these)
        template = get_template_by_index(selected_template_idx)
        if template and 'prev_template_idx' not in st.session_state:
            st.session_state['prev_template_idx'] = 0
        
        # Only update fields when template actually changes
        if template and st.session_state.get('prev_template_idx') != selected_template_idx:
            st.session_state['prev_template_idx'] = selected_template_idx
            st.session_state['tpl_prompt'] = template["prompt"]
            st.session_state['tpl_expected'] = template["expected"]
        
        prompt = st.text_area("Enter your prompt:", height=150,
                              placeholder="E.g., Explain quantum computing to a 5-year old.",
                              key="single_prompt")
        expected_output = st.text_area("Expected Output (Optional):",
                                       height=150,
                                       placeholder="Provide the ideal response for evaluation.",
                                       key="single_expected")

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            run_eval = st.button("🚀 Generate & Evaluate", type="primary", use_container_width=True)
        with col_btn2:
            run_consistency = st.button("🔁 Consistency Test",
                                        help="Run the same prompt 5 times to check reliability", use_container_width=True)

        # --- Main Evaluation ---
        if run_eval:
            if not prompt:
                st.warning("Please enter a prompt.")
            else:
                with st.spinner(f"🤖 Querying LLM ({selected_model})..."):
                    llm_response = generate_response(prompt, model=selected_model, temperature=temperature)
                st.session_state['last_response'] = llm_response

                with st.spinner("🧑‍⚖️ Running evaluation metrics & LLM-as-Judge..."):
                    eval_results = evaluate_response(prompt, llm_response, expected_output,
                                                      model=selected_model, use_judge=use_judge)
                
                with st.spinner("🧠 Running Hallucination Analysis..."):
                    from src.hallucination import check_rules, compute_hallucination_scores, classify_hallucination_risk
                    rule_flags = check_rules(llm_response, prompt)
                    ovs_e_score, ovs_p_score = compute_hallucination_scores(llm_response, expected_output, prompt)
                    hallucination_res = classify_hallucination_risk(rule_flags, ovs_e_score, ovs_p_score)
                st.session_state['last_hallucination_eval'] = hallucination_res

                st.session_state['last_eval'] = eval_results
                st.session_state['current_prompt'] = prompt
                st.session_state['expected_output'] = expected_output

        # --- Consistency Test ---
        if run_consistency:
            if not prompt:
                st.warning("Please enter a prompt.")
            else:
                n_runs = 5
                consistency_scores = []
                consistency_responses = []

                progress = st.progress(0)
                for i in range(n_runs):
                    progress.progress((i + 1) / n_runs, text=f"Run {i+1}/{n_runs}...")
                    resp = generate_response(prompt, model=selected_model, temperature=temperature)
                    ev = evaluate_response(prompt, resp, expected_output,
                                           model=selected_model, use_judge=use_judge)
                    consistency_scores.append(ev['overall_score'])
                    consistency_responses.append(resp)

                st.session_state['consistency_results'] = {
                    'scores': consistency_scores,
                    'responses': consistency_responses,
                    'mean': float(np.mean(consistency_scores)),
                    'std': float(np.std(consistency_scores)),
                    'min': min(consistency_scores),
                    'max': max(consistency_scores),
                }

    with col2:
        st.subheader("Results")

        # --- Show evaluation results ---
        if 'last_response' in st.session_state:
            with st.container(border=True):
                st.markdown("### 🤖 LLM Output")
                st.code(st.session_state['last_response'], language="text")

            results = st.session_state['last_eval']
            st.markdown("### Evaluation Scores")

            # Metrics row
            has_judge = results.get('judge_score') is not None
            metric_cols = st.columns(4 if has_judge else 3)
            with metric_cols[0]:
                render_score_bar(results['overall_score'], "Overall Score")
            metric_cols[1].metric("Word Count", results['word_count'])
            if results['semantic_similarity'] is not None:
                metric_cols[2].metric("Semantic Similarity", f"{results['semantic_similarity']*100:.1f}%")
            if has_judge:
                metric_cols[3].metric("LLM Judge", f"{results['judge_score']:.1f}/10")

            # Hallucination Analysis
            if 'last_hallucination_eval' in st.session_state:
                with st.expander("🧠 Hallucination Analysis", expanded=True):
                    hr = st.session_state['last_hallucination_eval']
                    
                    if hr["risk_level"] == "HIGH":
                        st.markdown("### 🔴 HIGH RISK")
                    elif hr["risk_level"] == "MEDIUM":
                        st.markdown("### 🟡 MEDIUM RISK")
                    else:
                        st.markdown("### 🟢 LOW RISK")
                        
                    st.markdown(f"**{hr['summary']}**")
                    
                    st.markdown("---")
                    hc1, hc2 = st.columns(2)
                    with hc1:
                        render_score_bar(hr["embedding_scores"]["vs_expected"] * 100, "Semantic match vs Expected")
                    with hc2:
                        render_score_bar(hr["embedding_scores"]["vs_prompt"] * 100, "On-topic match vs Prompt")
                    
                    if hr["triggered_rules"]:
                        st.markdown("---")
                        st.markdown("**Triggered Rules:**")
                        for rule in hr["triggered_rules"]:
                            st.markdown(f"- {rule}")

            # Feedback
            if results.get('feedback'):
                st.markdown("### 💡 Feedback")
                st.warning(results['feedback'])

            # How it works
            with st.expander("ℹ️ How are scores calculated?"):
                st.markdown("""
                **Scoring Formula:**
                - **Semantic Similarity (40-60%)**: Measures how close the *meaning* of the response is to your expected output using AI embeddings.
                - **Length Penalty (20%)**: Penalizes responses that are too long or too short compared to expected.
                - **LLM Judge (40%)**: When enabled, an AI reviews the answer for correctness, completeness, and clarity.
                
                The Overall Score is a weighted average of all active components.
                """)

            # --- Optimizer with Before/After ---
            if results['overall_score'] < 90 and st.session_state.get('expected_output'):
                st.markdown("---")
                st.subheader("🔧 Prompt Optimizer")
                st.markdown("Score is below 90. Want AI to suggest a better prompt?")
                if st.button("Optimize & Compare", key="optimize_btn"):
                    with st.spinner("Optimizing prompt..."):
                        improved, new_response, new_eval, did_improve = optimize_prompt(
                            st.session_state['current_prompt'],
                            st.session_state['expected_output'],
                            results['overall_score'],
                            model=selected_model,
                            use_judge=use_judge
                        )

                    # Show before vs after
                    if did_improve:
                        st.success("✅ Optimization Complete! Mathematical improvement guaranteed.")
                    else:
                        st.warning("⚠️ No strictly better prompt found after retries. Showing best attempt.")
                        
                    st.markdown("**Suggested Prompt:**")
                    st.code(improved, language="text")

                    st.markdown("### 📈 Before vs After")
                    ba1, ba2, ba3 = st.columns(3)
                    old_score = results['overall_score']
                    new_score = new_eval['overall_score']
                    delta = new_score - old_score
                    with ba1: render_score_bar(old_score, "Original")
                    with ba2: render_score_bar(new_score, "Optimized")
                    ba3.metric("Improvement", f"{delta:+.1f}", delta=f"{delta:+.1f}")

                    if new_eval.get('feedback'):
                        st.info(f"**New Feedback:** {new_eval['feedback']}")

            # --- Save Features ---
            st.markdown("---")
            st.subheader("💾 Save Results")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Save to Database", key="save_db_single"):
                    save_to_db(DB_PATH,
                               st.session_state['current_prompt'],
                               st.session_state.get('expected_output', ''),
                               st.session_state['last_response'],
                               selected_model,
                               results['overall_score'],
                               judge_score=results.get('judge_score'),
                               feedback=results.get('feedback'),
                               semantic_similarity=results.get('semantic_similarity'))
                    st.success("✅ Saved to Database!")
            with c2:
                if st.button("Export to CSV", key="save_csv_single"):
                    save_results_csv([{
                        "prompt": st.session_state['current_prompt'],
                        "expected_output": st.session_state.get('expected_output', ''),
                        "llm_output": st.session_state['last_response'],
                        "model_name": selected_model,
                        "score": results['overall_score']
                    }], CSV_PATH)
                    st.success("✅ Appended to CSV!")

            # --- Export PDF Feature ---
            st.markdown("---")
            if st.button("📄 Export PDF Report", key="gen_pdf_single", use_container_width=True):
                with st.spinner("📄 Compiling PDF..."):
                    from src.pdf_report import generate_pdf_report as generate_single_pdf_report
                    from datetime import datetime
                    
                    pdf_bytes = generate_single_pdf_report(
                        prompt=st.session_state['current_prompt'],
                        expected_output=st.session_state.get('expected_output', ''),
                        model_output=st.session_state['last_response'],
                        final_score=st.session_state['last_eval']['overall_score'],
                        hallucination_result=st.session_state.get('last_hallucination_eval', {})
                    )
                    filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                
                st.download_button("📥 Download PDF", data=pdf_bytes, file_name=filename, mime="application/pdf", use_container_width=True)
                st.success("✅ Report ready! Click to download.")

        # --- Show consistency results ---
        if 'consistency_results' in st.session_state:
            st.markdown("---")
            st.subheader("🔁 Consistency Test Results")
            cr = st.session_state['consistency_results']

            cc1, cc2, cc3, cc4 = st.columns(4)
            cc1.metric("Mean Score", f"{cr['mean']:.1f}")
            cc2.metric("Std Dev", f"{cr['std']:.1f}")
            cc3.metric("Min", f"{cr['min']:.1f}")
            cc4.metric("Max", f"{cr['max']:.1f}")

            # Stability indicator
            if cr['std'] < 5:
                st.success("✅ **Stable** — The model gives consistent results for this prompt.")
            elif cr['std'] < 15:
                st.warning("⚠️ **Moderate Variance** — Results vary somewhat. Consider a lower temperature.")
            else:
                st.error("❌ **Inconsistent** — Results vary wildly. Refine the prompt or lower the temperature.")

            # Show individual runs
            with st.expander("View all 5 runs"):
                for i, (score, resp) in enumerate(zip(cr['scores'], cr['responses'])):
                    st.markdown(f"**Run {i+1}** — Score: {score:.1f}")
                    st.text(resp[:200] + "..." if len(resp) > 200 else resp)
                    st.markdown("---")


# ========================================
# TAB 2: Batch Evaluation
# ========================================
elif st.session_state["active_page"] == "Bulk Eval":
    selected_model, temperature, use_judge = render_settings()
    st.subheader("📊 Bulk Evaluation")
    st.markdown("Run evaluations on the dataset in `data/prompts.csv`.")

    with st.expander("ℹ️ How does Batch Evaluation work?"):
        st.markdown("""
        1. **Loads** all prompts + expected outputs from `data/prompts.csv`
        2. **Sends** each prompt to your selected Ollama model
        3. **Scores** each response using Semantic Similarity + Length Penalty (+ LLM Judge if enabled)
        4. **Analyzes** results by category and identifies weak areas
        5. **Shows** results in a table with downloadable CSV and PDF report
        
        ⚠️ *With LLM Judge enabled, batch evaluation takes ~3-5 min for 50 prompts.*
        """)

    uploaded_file = st.file_uploader("📂 Upload a CSV file (Must contain 'prompt' and 'expected_output')", type=["csv"])
    DATA_FILE = os.path.join("data", "prompts.csv")
    
    if uploaded_file is not None:
        df_prompts = pd.read_csv(uploaded_file)
        st.write(f"Loaded **{len(df_prompts)}** prompts from uploaded file.")
    elif os.path.exists(DATA_FILE):
        df_prompts = pd.read_csv(DATA_FILE)
        st.write(f"Loaded **{len(df_prompts)}** prompts from default dataset `data/prompts.csv`.")
    else:
        df_prompts = None
        st.error("Dataset not found. Please upload a CSV file.")

    if df_prompts is not None:
        if 'prompt' not in df_prompts.columns or 'expected_output' not in df_prompts.columns:
            st.error("Uploaded CSV must contain 'prompt' and 'expected_output' columns.")
        else:
            st.dataframe(df_prompts.head())

            if st.button("▶️ Run Batch Test", type="primary"):
                batch_results = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                for idx, row in df_prompts.iterrows():
                    status_text.text(f"Processing ({idx+1}/{len(df_prompts)}): {str(row['prompt'])[:50]}...")

                    resp = generate_response(row['prompt'], model=selected_model, temperature=temperature)
                    eval_res = evaluate_response(row['prompt'], resp, row['expected_output'],
                                                  model=selected_model, use_judge=use_judge)

                    batch_results.append({
                        "Prompt": row['prompt'],
                        "Expected": row['expected_output'],
                        "Category": row.get('category', 'Uncategorized'),
                        "LLM Output": resp,
                        "Score": eval_res['overall_score'],
                        "Similarity": eval_res['semantic_similarity'],
                        "Judge": eval_res.get('judge_score'),
                        "Feedback": eval_res.get('feedback', ''),
                    })
                    progress_bar.progress((idx + 1) / len(df_prompts))

                status_text.text("✅ Batch Evaluation Complete!")
                res_df = pd.DataFrame(batch_results)
                st.session_state['batch_results'] = batch_results
                st.session_state['batch_df'] = res_df

            # Show results if available
            if 'batch_df' in st.session_state:
                res_df = st.session_state['batch_df']
                
                # --- Summary Statistics ---
                st.markdown("### 📈 Summary Statistics")
                s1, s2, s3, s4 = st.columns(4)
                avg = res_df['Score'].mean()
                good_count = len(res_df[res_df['Score'] >= 80])
                ok_count = len(res_df[(res_df['Score'] >= 60) & (res_df['Score'] < 80)])
                poor_count = len(res_df[res_df['Score'] < 60])
                
                with s1:
                    render_score_bar(avg, "Average Score")
                s2.metric("✅ Good (80+)", good_count)
                s3.metric("⚠️ OK (60-79)", ok_count)
                s4.metric("❌ Poor (<60)", poor_count)

                # --- Category Analytics ---
                if 'Category' in res_df.columns and not res_df['Category'].eq('Uncategorized').all():
                    st.markdown("### 🏷️ Category Performance")
                    cat_df = res_df.groupby('Category')['Score'].mean().reset_index()
                    st.bar_chart(cat_df, x='Category', y='Score')

                # --- Score Distribution Chart ---
                st.markdown("### Score Distribution")
                st.bar_chart(res_df['Score'])

                # --- Full Results Table ---
                st.markdown("### Detailed Results")
                st.dataframe(res_df)

                # --- Top 5 Best and Worst ---
                st.markdown("### 🏆 Top 5 Best Prompts")
                top5 = res_df.nlargest(5, 'Score')[['Prompt', 'Score']].reset_index(drop=True)
                top5.index = top5.index + 1
                st.dataframe(top5)

                st.markdown("### ⚠️ Top 5 Worst Prompts (Need Optimization)")
                worst5 = res_df.nsmallest(5, 'Score')[['Prompt', 'Score', 'Feedback']].reset_index(drop=True)
                worst5.index = worst5.index + 1
                st.dataframe(worst5)

                # --- Export buttons ---
                st.markdown("---")
                exp1, exp2 = st.columns(2)
                with exp1:
                    csv = res_df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download CSV", data=csv,
                                       file_name="batch_results.csv", mime="text/csv", use_container_width=True)
                with exp2:
                    if st.button("📄 Generate PDF Report", key="batch_pdf", use_container_width=True):
                        with st.spinner("📄 Compiling PDF Report..."):
                            pdf_bytes = generate_pdf_report(
                                st.session_state['batch_results'],
                                model_name=selected_model,
                                temperature=temperature
                            )
                        st.download_button("📥 Download PDF", data=pdf_bytes,
                                           file_name="evaluation_report.pdf", mime="application/pdf", use_container_width=True)

                # --- Batch Optimizer ---
                st.markdown("---")
                st.subheader("🔧 Optimize a Low-Scoring Prompt")
                row_idx = st.number_input("Select row # to optimize", min_value=0,
                                           max_value=len(res_df)-1, value=0, step=1)
                
                selected_row = st.session_state['batch_results'][row_idx]
                st.caption(f"**Prompt:** {selected_row['Prompt'][:100]}...")
                st.caption(f"**Current Score:** {selected_row['Score']:.1f}/100")
                
                if st.button("Optimize Selected Row", key="batch_optimize", use_container_width=True):
                    with st.spinner("🤖 Optimizing prompt internally (this might take a few moments)..."):
                        improved, new_resp, new_eval, did_improve = optimize_prompt(
                            selected_row['Prompt'], 
                            selected_row['Expected'],
                            selected_row['Score'], 
                            model=selected_model,
                            use_judge=use_judge
                        )

                    if did_improve:
                        st.success("✅ Optimization Complete! Guaranteed higher score.")
                    else:
                        st.warning("⚠️ Could not definitively beat the original score. Kept original/best attempt.")
                        
                    st.markdown("**Suggested Improved Prompt:**")
                    st.code(improved, language="text")
                    
                    bo1, bo2, bo3 = st.columns(3)
                    with bo1: render_score_bar(selected_row['Score'], "Original")
                    with bo2: render_score_bar(new_eval['overall_score'], "Optimized")
                    delta = new_eval['overall_score'] - selected_row['Score']
                    bo3.metric("Improvement", f"{delta:+.1f}", delta=f"{delta:+.1f}")

                # --- Auto-Optimize All Poor Prompts ---
                st.markdown("---")
                st.subheader("🚀 Auto-Optimize Failing Prompts")
                if poor_count > 0:
                    st.markdown(f"Found **{poor_count}** prompts scoring below 60. Click below to automatically optimize all of them.")
                    if st.button("Auto-Optimize All < 60", type="primary", use_container_width=True):
                        progress_auto = st.progress(0)
                        status_auto = st.empty()
                        poor_indices = res_df[res_df['Score'] < 60].index
                        
                        for i, idx in enumerate(poor_indices):
                            row = st.session_state['batch_results'][idx]
                            status_auto.text(f"Optimizing prompt {i+1}/{poor_count} ({row['Category']})...")
                            improved, new_resp, new_eval, did_improve = optimize_prompt(
                                row['Prompt'], row['Expected'], row['Score'], model=selected_model, use_judge=use_judge
                            )
                            
                            if did_improve:
                                # Update the results directly in session state
                                st.session_state['batch_results'][idx].update({
                                    'Prompt': improved,
                                    'LLM Output': new_resp,
                                    'Score': new_eval['overall_score'],
                                    'Similarity': new_eval['semantic_similarity'],
                                    'Judge': new_eval.get('judge_score'),
                                    'Feedback': new_eval.get('feedback', '')
                                })
                            progress_auto.progress((i + 1) / poor_count)
                        
                        st.session_state['batch_df'] = pd.DataFrame(st.session_state['batch_results'])
                        status_auto.success("✅ Auto-Optimization Complete!")
                        st.rerun() # Refresh with new scores
                else:
                    st.success("All prompts scored 60 or higher! Excellent!")


# ========================================
# TAB 3: Prompt Comparison
# ========================================
elif st.session_state["active_page"] == "A/B Compare":
    selected_model, temperature, use_judge = render_settings()
    st.subheader("⚖️ Multi-Prompt Comparison")
    st.markdown("Enter multiple prompts to test the same question. The system will **rank** them from best to worst.")

    with st.expander("ℹ️ How does Prompt Comparison work?"):
        st.markdown("""
        1. Enter **multiple prompts** (one per line) — these are different ways of asking the same question
        2. Enter the **expected output** you want to match
        3. Each prompt is sent to the model and scored
        4. Results are **ranked from best to worst**
        5. The winning prompt is highlighted — use it!
        """)

    compare_prompts = st.text_area(
        "Enter multiple prompts (one per line):",
        height=200,
        placeholder="Prompt version 1\nPrompt version 2\nPrompt version 3",
        key="compare_prompts"
    )
    compare_expected = st.text_area(
        "Expected Output:",
        height=100,
        placeholder="The ideal response you want to match.",
        key="compare_expected"
    )

    if st.button("🏆 Compare & Rank", type="primary", use_container_width=True):
        prompts = [p.strip() for p in compare_prompts.strip().split("\n") if p.strip()]
        if len(prompts) < 2:
            st.warning("Please enter at least 2 prompts (one per line).")
        elif not compare_expected:
            st.warning("Please provide an expected output.")
        else:
            comparison_results = []
            progress = st.progress(0)

            for i, p in enumerate(prompts):
                progress.progress((i + 1) / len(prompts), text=f"Testing prompt {i+1}/{len(prompts)}...")
                resp = generate_response(p, model=selected_model, temperature=temperature)
                ev = evaluate_response(p, resp, compare_expected,
                                        model=selected_model, use_judge=use_judge)
                comparison_results.append({
                    "Rank": 0,
                    "Prompt": p,
                    "LLM Output": resp[:200] + "..." if len(resp) > 200 else resp,
                    "Score": ev['overall_score'],
                    "Similarity": ev['semantic_similarity'],
                    "Judge": ev.get('judge_score'),
                    "Feedback": ev.get('feedback', ''),
                })

            # Sort by score (best first) and assign ranks
            comparison_results.sort(key=lambda x: x['Score'], reverse=True)
            for i, r in enumerate(comparison_results):
                r['Rank'] = i + 1

            st.session_state['comparison_results'] = comparison_results

    # Show comparison results
    if 'comparison_results' in st.session_state:
        comp = st.session_state['comparison_results']
        st.markdown("### 🏆 Ranking Results")

        # Highlight the winner
        winner = comp[0]
        st.success(f"**🥇 Best Prompt (Score: {winner['Score']:.1f}/100):** {winner['Prompt']}")

        if len(comp) >= 2:
            loser = comp[-1]
            st.error(f"**🥉 Worst Prompt (Score: {loser['Score']:.1f}/100):** {loser['Prompt']}")

        # Full table
        comp_df = pd.DataFrame(comp)
        st.dataframe(comp_df)

        # Show improvement potential
        if len(comp) >= 2:
            best = comp[0]['Score']
            worst = comp[-1]['Score']
            st.info(f"📈 The best prompt scored **{best - worst:.1f} points higher** than the worst. "
                    f"Prompt wording matters!")


# ========================================
# TAB 4: Model Comparison
# ========================================
elif st.session_state["active_page"] == "Model Compare":
    selected_model, temperature, use_judge = render_settings()
    st.subheader("⚖️ Multi-Model Comparison")
    st.markdown("Compare the performance of different Ollama models simultaneously.")
    
    with st.expander("ℹ️ How does Model Comparison work?"):
        st.markdown('''
        1. Write a **single prompt** and the **expected output**.
        2. Select the **models** you want to test from your local Ollama instance.
        3. The system runs them all and reports the absolute **WINNER** based on optimal composite scoring tied to response latency.
        ''')
        
    mcomp_col1, mcomp_col2 = st.columns([1, 2])
    
    with mcomp_col1:
        st.markdown("### Inputs")
        mcomp_prompt = st.text_area("Enter your prompt:", height=150, key="mcomp_prompt", placeholder="Explain quantum computing to a 5-year old.")
        mcomp_expected = st.text_area("Expected Output / Ground Truth:", height=150, key="mcomp_expected", placeholder="Provide the ideal response for embedding evaluation.")
        
        # Safe fallback matching 
        default_selections = []
        if "llama3" in available_models:
            default_selections.append("llama3")
        if "phi3:mini" in available_models:
            default_selections.append("phi3:mini")
            
        # Fallback to first two if empty
        if not default_selections and len(available_models) > 0:
            default_selections = available_models[:2]

        selected_compare_models = st.multiselect(
            "Select Models to Compare:",
            options=available_models,
            default=default_selections
        )
        
        run_mcomp = st.button("🚀 Run Comparison", type="primary", use_container_width=True)

    with mcomp_col2:
        st.markdown("### Results")
        if run_mcomp:
            if not mcomp_prompt:
                st.warning("Please enter a prompt.")
            elif not selected_compare_models or len(selected_compare_models) < 1:
                st.warning("Please select at least one model to compare.")
            else:
                from src.model_comparison import compare_models, get_winner
                import uuid
                
                with st.spinner(f"🤖 Comparing {len(selected_compare_models)} models..."):
                    comparison_results = compare_models(
                        mcomp_prompt, 
                        mcomp_expected, 
                        models=selected_compare_models,
                        use_judge=use_judge,
                        temperature=temperature
                    )
                
                st.session_state['mcomp_results'] = comparison_results
                st.session_state['mcomp_winner'] = get_winner(comparison_results)
                st.session_state['mcomp_run_id'] = str(uuid.uuid4())
                st.session_state['mcomp_prompt_saved'] = mcomp_prompt
                st.session_state['mcomp_expected_saved'] = mcomp_expected
                
        # Render Results strictly if available in session state
        if 'mcomp_results' in st.session_state and st.session_state['mcomp_results']:
            mcomp_results = st.session_state['mcomp_results']
            winner = st.session_state['mcomp_winner']
            
            # 1. 🏆 Winner Banner
            if winner:
                st.markdown(f'''
                <div style="background-color: #161B22; border: 1px solid #30363D; border-left: 4px solid #10B981; border-radius: 10px; padding: 20px; margin-bottom: 20px;">
                    <h2 style="margin-top: 0; color: #10B981;">🏆 Winner: {winner['model']}</h2>
                    <h1 style="margin: 0;">{winner['composite_score']:.1f}<span style="font-size: 20px; color: #8B949E;">/100</span></h1>
                    <p style="margin-bottom: 0; color: #8B949E;">Latency: {winner['latency_ms']:.0f} ms</p>
                </div>
                ''', unsafe_allow_html=True)
            
            # 2. 📊 Side-by-Side Comparison Table
            st.markdown("### 📊 Model Scorecards")
            cols = st.columns(len(selected_compare_models))
            
            # We map iterating to the lengths of columns, safe checking if model array changed
            for index, res in enumerate(mcomp_results):
                if index < len(cols):
                    with cols[index]:
                        with st.container(border=True):
                            st.markdown(f"#### {res['model']}")
                            if res["error"]:
                                st.error("Model Failed")
                                with st.expander("View Error Output"):
                                    st.code(res["output"], language="text")
                            else:
                                render_score_bar(res['composite_score'], "Composite Score")
                                st.progress(res['semantic_score'] / 100.0, text=f"Semantic Score ({res['semantic_score']:.1f}%)")
                                if use_judge:
                                    st.progress(max(0.0, min(1.0, res['judge_score'] / 100.0)), text=f"Judge Score ({res['judge_score']/10:.1f}/10)")
                                
                                st.markdown(f"**Length Penalty Proxy (Words)**: {res['length_penalty']}")
                                st.markdown(f"**Latency**: {res['latency_ms']:.0f} ms")
                                
                                with st.expander("View Full Output"):
                                    st.code(res["output"], language="text")
            
            # 3. 📈 Score Bar Chart
            st.markdown("### 📈 Model Performance Breakdown")
            chart_data = []
            for r in [res for res in mcomp_results if not res["error"]]:
                chart_data.append({"Model": r["model"], "Score Type": "Composite", "Score": r["composite_score"]})
                chart_data.append({"Model": r["model"], "Score Type": "Semantic", "Score": r["semantic_score"]})
                if use_judge:
                    chart_data.append({"Model": r["model"], "Score Type": "Judge", "Score": r["judge_score"]})
            
            if chart_data:
                import pandas as pd
                df_chart = pd.DataFrame(chart_data)
                import altair as alt
                chart = alt.Chart(df_chart).mark_bar().encode(
                    x=alt.X('Model:N', title=None),
                    y=alt.Y('Score:Q', scale=alt.Scale(domain=[0, 100])),
                    color='Score Type:N',
                    xOffset='Score Type:N'
                ).properties(height=300)
                st.altair_chart(chart, use_container_width=True)

            # 4. 💾 Save to History
            st.markdown("---")
            if st.button("💾 Save Results to History", use_container_width=True):
                run_id = st.session_state['mcomp_run_id']
                p_saved = st.session_state['mcomp_prompt_saved']
                e_saved = st.session_state['mcomp_expected_saved']
                
                for r in mcomp_results:
                    if not r["error"]:
                        save_to_db(
                            DB_PATH,
                            p_saved,
                            e_saved,
                            r["output"],
                            r["model"],
                            r["composite_score"],
                            judge_score=(r["judge_score"] / 10.0) if use_judge else None,
                            feedback=None,
                            semantic_similarity=r["semantic_score"] / 100.0,
                            comparison_run_id=run_id
                        )
                st.success("✅ Results seamlessly saved to History database!")


# ========================================
# TAB 5: History
# ========================================
elif st.session_state["active_page"] == "History":
    st.subheader("📜 Evaluation History")
    st.markdown("View all past evaluations saved to the database.")

    with st.expander("ℹ️ How does History work?"):
        st.markdown("""
        Every time you click **"Save to Database"** in the Individual Test tab, 
        your results are stored in a local SQLite database (`db/results.db`).
        
        This tab shows all past evaluations so you can:
        - Track how your prompts improve over time
        - See which models perform best
        - Export historical data as PDF reports
        """)

    history = get_history(DB_PATH, limit=200)

    if history:
        hist_df = pd.DataFrame(history)

        # Summary stats
        h1, h2, h3 = st.columns(3)
        h1.metric("Total Evaluations", len(hist_df))
        with h2: render_score_bar(hist_df['score'].mean(), "Average Score")
        with h3: render_score_bar(hist_df['score'].max(), "Best Score")

        # Score trend chart
        st.markdown("### Score Trend Over Time")
        if 'timestamp' in hist_df.columns:
            chart_df = hist_df[['timestamp', 'score']].copy()
            chart_df['timestamp'] = pd.to_datetime(chart_df['timestamp'])
            chart_df = chart_df.sort_values('timestamp')
            chart_df = chart_df.set_index('timestamp')
            st.line_chart(chart_df['score'])

        # Full table
        st.markdown("### All Results")
        display_cols = ['id', 'prompt', 'model_name', 'score', 'judge_score', 'timestamp']
        available_cols = [c for c in display_cols if c in hist_df.columns]
        st.dataframe(hist_df[available_cols])

        # Export and clear buttons
        hc1, hc2, hc3 = st.columns(3)
        with hc1:
            csv = hist_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download History CSV", data=csv,
                               file_name="history_export.csv", mime="text/csv")
        with hc2:
            if st.button("📄 Generate PDF Report", key="history_pdf"):
                pdf_results = []
                for _, row in hist_df.iterrows():
                    pdf_results.append({
                        "Prompt": row.get('prompt', ''),
                        "Expected": row.get('expected_output', ''),
                        "LLM Output": row.get('llm_output', ''),
                        "Score": row.get('score', 0),
                        "Similarity": row.get('semantic_similarity'),
                        "Judge": row.get('judge_score'),
                        "Feedback": row.get('feedback', ''),
                    })
                with st.spinner("📄 Generating PDF..."):
                    pdf_bytes = generate_pdf_report(pdf_results, model_name="Mixed", temperature=0)
                st.download_button("📥 Download PDF", data=pdf_bytes,
                                   file_name="history_report.pdf", mime="application/pdf", use_container_width=True)
        with hc3:
            if st.button("🗑️ Clear All History", key="clear_history", use_container_width=True):
                clear_history(DB_PATH)
                st.success("History cleared!")
                st.rerun()
    else:
        st.info("No evaluation history found. Save some results from the Individual Test tab to see them here.")
