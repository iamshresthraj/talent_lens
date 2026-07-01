import os
import sys
import json
import yaml
import tempfile
import pandas as pd
import numpy as np
import gradio as gr
from datetime import datetime

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from src.text_build import build_candidate_doc
from src.rules import (
    detect_honeypot,
    check_pure_research_no_production,
    check_recent_langchain_only_no_legacy_ml,
    check_senior_no_hands_on_code_18mo,
    check_consulting_only_career,
    check_cv_speech_robotics_only_no_nlp,
    check_title_chaser_job_hopper,
    check_closed_source_only_no_external_validation,
    check_location_fit,
    check_notice_period_fit,
    check_salary_sanity_fit,
    check_experience_band_score,
    check_production_evidence_score
)
from src.scoring import (
    compute_skill_depth_fit_raw,
    compute_behavioral_multiplier,
    min_max_normalize
)
from src.reasoning import generate_reasoning


# Generate sandbox sample candidates from the main dataset if not present
def ensure_sample_candidates():
    sample_path = os.path.join(base_dir, "sandbox", "sample_candidates.json")
    if os.path.exists(sample_path):
        return sample_path
        
    os.makedirs(os.path.dirname(sample_path), exist_ok=True)
    candidates_path = os.path.join(base_dir, "data", "candidates.jsonl")
    
    if not os.path.exists(candidates_path):
        # Fallback empty list if dataset missing (should not happen in workspace)
        with open(sample_path, "w") as f:
            json.dump([], f)
        return sample_path
        
    print("Generating sandbox/sample_candidates.json from first 100 profiles...")
    samples = []
    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(samples) >= 100:
                break
            line = line.strip()
            if line:
                samples.append(json.loads(line))
                
    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)
    return sample_path


def run_sandbox_ranking(file_obj, semantic_w, skill_w, lexical_w, struct_w, logistics_w, reasoning_mode):
    # Load candidate lists
    if file_obj is None:
        sample_path = ensure_sample_candidates()
        with open(sample_path, "r", encoding="utf-8") as f:
            candidates = json.load(f)
    else:
        try:
            # Check if JSONL or JSON
            content = file_obj.name
            candidates = []
            with open(content, "r", encoding="utf-8") as f:
                first_char = f.read(1)
                f.seek(0)
                if first_char == '[':
                    candidates = json.load(f)
                else:
                    for line in f:
                        line = line.strip()
                        if line:
                            candidates.append(json.loads(line))
        except Exception as e:
            return f"Error loading candidate file: {e}", None
            
    if not candidates:
        return "No candidates found to process.", None
        
    # Cap processing size for sandbox performance safety
    candidates = candidates[:100]
    
    # Load JD Config
    jd_config_path = os.path.join(base_dir, "config", "jd_requirements.yaml")
    with open(jd_config_path, "r", encoding="utf-8") as f:
        jd_config = yaml.safe_load(f)
        
    must_have_skills = jd_config.get("must_have_skills", {})
    consulting_firms = jd_config.get("consulting_firms", [])
    pref_locations = jd_config.get("preferred_locations", [])
    pref_country = jd_config.get("preferred_country", "india")
    exp_band = jd_config.get("experience_band", {})
    ideal_notice_days = jd_config.get("notice_period_ideal_days", 30)
    
    # 1. Compute text docs and fit TF-IDF on the fly
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts = [build_candidate_doc(c) for c in candidates]
    
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        token_pattern=r"(?u)\b[\w\-\#\+\.]+\b",
        norm="l2"
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Lexical Query
    query_terms = []
    for s_def in must_have_skills.values():
        query_terms.extend(s_def.get("terms", []))
    for s_def in jd_config.get("nice_to_have_skills", {}).values():
        query_terms.extend(s_def.get("terms", []))
    query_text = " ".join(query_terms)
    
    query_vec = vectorizer.transform([query_text])
    lexical_raw = tfidf_matrix.dot(query_vec.T).toarray().ravel()
    lexical_fit = min_max_normalize(lexical_raw)
    
    # 2. Compute embeddings on the fly
    from sentence_transformers import SentenceTransformer
    model_path = os.path.join(base_dir, "models", "all-MiniLM-L6-v2")
    if os.path.exists(model_path):
        model = SentenceTransformer(model_path)
    else:
        # Fallback to online download in case sandbox is deployed on HF Spaces directly
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
    embeddings = model.encode(texts, convert_to_numpy=True)
    jd_vector = model.encode(jd_config.get("ideal_candidate_text", ""), convert_to_numpy=True)
    
    # Cosine Similarity
    dot_products = np.dot(embeddings, jd_vector)
    emb_norms = np.linalg.norm(embeddings, axis=1)
    jd_norm = np.linalg.norm(jd_vector)
    emb_norms[emb_norms == 0] = 1.0
    if jd_norm == 0: jd_norm = 1.0
    
    similarities = dot_products / (emb_norms * jd_norm)
    semantic_fit = min_max_normalize(similarities)
    
    # 3. Compute skill depth and rules
    skill_depth_raw = np.array([compute_skill_depth_fit_raw(c, must_have_skills) for c in candidates])
    skill_depth_fit = min_max_normalize(skill_depth_raw)
    
    # 4. Extract rule features on the fly
    features_list = []
    for c in candidates:
        honeypot = detect_honeypot(c)
        hard_disq = check_pure_research_no_production(c)
        
        s_langchain = check_recent_langchain_only_no_legacy_ml(c)
        s_senior_nocode = check_senior_no_hands_on_code_18mo(c)
        s_consulting = check_consulting_only_career(c, consulting_firms)
        s_cv_robotics = check_cv_speech_robotics_only_no_nlp(c)
        s_hopper = check_title_chaser_job_hopper(c)
        s_closed_source = check_closed_source_only_no_external_validation(c)
        
        loc_fit = check_location_fit(c, pref_locations, pref_country)
        notice_fit = check_notice_period_fit(c, ideal_notice_days)
        salary_fit = check_salary_sanity_fit(c)
        
        exp_score = check_experience_band_score(c, exp_band)
        prod_score = check_production_evidence_score(c)
        
        features_list.append({
            "is_honeypot": int(honeypot),
            "hard_disqualified": int(hard_disq),
            "soft_disq_recent_langchain": int(s_langchain),
            "soft_disq_senior_no_code": int(s_senior_nocode),
            "soft_disq_consulting_only": int(s_consulting),
            "soft_disq_cv_speech_robotics": int(s_cv_robotics),
            "soft_disq_title_chaser": int(s_hopper),
            "soft_disq_closed_source": int(s_closed_source),
            "location_fit": float(loc_fit),
            "notice_period_fit": float(notice_fit),
            "salary_sanity_fit": float(salary_fit),
            "experience_band_score": float(exp_score),
            "production_evidence_score": float(prod_score)
        })
        
    features_df = pd.DataFrame(features_list)
    
    # Structural fit
    exp_score = features_df["experience_band_score"].values
    prod_score = features_df["production_evidence_score"].values
    soft_penalties = (
        features_df["soft_disq_recent_langchain"].values * 0.15 +
        features_df["soft_disq_senior_no_code"].values * 0.10 +
        features_df["soft_disq_consulting_only"].values * 0.20 +
        features_df["soft_disq_cv_speech_robotics"].values * 0.25 +
        features_df["soft_disq_title_chaser"].values * 0.20 +
        features_df["soft_disq_closed_source"].values * 0.10
    )
    struct_raw = exp_score * 0.5 + prod_score * 0.5 - soft_penalties
    struct_raw = np.clip(struct_raw, 0.0, 1.0)
    structural_fit = min_max_normalize(struct_raw)
    
    # Logistics fit
    loc_fit = features_df["location_fit"].values
    notice_fit = features_df["notice_period_fit"].values
    sal_fit = features_df["salary_sanity_fit"].values
    logistics_raw = loc_fit * 0.5 + notice_fit * 0.3 + sal_fit * 0.2
    logistics_fit = min_max_normalize(logistics_raw)
    
    # 5. Linear score assembly
    linear_score = (
        semantic_w * semantic_fit +
        skill_w * skill_depth_fit +
        lexical_w * lexical_fit +
        struct_w * structural_fit +
        logistics_w * logistics_fit
    )
    
    # 6. Behavioral multipliers
    behavioral_mults = np.array([compute_behavioral_multiplier(c) for c in candidates])
    
    # 7. Soft disqualifier multipliers
    soft_disq_mult = np.ones(len(candidates))
    soft_disq_mult[features_df["soft_disq_recent_langchain"] == 1] *= 0.25
    soft_disq_mult[features_df["soft_disq_senior_no_code"] == 1] *= 0.30
    soft_disq_mult[features_df["soft_disq_consulting_only"] == 1] *= 0.20
    soft_disq_mult[features_df["soft_disq_cv_speech_robotics"] == 1] *= 0.15
    soft_disq_mult[features_df["soft_disq_title_chaser"] == 1] *= 0.20
    soft_disq_mult[features_df["soft_disq_closed_source"] == 1] *= 0.30
    
    final_scores = linear_score * behavioral_mults * soft_disq_mult
    
    # 8. Filter Honeypots & Hard Disqualifiers
    is_honeypot = features_df["is_honeypot"].values
    hard_disq = features_df["hard_disqualified"].values
    
    eligible_indices = np.where((is_honeypot == 0) & (hard_disq == 0))[0]
    
    # Sort
    sort_list = []
    for idx in eligible_indices:
        sort_list.append({
            "idx": idx,
            "id": candidates[idx]["candidate_id"],
            "score": final_scores[idx]
        })
    sort_list.sort(key=lambda x: (-x["score"], x["id"]))
    
    # Format top 100 (or max length of slice)
    results = sort_list[:100]
    
    table_rows = []
    csv_rows = []
    
    def extract_top_skills(cand):
        matched = []
        cand_skills = cand.get("skills", [])
        for cat_name, cat_def in must_have_skills.items():
            terms = cat_def.get("terms", [])
            for skill in cand_skills:
                name = skill.get("name", "").lower()
                if any(t.lower() in name for t in terms):
                    matched.append((skill.get("name"), skill.get("proficiency", "intermediate")))
                    break
        return matched[:3]
        
    def get_concern_sandbox(cand, row):
        if row["soft_disq_recent_langchain"] == 1:
            return "recent experience is focused only on LangChain/OpenAI wrappers"
        if row["soft_disq_senior_no_code"] == 1:
            return "moved to leadership and has not coded in past 18 months"
        if row["soft_disq_consulting_only"] == 1:
            return "career is entirely in IT consulting services"
        if row["soft_disq_cv_speech_robotics"] == 1:
            return "background is primarily computer vision/speech/robotics without NLP"
        if row["soft_disq_title_chaser"] == 1:
            return "frequent job changes with rapid title growth"
        if row["soft_disq_closed_source"] == 1:
            return "experience is entirely on closed-source systems without public validation"
        
        signals = cand.get("redrob_signals", {})
        days = signals.get("notice_period_days", 0)
        if days > 30:
            return f"notice period is {days} days"
        return None
        
    def get_recency_text_sandbox(cand):
        signals = cand.get("redrob_signals", {})
        last_active = signals.get("last_active_date", "")
        if not last_active: return "inactive recently"
        try:
            curr_date = datetime.strptime("2026-06-17", "%Y-%m-%d").date()
            act_date = datetime.strptime(last_active, "%Y-%m-%d").date()
            days = (curr_date - act_date).days
            if days <= 7: return "active this week"
            elif days <= 30: return "active this month"
            else: return "active recently"
        except Exception:
            return "active recently"
            
    # Generate reasoning for results
    for rank_idx, item in enumerate(results, start=1):
        idx = item["idx"]
        cand = candidates[idx]
        profile = cand.get("profile", {})
        row_feat = features_df.iloc[idx]
        
        facts = {
            "current_title": profile.get("current_title"),
            "current_company": profile.get("current_company"),
            "years_of_experience": profile.get("years_of_experience"),
            "top_skills": extract_top_skills(cand),
            "concern": get_concern_sandbox(cand, row_feat),
            "recency_text": get_recency_text_sandbox(cand)
        }
        
        reasoning = generate_reasoning(cand, facts, mode=reasoning_mode)
        
        table_rows.append([
            rank_idx,
            cand["candidate_id"],
            profile.get("anonymized_name", "Anonymous"),
            profile.get("current_title", "N/A"),
            profile.get("years_of_experience", 0),
            profile.get("location", "N/A"),
            round(item["score"], 4),
            reasoning
        ])
        
        csv_rows.append({
            "candidate_id": cand["candidate_id"],
            "rank": rank_idx,
            "score": round(item["score"], 6),
            "reasoning": reasoning
        })
        
    df_results = pd.DataFrame(table_rows, columns=["Rank", "Candidate ID", "Name", "Current Title", "Experience (Yrs)", "Location", "Score", "Reasoning"])
    
    # Save CSV to temp file for download
    temp_dir = tempfile.gettempdir()
    csv_path = os.path.join(temp_dir, "sandbox_submission.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    
    summary_text = (
        f"### Ranking Run Complete!\n"
        f"- **Total Uploaded**: {len(candidates)}\n"
        f"- **Filtered Honeypots**: {sum(is_honeypot)}\n"
        f"- **Filtered Hard Disqualified**: {sum(hard_disq)}\n"
        f"- **Eligible**: {len(eligible_indices)}"
    )
    
    return summary_text, df_results, csv_path


# Build Gradio UI
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    font=[gr.themes.GoogleFont("Outfit"), "sans-serif"]
)

with gr.Blocks(theme=theme, title="Redrob AI Intelligent Candidate Ranker", css="footer {visibility: hidden}") as demo:
    gr.Markdown(
        """
        # 🌌 Redrob AI Candidate Discovery & Ranking Sandbox
        ### Founding Senior AI Engineer Role — Noida/Pune Hybrid
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🛠️ Configuration & Inputs")
            
            file_input = gr.File(
                label="Upload Candidates (JSON / JSONL)",
                file_types=[".json", ".jsonl"],
                type="filepath"
            )
            gr.Markdown("*Leave empty to use the bundled sample of 100 profiles from the dataset.*")
            
            with gr.Accordion("⚖️ Weight Calibration", open=False):
                w_semantic = gr.Slider(0.0, 1.0, value=0.28, step=0.01, label="Semantic Vector Fit Weight")
                w_skills = gr.Slider(0.0, 1.0, value=0.20, step=0.01, label="Skill Depth Weight")
                w_lexical = gr.Slider(0.0, 1.0, value=0.12, step=0.01, label="Lexical TF-IDF Weight")
                w_struct = gr.Slider(0.0, 1.0, value=0.30, step=0.01, label="Structural / Penalty Weight")
                w_logistics = gr.Slider(0.0, 1.0, value=0.10, step=0.01, label="Logistics Weight")
                
            reasoning_backend = gr.Radio(
                choices=["template", "llm"],
                value="template",
                label="📝 Reasoning Generation Backend",
                info="LLM mode uses a local, CPU-based Qwen-0.5B model. Template mode is deterministic and fast."
            )
            
            btn_run = gr.Button("🚀 Rank Candidates", variant="primary")
            
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Ranking Results")
            status_output = gr.Markdown("Click **Rank Candidates** to run the discovery pipeline.")
            
            csv_output = gr.File(label="📥 Download Ranked CSV Submission")
            
            table_output = gr.DataFrame(
                headers=["Rank", "Candidate ID", "Name", "Current Title", "Experience (Yrs)", "Location", "Score", "Reasoning"],
                datatype=["number", "str", "str", "str", "number", "str", "number", "str"],
                wrap=True
            )
            
    btn_run.click(
        fn=run_sandbox_ranking,
        inputs=[file_input, w_semantic, w_skills, w_lexical, w_struct, w_logistics, reasoning_backend],
        outputs=[status_output, table_output, csv_output]
    )

if __name__ == "__main__":
    ensure_sample_candidates()
    demo.launch(server_name="127.0.0.1", server_port=7860)
