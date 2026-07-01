import os
import sys
import json
import yaml
import tempfile
import re
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


def run_sandbox_ranking(file_obj, jd_text_input, jd_file_input, semantic_w, skill_w, lexical_w, struct_w, logistics_w, reasoning_mode):
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
            return f"Error loading candidate file: {e}", None, None
            
    if not candidates:
        return "No candidates found to process.", None, None
        
    # Cap processing size for sandbox performance safety
    candidates = candidates[:100]
    
    # Load default JD Config
    jd_config_path = os.path.join(base_dir, "config", "jd_requirements.yaml")
    with open(jd_config_path, "r", encoding="utf-8") as f:
        jd_config = yaml.safe_load(f)
        
    # Extract Job Description text
    jd_text = ""
    if jd_text_input and jd_text_input.strip():
        jd_text = jd_text_input.strip()
    elif jd_file_input is not None:
        try:
            if jd_file_input.name.endswith(".pdf"):
                import pypdf
                reader = pypdf.PdfReader(jd_file_input.name)
                for page in reader.pages:
                    jd_text += (page.extract_text() or "") + "\n"
            else:
                with open(jd_file_input.name, "r", encoding="utf-8") as f:
                    jd_text = f.read()
            jd_text = jd_text.strip()
        except Exception as e:
            return f"Error reading job description file: {e}", None, None
            
    if not jd_text:
        jd_text = jd_config.get("ideal_candidate_text", "")
        
    jd_lower = jd_text.lower()
    
    # 1. Parse Experience Band
    soft_min, soft_max = 5, 9
    match = re.search(r'(\d+)\s*(?:to|-)\s*(\d+)\s*years?', jd_lower)
    if match:
        soft_min = int(match.group(1))
        soft_max = int(match.group(2))
    else:
        match_plus = re.search(r'(\d+)\s*\+\s*years?', jd_lower)
        if match_plus:
            soft_min = int(match_plus.group(1))
            soft_max = soft_min + 4
            
    exp_band = {
        "soft_min": soft_min,
        "soft_max": soft_max,
        "hard_floor": max(1, soft_min - 2),
        "hard_ceiling": soft_max + 7
    }
    
    # 2. Parse Locations
    cities = ["pune", "noida", "hyderabad", "bangalore", "bengaluru", "mumbai", "delhi", "gurgaon", "gurugram", "ncr", "chennai"]
    pref_locations = [c for c in cities if c in jd_lower]
    if not pref_locations:
        pref_locations = jd_config.get("preferred_locations", ["pune", "noida"])
        
    # 3. Parse Skills
    common_skills = {
        "python": ["python"],
        "pytorch": ["pytorch"],
        "tensorflow": ["tensorflow", "keras"],
        "scikit-learn": ["scikit-learn", "sklearn"],
        "xgboost": ["xgboost", "lightgbm"],
        "pandas": ["pandas", "numpy"],
        "sql": ["sql", "postgresql", "mysql"],
        "elasticsearch": ["elasticsearch", "opensearch"],
        "pinecone": ["pinecone", "weaviate", "qdrant", "milvus", "faiss"],
        "embeddings": ["embeddings", "dense retrieval"],
        "llm": ["llm", "langchain", "llama", "gpt", "openai"],
        "nlp": ["nlp", "natural language", "bert", "transformers"]
    }
    
    must_have_skills = {}
    for skill_key, terms in common_skills.items():
        if any(term in jd_lower for term in terms):
            must_have_skills[skill_key] = {"weight": 1.0, "terms": terms}
            
    if not must_have_skills:
        must_have_skills = jd_config.get("must_have_skills", {
            "python": {"weight": 1.0, "terms": ["python"]}
        })
        
    consulting_firms = jd_config.get("consulting_firms", [])
    pref_country = jd_config.get("preferred_country", "india")
    ideal_notice_days = jd_config.get("notice_period_ideal_days", 30)
    
    # Compute text docs and fit TF-IDF on the fly
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
    
    # Compute embeddings on the fly
    from sentence_transformers import SentenceTransformer
    model_path = os.path.join(base_dir, "models", "all-MiniLM-L6-v2")
    if os.path.exists(model_path):
        model = SentenceTransformer(model_path)
    else:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
    embeddings = model.encode(texts, convert_to_numpy=True)
    jd_vector = model.encode(jd_text, convert_to_numpy=True)
    
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


# Helper functions for the custom ranker API
def extract_top_skills(cand, must_have_skills):
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


def run_custom_ranking_api(jd_text_input, candidates_json_str, w_sem, w_ski, w_lex, w_str, w_log, reasoning_backend):
    try:
        # Parse weights
        semantic_w = float(w_sem) if w_sem else 0.28
        skill_w = float(w_ski) if w_ski else 0.20
        lexical_w = float(w_lex) if w_lex else 0.12
        struct_w = float(w_str) if w_str else 0.30
        logistics_w = float(w_log) if w_log else 0.10
        
        # Load candidate lists
        if not candidates_json_str or candidates_json_str.strip() == "":
            sample_path = ensure_sample_candidates()
            with open(sample_path, "r", encoding="utf-8") as f:
                candidates = json.load(f)
        else:
            try:
                candidates = json.loads(candidates_json_str)
            except Exception:
                # Try parsing as JSONL
                candidates = []
                for line in candidates_json_str.strip().split("\n"):
                    line = line.strip()
                    if line:
                        candidates.append(json.loads(line))
                        
        if not candidates:
            return json.dumps({"error": "No candidates found to process."})
            
        # Cap processing size for sandbox performance safety
        candidates = candidates[:100]
        
        # Load default JD Config
        jd_config_path = os.path.join(base_dir, "config", "jd_requirements.yaml")
        with open(jd_config_path, "r", encoding="utf-8") as f:
            jd_config = yaml.safe_load(f)
            
        # Extract Job Description text
        jd_text = jd_text_input.strip() if jd_text_input and jd_text_input.strip() else jd_config.get("ideal_candidate_text", "")
        jd_lower = jd_text.lower()
        
        # 1. Parse Experience Band
        soft_min, soft_max = 5, 9
        match = re.search(r'(\d+)\s*(?:to|-)\s*(\d+)\s*years?', jd_lower)
        if match:
            soft_min = int(match.group(1))
            soft_max = int(match.group(2))
        else:
            match_plus = re.search(r'(\d+)\s*\+\s*years?', jd_lower)
            if match_plus:
                soft_min = int(match_plus.group(1))
                soft_max = soft_min + 4
                
        exp_band = {
            "soft_min": soft_min,
            "soft_max": soft_max,
            "hard_floor": max(1, soft_min - 2),
            "hard_ceiling": soft_max + 7
        }
        
        # 2. Parse Locations
        cities = ["pune", "noida", "hyderabad", "bangalore", "bengaluru", "mumbai", "delhi", "gurgaon", "gurugram", "ncr", "chennai"]
        pref_locations = [c for c in cities if c in jd_lower]
        if not pref_locations:
            pref_locations = jd_config.get("preferred_locations", ["pune", "noida"])
            
        # 3. Parse Skills
        common_skills = {
            "python": ["python"],
            "pytorch": ["pytorch"],
            "tensorflow": ["tensorflow", "keras"],
            "scikit-learn": ["scikit-learn", "sklearn"],
            "xgboost": ["xgboost", "lightgbm"],
            "pandas": ["pandas", "numpy"],
            "sql": ["sql", "postgresql", "mysql"],
            "elasticsearch": ["elasticsearch", "opensearch"],
            "pinecone": ["pinecone", "weaviate", "qdrant", "milvus", "faiss"],
            "embeddings": ["embeddings", "dense retrieval"],
            "llm": ["llm", "langchain", "llama", "gpt", "openai"],
            "nlp": ["nlp", "natural language", "bert", "transformers"]
        }
        
        must_have_skills = {}
        for skill_key, terms in common_skills.items():
            if any(term in jd_lower for term in terms):
                must_have_skills[skill_key] = {"weight": 1.0, "terms": terms}
                
        if not must_have_skills:
            must_have_skills = jd_config.get("must_have_skills", {
                "python": {"weight": 1.0, "terms": ["python"]}
            })
            
        consulting_firms = jd_config.get("consulting_firms", [])
        pref_country = jd_config.get("preferred_country", "india")
        ideal_notice_days = jd_config.get("notice_period_ideal_days", 30)
        
        # Compute text docs and fit TF-IDF on the fly
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
        
        # Compute embeddings on the fly
        from sentence_transformers import SentenceTransformer
        model_path = os.path.join(base_dir, "models", "all-MiniLM-L6-v2")
        if os.path.exists(model_path):
            model = SentenceTransformer(model_path)
        else:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            
        embeddings = model.encode(texts, convert_to_numpy=True)
        jd_vector = model.encode(jd_text, convert_to_numpy=True)
        
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
        
        results = sort_list[:100]
        
        out_candidates = []
        csv_rows = []
        
        for rank_idx, item in enumerate(results, start=1):
            idx = item["idx"]
            cand = candidates[idx]
            profile = cand.get("profile", {})
            row_feat = features_df.iloc[idx]
            
            top_skills_list = extract_top_skills(cand, must_have_skills)
            skills_ui = [{"name": s[0], "proficiency": s[1]} for s in top_skills_list]
            
            facts = {
                "current_title": profile.get("current_title"),
                "current_company": profile.get("current_company"),
                "years_of_experience": profile.get("years_of_experience"),
                "top_skills": top_skills_list,
                "concern": get_concern_sandbox(cand, row_feat),
                "recency_text": get_recency_text_sandbox(cand)
            }
            
            reasoning = generate_reasoning(cand, facts, mode=reasoning_backend)
            
            out_candidates.append({
                "rank": rank_idx,
                "id": cand["candidate_id"],
                "name": profile.get("anonymized_name", "Anonymous"),
                "title": profile.get("current_title", "N/A"),
                "experience": profile.get("years_of_experience", 0),
                "location": profile.get("location", "N/A"),
                "score": float(item["score"]),
                "reasoning": reasoning,
                "skills": skills_ui,
                "raw_json": cand
            })
            
            csv_rows.append({
                "candidate_id": cand["candidate_id"],
                "rank": rank_idx,
                "score": round(item["score"], 6),
                "reasoning": reasoning
            })
            
        df_csv = pd.DataFrame(csv_rows)
        csv_string = df_csv.to_csv(index=False)
        
        # Save CSV file directly to the workspace root for local convenience
        try:
            submission_csv_path = os.path.join(base_dir, "submission.csv")
            df_csv.to_csv(submission_csv_path, index=False)
            print(f"Saved ranked output directly to: {submission_csv_path}")
        except Exception as csv_err:
            print("Failed to save submission.csv to workspace root:", csv_err)

        response = {
            "summary": {
                "total": len(candidates),
                "eligible": len(eligible_indices),
                "honeypots": int(sum(is_honeypot)),
                "disqualified": int(sum(hard_disq))
            },
            "candidates": out_candidates,
            "csv_content": csv_string
        }
        return json.dumps(response)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Parse job_descriptions_final.md into structured JD presets
# ---------------------------------------------------------------------------
def parse_jd_markdown(filepath):
    """Parse the JD markdown file and extract each role as a preset."""
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found. Using empty preset list.")
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Split on ## headings that look like role titles (## N. Title or ## Title)
    # We skip headings like "About", "How to Apply", "Placeholder Reference"
    sections = re.split(r'^(## \d+\..*$)', content, flags=re.MULTILINE)

    presets = []
    i = 1  # sections[0] is everything before the first role heading
    while i < len(sections) - 1:
        heading = sections[i].strip()
        body = sections[i + 1].strip()

        # Extract the role title (remove "## N. " prefix)
        title_match = re.match(r'## \d+\.\s*(.+)', heading)
        if title_match:
            title = title_match.group(1).strip()
        else:
            title = heading.replace("## ", "").strip()

        # Extract a clean short description for the sidebar
        # Pull experience and a summary sentence from the body
        exp_text = ""
        for line in body.split("\n"):
            line_stripped = line.strip()
            if line_stripped.startswith("**Experience Required:**"):
                exp_text = line_stripped.replace("**Experience Required:**", "").strip()
                break

        # Find the first substantive paragraph after "Let's be honest"
        summary_sentence = ""
        in_honest_section = False
        for line in body.split("\n"):
            line_stripped = line.strip()
            if "let's be honest" in line_stripped.lower():
                in_honest_section = True
                continue
            if in_honest_section and line_stripped and not line_stripped.startswith("#"):
                # Clean markdown bold/italic
                clean = re.sub(r'\*\*|__|\*|_', '', line_stripped)
                # Skip very short lines
                if len(clean) > 30:
                    # Truncate to ~120 chars
                    if len(clean) > 120:
                        clean = clean[:117] + "..."
                    summary_sentence = clean
                    break

        if exp_text and summary_sentence:
            short_desc = f"{exp_text}. {summary_sentence}"
        elif summary_sentence:
            short_desc = summary_sentence
        elif exp_text:
            short_desc = f"{exp_text} experience required."
        else:
            short_desc = title

        # Full JD text = heading + body (for pasting into the chat)
        full_text = f"{heading}\n\n{body}"
        # Trim trailing --- divider if present
        full_text = re.sub(r'\n---\s*$', '', full_text).strip()

        presets.append({
            "title": title,
            "desc": short_desc,
            "text": full_text
        })

        i += 2

    return presets


jd_md_path = os.path.join(base_dir, "job_descriptions_final.md")
JD_PRESETS = parse_jd_markdown(jd_md_path)
print(f"Loaded {len(JD_PRESETS)} JD presets from job_descriptions_final.md")

# Read HTML contents for UI
ui_head_path = os.path.join(base_dir, "sandbox", "ui_head.html")
with open(ui_head_path, "r", encoding="utf-8") as f:
    head_content = f.read()

ui_body_path = os.path.join(base_dir, "sandbox", "ui_body.html")
with open(ui_body_path, "r", encoding="utf-8") as f:
    body_content = f.read()

# Inject the parsed JD presets JSON into the head template
jd_presets_json = json.dumps(JD_PRESETS, ensure_ascii=False)
head_content = head_content.replace(
    "/* __JD_PRESETS_PLACEHOLDER__ */",
    f"window.__JD_PRESETS__ = {jd_presets_json};"
)

# Build Custom Gradio UI
with gr.Blocks(title="Redrob AI Intelligent Candidate Ranker") as demo:
    # 1. Hidden inputs/outputs in a column styled as none
    with gr.Column(elem_id="gradio-bridge"):
        # Inputs
        jd_input = gr.Textbox(elem_id="jd-input", visible=True)
        candidates_input = gr.Textbox(elem_id="candidates-input", visible=True)
        w_sem_input = gr.Textbox(elem_id="w-sem-input", visible=True)
        w_ski_input = gr.Textbox(elem_id="w-ski-input", visible=True)
        w_lex_input = gr.Textbox(elem_id="w-lex-input", visible=True)
        w_str_input = gr.Textbox(elem_id="w-str-input", visible=True)
        w_log_input = gr.Textbox(elem_id="w-log-input", visible=True)
        backend_input = gr.Textbox(elem_id="backend-input", visible=True)
        
        # Trigger button
        run_btn = gr.Button(value="Run", elem_id="run-btn", visible=True)
        
        # Output
        results_output = gr.Textbox(elem_id="results-output", visible=True)

    # 2. Render the custom SPA body
    gr.HTML(body_content)

    # Bind click event
    run_btn.click(
        fn=run_custom_ranking_api,
        inputs=[jd_input, candidates_input, w_sem_input, w_ski_input, w_lex_input, w_str_input, w_log_input, backend_input],
        outputs=[results_output]
    )

if __name__ == "__main__":
    ensure_sample_candidates()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7863,
        head=head_content,
        css="""
            #gradio-bridge { display: none !important; }
            gradio-app, .gradio-container, .main, .wrap, footer { 
                margin: 0 !important; padding: 0 !important;
                max-height: 100vh !important; overflow: hidden !important;
            }
            footer, .footer, .built-with { display: none !important; }
        """
    )
