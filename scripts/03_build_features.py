import os
import json
import sys
import yaml
import pandas as pd

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates_path = os.path.join(base_dir, "data", "candidates.jsonl")
    jd_config_path = os.path.join(base_dir, "config", "jd_requirements.yaml")
    artifacts_dir = os.path.join(base_dir, "artifacts")
    
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Import source modules
    sys.path.append(base_dir)
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
    
    # Load JD Config
    print(f"Loading JD requirements from {jd_config_path}...")
    with open(jd_config_path, "r", encoding="utf-8") as f:
        jd_config = yaml.safe_load(f)
        
    consulting_firms = jd_config.get("consulting_firms", [])
    pref_locations = jd_config.get("preferred_locations", [])
    pref_country = jd_config.get("preferred_country", "india")
    exp_band = jd_config.get("experience_band", {})
    ideal_notice_days = jd_config.get("notice_period_ideal_days", 30)
    
    print(f"Processing candidate features from {candidates_path}...")
    features_list = []
    
    count = 0
    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                cand = json.loads(line)
                c_id = cand["candidate_id"]
                signals = cand.get("redrob_signals", {})
                
                # Check rules
                honeypot = detect_honeypot(cand)
                hard_disq = check_pure_research_no_production(cand)
                
                # Soft disqualifiers
                s_langchain = check_recent_langchain_only_no_legacy_ml(cand)
                s_senior_nocode = check_senior_no_hands_on_code_18mo(cand)
                s_consulting = check_consulting_only_career(cand, consulting_firms)
                s_cv_robotics = check_cv_speech_robotics_only_no_nlp(cand)
                s_hopper = check_title_chaser_job_hopper(cand)
                s_closed_source = check_closed_source_only_no_external_validation(cand)
                
                # Logistics fits
                loc_fit = check_location_fit(cand, pref_locations, pref_country)
                notice_fit = check_notice_period_fit(cand, ideal_notice_days)
                salary_fit = check_salary_sanity_fit(cand)
                
                # Structural sub-features
                exp_score = check_experience_band_score(cand, exp_band)
                prod_score = check_production_evidence_score(cand)
                
                # Append to list
                features_list.append({
                    "candidate_id": c_id,
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
                    "production_evidence_score": float(prod_score),
                    "open_to_work_flag": int(signals.get("open_to_work_flag", False)),
                    "recruiter_response_rate": float(signals.get("recruiter_response_rate", 0.0)),
                    "interview_completion_rate": float(signals.get("interview_completion_rate", 0.0)),
                    "offer_acceptance_rate": float(signals.get("offer_acceptance_rate", -1.0)),
                    "profile_completeness_score": float(signals.get("profile_completeness_score", 0.0)),
                    "verified_email": int(signals.get("verified_email", False)),
                    "verified_phone": int(signals.get("verified_phone", False)),
                    "linkedin_connected": int(signals.get("linkedin_connected", False)),
                    "github_activity_score": float(signals.get("github_activity_score", -1.0)),
                    "last_active_date": signals.get("last_active_date", "")
                })
                
                count += 1
                if count % 20000 == 0:
                    print(f"Processed features for {count} candidates...")
            except Exception as e:
                print(f"Error processing candidate features on line {count}: {e}")
                
    print(f"Finished processing {len(features_list)} candidates.")
    
    # Save to Parquet
    df = pd.DataFrame(features_list)
    parquet_file = os.path.join(artifacts_dir, "features.parquet")
    print(f"Saving features dataframe (shape: {df.shape}) to {parquet_file}...")
    df.to_parquet(parquet_file, index=False)
    print("Features precomputation completed successfully.")

if __name__ == "__main__":
    main()
