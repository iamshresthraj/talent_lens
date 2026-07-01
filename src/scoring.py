import math
import numpy as np
from datetime import datetime

def compute_skill_depth_fit_raw(candidate: dict, must_have_skills: dict) -> float:
    """
    Calculate raw skill depth score for a single candidate based on must-have skills,
    proficiency, duration, and endorsements.
    Formula: proficiency_weight * log1p(duration_months) * (1.1 if endorsements > 0 else 1.0)
    """
    cand_skills = candidate.get("skills", [])
    if not cand_skills:
        return 0.0
        
    prof_map = {
        "beginner": 0.25,
        "intermediate": 0.5,
        "advanced": 0.75,
        "expert": 1.0
    }
    
    total_weight = 0.0
    weighted_score_sum = 0.0
    
    for skill_name, skill_def in must_have_skills.items():
        weight = skill_def.get("weight", 1.0)
        terms = skill_def.get("terms", [])
        total_weight += weight
        
        best_skill_score = 0.0
        for cand_skill in cand_skills:
            cand_skill_name = cand_skill.get("name", "").lower()
            
            # Substring match against terms
            is_match = False
            for term in terms:
                if term.lower() in cand_skill_name:
                    is_match = True
                    break
                    
            if is_match:
                prof = cand_skill.get("proficiency", "beginner").lower()
                prof_score = prof_map.get(prof, 0.25)
                duration = cand_skill.get("duration_months", 0)
                endorsements = cand_skill.get("endorsements", 0)
                endorsement_bonus = 1.1 if endorsements > 0 else 1.0
                
                skill_score = prof_score * math.log1p(duration) * endorsement_bonus
                if skill_score > best_skill_score:
                    best_skill_score = skill_score
                    
        weighted_score_sum += best_skill_score * weight
        
    if total_weight > 0:
        return weighted_score_sum / total_weight
    return 0.0


def compute_behavioral_multiplier(candidate: dict, current_date_str: str = "2026-06-17") -> float:
    """
    Calculate behavioral multiplier in roughly [0.65, 1.25] based on:
    - recency of last_active_date
    - open_to_work_flag
    - recruiter_response_rate & interview_completion_rate
    - offer_acceptance_rate (neutral if -1)
    - profile completeness & contact verification
    - github activity (neutral if -1)
    """
    signals = candidate.get("redrob_signals", {})
    
    # 1. last_active_date recency decay
    last_active_str = signals.get("last_active_date", "")
    days = 365
    if last_active_str:
        try:
            curr_date = datetime.strptime(current_date_str, "%Y-%m-%d").date()
            act_date = datetime.strptime(last_active_str, "%Y-%m-%d").date()
            days = (curr_date - act_date).days
        except Exception:
            pass
            
    if days <= 14:
        recency_factor = 1.0
    else:
        # Decays to ~0.7 by 180 days
        recency_factor = 0.7 + 0.3 * (0.5 ** ((days - 14) / 180.0))
        
    # 2. open_to_work_flag (+0.05)
    open_to_work = signals.get("open_to_work_flag", False)
    open_work_bonus = 0.05 if open_to_work else 0.0
    
    # 3. recruiter response rate & interview completion rate (weight 0.1 each)
    resp_rate = signals.get("recruiter_response_rate", 0.0)
    int_rate = signals.get("interview_completion_rate", 0.0)
    
    # 4. offer acceptance rate (-1 is neutral 0.5)
    offer_rate = signals.get("offer_acceptance_rate", -1.0)
    offer_val = 0.5 if offer_rate == -1.0 else offer_rate
    
    # 5. profile completeness (up to +0.05)
    completeness = signals.get("profile_completeness_score", 0.0)
    completeness_bonus = (completeness / 100.0) * 0.05
    
    # 6. verification signals (+0.02 each, cap at 0.06)
    v_email = signals.get("verified_email", False)
    v_phone = signals.get("verified_phone", False)
    v_linkedin = signals.get("linkedin_connected", False)
    
    verification_bonus = 0.0
    if v_email: verification_bonus += 0.02
    if v_phone: verification_bonus += 0.02
    if v_linkedin: verification_bonus += 0.02
    verification_bonus = min(verification_bonus, 0.06)
    
    # 7. github activity (neutral if -1)
    github_score = signals.get("github_activity_score", -1.0)
    github_bonus = 0.0
    if github_score > 0:
        github_bonus = (github_score / 100.0) * 0.05
        
    raw_sum = (
        recency_factor +
        open_work_bonus +
        resp_rate * 0.10 +
        int_rate * 0.10 +
        offer_val * 0.10 +
        completeness_bonus +
        verification_bonus +
        github_bonus
    )
    
    # Map raw_sum in [0.7, 1.5] linearly to [0.65, 1.25]
    mult = 0.65 + 0.60 * ((raw_sum - 0.7) / (1.5 - 0.7))
    return max(0.65, min(1.25, mult))


def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    """
    Min-max normalize a numpy array to [0, 1].
    Safeguards against zero division when min == max.
    """
    if len(scores) == 0:
        return scores
    s_min = np.min(scores)
    s_max = np.max(scores)
    if s_max == s_min:
        return np.ones_like(scores)
    return (scores - s_min) / (s_max - s_min)
