import math
import re
import numpy as np
from datetime import datetime

def match_skill_term(term: str, skill_name: str) -> bool:
    term_lower = term.lower()
    skill_lower = skill_name.lower()
    
    # Handle specific short terms with special characters
    if term_lower in ["c++", "c#", ".net"]:
        return term_lower in skill_lower
        
    # If the term is very short, require word boundary match to avoid false positives
    # (e.g. 'go' matching 'django', 'mongodb')
    if len(term_lower) <= 3:
        pattern = r'\b' + re.escape(term_lower) + r'\b'
        return bool(re.search(pattern, skill_lower))
        
    return term_lower in skill_lower

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
            
            # Word-boundary safe match against terms
            is_match = False
            for term in terms:
                if match_skill_term(term, cand_skill_name):
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


# ---------------------------------------------------------------------------
# JD-aware recruiter scoring (shared by rank.py and sandbox/app.py)
#
# Historically rank.py and both sandbox ranking paths each carried their own
# copy of the recruiter scoring loop with the must-have / good-to-have
# categories HARDCODED to the default Senior AI Engineer role. That meant the
# dominant score component (the must-have tier band) was identical for every
# job description, so every JD produced (nearly) the same ranked CSV. The
# functions below take the parsed JD's rubric as input so the ranking follows
# the actual JD.
# ---------------------------------------------------------------------------

try:
    from src.rules import (
        detect_honeypot,
        check_pure_research_no_production,
        check_irrelevant_role,
        check_experience_band_score,
        check_location_fit,
    )
except ImportError:  # pragma: no cover - direct module import fallback
    from rules import (
        detect_honeypot,
        check_pure_research_no_production,
        check_irrelevant_role,
        check_experience_band_score,
        check_location_fit,
    )

# Terms that denote a bare programming language. A must-have category made up
# solely of these gets the "listed skill + role-matching title" evidence
# shortcut (generalization of the old strong_python special case).
LANGUAGE_TERMS = {
    "python", "go", "golang", "java", "c#", ".net", "ruby", "node.js",
    "javascript", "typescript", "ts", "js", "swift", "kotlin", "objective-c",
    "scala", "php", "sql", "bash", "shell", "c++",
}

# Title fragments that make a candidate's current role plausible evidence for
# a language-only must-have category, per target role type.
ROLE_TITLE_HINTS = {
    "ml_ai": ["ml", "ai", "data", "search", "recommend", "nlp", "backend"],
    "backend": ["backend", "back-end", "software", "platform", "api", "sde", "developer", "engineer"],
    "frontend": ["frontend", "front-end", "ui", "web", "software", "developer", "engineer"],
    "fullstack": ["full stack", "full-stack", "fullstack", "software", "web", "developer", "engineer"],
    "devops": ["devops", "sre", "platform", "infrastructure", "cloud", "reliability", "systems"],
    "data_engineer": ["data", "etl", "analytics", "platform", "engineer"],
    "tpm": ["product", "program", "tpm", "manager"],
    "mobile": ["mobile", "ios", "android", "app", "developer", "engineer"],
}

# Generic technical-title fragments used to accept free-text evidence.
TECH_TITLE_HINTS = ["engineer", "scientist", "developer", "programmer", "architect", "analyst", "mle", "lead"]


def _category_terms(cat_def):
    """Accept either {'terms': [...]} dicts or plain term lists."""
    if isinstance(cat_def, dict):
        return cat_def.get("terms", [])
    return list(cat_def or [])


def check_must_have_matches(candidate: dict, must_have_skills: dict, role_type: str = "ml_ai") -> int:
    """
    Count how many of the JD's must-have skill categories the candidate
    satisfies with supporting evidence (career-history mention, >=12 months of
    use, endorsements, a role-matching title for language-only categories, or
    free-text evidence in a technical profile).
    """
    skills = candidate.get("skills", [])
    history = candidate.get("career_history", [])
    profile = candidate.get("profile", {})

    headline = profile.get("headline", "").lower()
    summary = profile.get("summary", "").lower()
    history_text = " ".join([
        (job.get("title", "") + " " + job.get("description", "")).lower()
        for job in history
    ])
    profile_text = f"{headline} {summary} {history_text}"
    current_title = profile.get("current_title", "").lower()
    role_hints = ROLE_TITLE_HINTS.get(role_type, TECH_TITLE_HINTS)

    matched_count = 0
    for cat, cat_def in must_have_skills.items():
        terms = _category_terms(cat_def)
        is_lang_cat = bool(terms) and all(t.lower() in LANGUAGE_TERMS for t in terms)

        has_skill_in_list = False
        for s in skills:
            name = s.get("name", "").lower()
            if any(match_skill_term(t, name) for t in terms):
                in_history = any(t in history_text for t in terms)
                long_duration = s.get("duration_months", 0) >= 12
                has_endorsements = s.get("endorsements", 0) > 0
                is_role_title = any(r in current_title for r in role_hints)

                if in_history or long_duration or has_endorsements or (is_lang_cat and is_role_title):
                    has_skill_in_list = True
                    break

        has_text_evidence = False
        if any(t in profile_text for t in terms):
            if any(r in current_title for r in TECH_TITLE_HINTS):
                has_text_evidence = True

        if has_skill_in_list or has_text_evidence:
            matched_count += 1
    return matched_count


def check_good_to_have_matches(candidate: dict, good_to_have_skills: dict) -> int:
    """Count matched nice-to-have categories from the JD's rubric."""
    skills = candidate.get("skills", [])
    count = 0
    for cat, cat_def in (good_to_have_skills or {}).items():
        terms = _category_terms(cat_def)
        for s in skills:
            name = s.get("name", "").lower()
            if any(match_skill_term(t, name) for t in terms):
                count += 1
                break
    return count


def must_have_band(matched_count: int, total_categories: int):
    """
    Map the fraction of satisfied must-have categories to a recruiter score
    band. With the default 4-category rubric this reproduces the original
    behavior exactly (4/4 -> top band, 3/4 -> middle band, else low band).
    """
    if total_categories <= 0:
        return 0.10, 0.25
    frac = matched_count / float(total_categories)
    if frac >= 0.999:
        return 0.70, 0.85
    if frac >= 0.70:
        return 0.40, 0.55
    return 0.10, 0.25


def score_candidates(
    candidates,
    features_df,
    semantic_fit,
    skill_depth_fit,
    lexical_fit,
    must_have_skills,
    good_to_have_skills,
    role_type: str = "ml_ai",
    experience_band: dict = None,
    fit_weights=None,
    preferred_locations=None,
    preferred_country: str = "india",
    current_date_str: str = "2026-06-17",
):
    """
    JD-aware recruiter scoring. Returns (final_scores, eligible_mask).

    - must_have_skills / good_to_have_skills drive the tier band and bonuses
      (previously hardcoded to the default AI role for every JD).
    - The ML/NLP-specific soft disqualifiers (LangChain-only, CV/robotics
      without NLP) only apply when the target role is ml_ai.
    - experience_band, fit_weights and preferred_locations are optional; when
      omitted the scoring matches the original default-JD behavior exactly.
    """
    n = len(candidates)
    final_scores = np.zeros(n)
    eligible_mask = np.ones(n, dtype=bool)

    is_honeypot_arr = features_df["is_honeypot"].values
    hard_disq_arr = features_df["hard_disqualified"].values

    # Blend weights for semantic / skill-depth / lexical fit inside the band.
    if fit_weights:
        w_sem, w_skill, w_lex = fit_weights
        w_tot = (w_sem + w_skill + w_lex) or 1.0
        w_sem, w_skill, w_lex = w_sem / w_tot, w_skill / w_tot, w_lex / w_tot
    else:
        w_sem, w_skill, w_lex = 0.5, 0.3, 0.2

    ml_specific_disq = role_type == "ml_ai"
    total_categories = len(must_have_skills)

    for idx, cand in enumerate(candidates):
        row_feat = features_df.iloc[idx]

        # Hard disqualifiers (role-independent) + role mismatch
        is_hp = is_honeypot_arr[idx] == 1 or detect_honeypot(cand)
        is_irrelevant = check_irrelevant_role(cand, role_type)
        is_pure_res = hard_disq_arr[idx] == 1 or check_pure_research_no_production(cand)

        # Generic soft disqualifiers
        is_consulting = row_feat["soft_disq_consulting_only"] == 1
        is_senior_nocode = row_feat["soft_disq_senior_no_code"] == 1
        is_hopper = row_feat["soft_disq_title_chaser"] == 1
        is_closed_source = row_feat["soft_disq_closed_source"] == 1

        # ML/NLP-specific soft disqualifiers only make sense for ml_ai JDs
        is_cv_robotics = ml_specific_disq and row_feat["soft_disq_cv_speech_robotics"] == 1
        is_langchain = ml_specific_disq and row_feat["soft_disq_recent_langchain"] == 1

        disq_scores = []
        if is_hp: disq_scores.append(0.0)
        if is_irrelevant: disq_scores.append(0.0)
        if is_pure_res: disq_scores.append(0.05)
        if is_consulting: disq_scores.append(0.03)
        if is_cv_robotics: disq_scores.append(0.07)
        if is_senior_nocode: disq_scores.append(0.08)
        if is_langchain: disq_scores.append(0.12)
        if is_hopper: disq_scores.append(0.10)
        if is_closed_source: disq_scores.append(0.09)

        if disq_scores:
            final_scores[idx] = min(disq_scores)
            eligible_mask[idx] = False
            continue

        # Must-have tier band from the JD's own rubric
        matched_must = check_must_have_matches(cand, must_have_skills, role_type)
        range_min, range_max = must_have_band(matched_must, total_categories)

        s_val = w_sem * semantic_fit[idx] + w_skill * skill_depth_fit[idx] + w_lex * lexical_fit[idx]
        base_score = range_min + s_val * (range_max - range_min)

        # Adjustments
        adj = 0.0

        gth_count = check_good_to_have_matches(cand, good_to_have_skills)
        adj += gth_count * 0.03

        signals = cand.get("redrob_signals", {})
        last_active_str = signals.get("last_active_date", "")
        days_since = 365
        if last_active_str:
            try:
                curr_date = datetime.strptime(current_date_str, "%Y-%m-%d").date()
                act_date = datetime.strptime(last_active_str, "%Y-%m-%d").date()
                days_since = (curr_date - act_date).days
            except Exception:
                pass

        if days_since <= 30:
            adj += 0.03
        elif days_since >= 180:
            adj -= 0.05

        if signals.get("open_to_work_flag", False):
            adj += 0.02

        notice_days = signals.get("notice_period_days", 0)
        if notice_days <= 30:
            adj += 0.03
        elif notice_days <= 60:
            adj += 0.01

        if notice_days > 90:
            adj -= 0.02
        if notice_days > 120:
            adj -= 0.04

        # JD-specific experience band: no penalty inside the band, up to
        # -0.15 for candidates far outside it.
        if experience_band is not None:
            exp_fit = check_experience_band_score(cand, experience_band)
            adj += (exp_fit - 1.0) * 0.15

        # JD-specific location preference: +/-0.03 around neutral.
        if preferred_locations:
            loc_fit = check_location_fit(cand, preferred_locations, preferred_country)
            adj += (loc_fit - 0.5) * 0.06

        final_scores[idx] = max(0.0, min(1.0, base_score + adj))

    return final_scores, eligible_mask
