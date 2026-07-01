import re
from datetime import datetime

def detect_honeypot(candidate: dict) -> bool:
    """
    Flag a candidate as a honeypot if 2 or more of these rules trigger:
    1. Any skill with proficiency == "expert" and duration_months < 6 (2+ is strong trigger).
    2. abs(sum(career_history.duration_months)/12 - years_of_experience) > years_of_experience * 0.3 (and diff > 1 year).
    3. More than one career_history entry with is_current == true, or more than one with end_date == null.
    4. Any career_history entry where end_date < start_date.
    5. education[].end_year later than the start_year of the candidate's first career_history entry by an implausible margin.
    6. expected_salary_range_inr_lpa.min > expected_salary_range_inr_lpa.max.
    7. years_of_experience > 10 while education[].end_year is very recent (<3 years ago) and no earlier degree.
    """
    rules_fired_count = 0

    # Rule 1: Expert skill with duration < 6 months
    expert_short_duration_skills = [
        s for s in candidate.get("skills", [])
        if s.get("proficiency") == "expert" and s.get("duration_months", 0) < 6
    ]
    rule1 = len(expert_short_duration_skills) >= 1
    rule1_strong = len(expert_short_duration_skills) >= 2
    if rule1:
        rules_fired_count += 1
    if rule1_strong:
        rules_fired_count += 1  # 2+ is a strong trigger that counts as a second signal on its own

    # Rule 2: Work history duration doesn't match total years of experience
    career_history = candidate.get("career_history", [])
    years_of_experience = candidate.get("profile", {}).get("years_of_experience", 0)
    total_history_months = sum(job.get("duration_months", 0) for job in career_history)
    history_years = total_history_months / 12.0
    diff = abs(history_years - years_of_experience)
    rule2 = (diff > years_of_experience * 0.3) and (diff > 1.0)
    if rule2:
        rules_fired_count += 1

    # Rule 3: Multiple current jobs or null end dates
    current_jobs = [job for job in career_history if job.get("is_current") is True]
    no_end_date_jobs = [job for job in career_history if job.get("end_date") is None]
    rule3 = len(current_jobs) > 1 or len(no_end_date_jobs) > 1
    if rule3:
        rules_fired_count += 1

    # Rule 4: Job end date is before start date
    rule4 = False
    for job in career_history:
        start = job.get("start_date")
        end = job.get("end_date")
        if start and end:
            if end < start:
                rule4 = True
                break
    if rule4:
        rules_fired_count += 1

    # Find earliest job start year
    job_start_years = []
    for job in career_history:
        start = job.get("start_date")
        if start:
            try:
                y = int(start.split("-")[0])
                job_start_years.append(y)
            except ValueError:
                pass
    first_job_start_year = min(job_start_years) if job_start_years else None

    # Rule 5: Education finished too long after starting work, with no earlier degree explaining it
    rule5 = False
    education = candidate.get("education", [])
    if first_job_start_year is not None and education:
        has_prior_degree = False
        has_late_degree = False
        for edu in education:
            end_y = edu.get("end_year")
            if end_y:
                if end_y <= first_job_start_year:
                    has_prior_degree = True
                if end_y > first_job_start_year + 3:
                    has_late_degree = True
        if has_late_degree and not has_prior_degree:
            rule5 = True
    if rule5:
        rules_fired_count += 1

    # Rule 6: Expected salary min > max
    salary = candidate.get("redrob_signals", {}).get("expected_salary_range_inr_lpa", {})
    rule6 = False
    if isinstance(salary, dict):
        sal_min = salary.get("min")
        sal_max = salary.get("max")
        if sal_min is not None and sal_max is not None:
            rule6 = sal_min > sal_max
    if rule6:
        rules_fired_count += 1

    # Rule 7: High experience, but only a very recent degree
    rule7 = False
    if years_of_experience > 10 and education:
        edu_ends = [e.get("end_year") for e in education if e.get("end_year")]
        if edu_ends:
            latest_edu_end = max(edu_ends)
            earliest_edu_end = min(edu_ends)
            # Both ends are recent (relative to 2026 current year, e.g. >= 2023)
            if latest_edu_end >= 2023 and earliest_edu_end >= 2023:
                rule7 = True
    if rule7:
        rules_fired_count += 1

    return rules_fired_count >= 2


def check_pure_research_no_production(candidate: dict) -> bool:
    """
    Hard disqualifier: Career history shows only academic/research-lab titles/employers,
    and there is no shipped/production work mentioned.
    """
    history = candidate.get("career_history", [])
    if not history:
        return False
    
    academic_keywords = {"researcher", "research intern", "research assistant", "phd", "postdoc", "professor", "academic", "student", "fellow"}
    all_academic = True
    for job in history:
        title = job.get("title", "").lower()
        company = job.get("company", "").lower()
        is_academic = False
        for kw in academic_keywords:
            if kw in title or kw in company:
                is_academic = True
                break
        if not is_academic:
            all_academic = False
            break
            
    # Check for production/scale keywords in profile headline, summary, and work history
    headline = candidate.get("profile", {}).get("headline", "").lower()
    summary = candidate.get("profile", {}).get("summary", "").lower()
    
    prod_keywords = {"production", "shipped", "deployed", "launch", "commercial", "scale", "infrastructure", "product company"}
    has_production = False
    for kw in prod_keywords:
        if kw in headline or kw in summary:
            has_production = True
            break
        for job in history:
            if kw in job.get("description", "").lower() or kw in job.get("title", "").lower():
                has_production = True
                break
                
    return all_academic and not has_production


def check_recent_langchain_only_no_legacy_ml(candidate: dict) -> bool:
    """
    Soft disqualifier: AI experience consisting only of <12-month-old LangChain-calls-OpenAI work
    without pre-LLM-era ML production experience.
    """
    skills_text = " ".join([s.get("name", "").lower() for s in candidate.get("skills", [])])
    history_text = ""
    for job in candidate.get("career_history", []):
        history_text += " " + job.get("title", "").lower() + " " + job.get("description", "").lower()
    full_text = skills_text + " " + history_text
    
    has_llm_wrappers = "langchain" in full_text or "openai" in full_text or "gpt" in full_text or "prompt engineering" in full_text
    has_traditional_ml = any(kw in full_text for kw in [
        "scikit-learn", "xgboost", "random forest", "svm", "tensorflow", "pytorch", 
        "keras", "pandas", "numpy", "classical ml", "traditional ml", "regression", 
        "clustering", "classification", "scipy", "statsmodels", "logistic regression"
    ])
    
    return has_llm_wrappers and not has_traditional_ml


def check_senior_no_hands_on_code_18mo(candidate: dict) -> bool:
    """
    Soft disqualifier: Senior engineers who haven't written production code in 18+ months
    due to moving into pure architecture/tech-lead roles.
    """
    current_jobs = [job for job in candidate.get("career_history", []) if job.get("is_current")]
    if not current_jobs:
        return False
    
    for job in current_jobs:
        title = job.get("title", "").lower()
        is_pure_lead = any(kw in title for kw in ["manager", "director", "architect", "lead"]) and not any(kw in title for kw in ["developer", "engineer", "coder", "programmer"])
        duration = job.get("duration_months", 0)
        if is_pure_lead and duration >= 18:
            desc = job.get("description", "").lower()
            coding_keywords = ["python", "code", "coding", "programming", "hands-on", "develop", "implementing", "building", "engineer"]
            has_coding = any(kw in desc for kw in coding_keywords)
            if not has_coding:
                return True
    return False


def check_consulting_only_career(candidate: dict, consulting_firms: list) -> bool:
    """
    Soft disqualifier: Candidate whose entire career is at pure consulting firms.
    """
    history = candidate.get("career_history", [])
    if not history:
        return False
    
    for job in history:
        comp = job.get("company", "").lower()
        is_consulting = False
        for firm in consulting_firms:
            if firm in comp:
                is_consulting = True
                break
        if not is_consulting:
            return False
    return True


def check_cv_speech_robotics_only_no_nlp(candidate: dict) -> bool:
    """
    Soft disqualifier: Primary expertise is CV/speech/robotics with no significant NLP/IR exposure.
    """
    skills_text = " ".join([s.get("name", "").lower() for s in candidate.get("skills", [])])
    history_text = ""
    for job in candidate.get("career_history", []):
        history_text += " " + job.get("title", "").lower() + " " + job.get("description", "").lower()
    full_text = skills_text + " " + history_text
    
    has_cv_speech_robotics = any(kw in full_text for kw in [
        "computer vision", "image processing", "cnn", "yolo", "opencv", "speech recognition", 
        "text-to-speech", "speech-to-text", "robotics", "lidar", "slam", "ros", "audio processing"
    ])
    has_nlp_ir = any(kw in full_text for kw in [
        "nlp", "natural language", "bert", "gpt", "transformer", "text classification", 
        "search", "retrieval", "embeddings", "vector db", "milvus", "qdrant", "pinecone", 
        "faiss", "elasticsearch", "information retrieval", "bm25", "tf-idf"
    ])
    return has_cv_speech_robotics and not has_nlp_ir


def check_title_chaser_job_hopper(candidate: dict) -> bool:
    """
    Soft disqualifier: Company hopping every ~1.5 years climbing Senior -> Staff -> Principal.
    """
    history = candidate.get("career_history", [])
    if not history or len(history) < 2:
        return False
        
    total_months = sum(job.get("duration_months", 0) for job in history)
    avg_months = total_months / len(history)
    is_hopper = avg_months < 20  # less than ~1.6 years on average
    
    titles = [job.get("title", "").lower() for job in history]
    has_senior = any("senior" in t for t in titles)
    has_staff = any("staff" in t for t in titles)
    has_principal = any("principal" in t for t in titles)
    
    is_climber = (has_senior and has_staff) or (has_staff and has_principal) or (has_senior and has_principal)
    
    return is_hopper and is_climber


def check_closed_source_only_no_external_validation(candidate: dict) -> bool:
    """
    Soft disqualifier: 5+ years experience entirely on closed-source proprietary systems
    with zero external validation (papers, talks, OSS).
    """
    years_exp = candidate.get("profile", {}).get("years_of_experience", 0)
    if years_exp < 5:
        return False
        
    github_score = candidate.get("redrob_signals", {}).get("github_activity_score", -1)
    if github_score != -1:
        return False
        
    summary = candidate.get("profile", {}).get("summary", "").lower()
    headline = candidate.get("profile", {}).get("headline", "").lower()
    skills_text = " ".join([s.get("name", "").lower() for s in candidate.get("skills", [])])
    
    external_keywords = ["paper", "publication", "patent", "talk", "speaker", "conference", "open source", "oss", "contributor"]
    has_validation = any(kw in summary or kw in headline or kw in skills_text for kw in external_keywords)
    
    return not has_validation


def check_location_fit(candidate: dict, preferred_locations: list, preferred_country: str) -> float:
    """
    Calculate location fit score:
    - 1.0 if current location matches preferred OR country matches and willing to relocate
    - 0.5 if country matches but not in preferred location and not willing to relocate
    - 0.0 if country does not match (case-by-case, no visa)
    """
    profile = candidate.get("profile", {})
    loc = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    signals = candidate.get("redrob_signals", {})
    willing = signals.get("willing_to_relocate", False)
    
    # Standardize country
    if country == "india" or preferred_country.lower() in country:
        # Check if in preferred locations
        is_preferred_loc = any(p_loc in loc for p_loc in preferred_locations)
        if is_preferred_loc or willing:
            return 1.0
        else:
            return 0.5
    else:
        return 0.0


def check_notice_period_fit(candidate: dict, ideal_days: int = 30) -> float:
    """
    Calculate notice period fit:
    - <= 30 days: 1.0
    - 31-60 days: 0.6
    - 61-90 days: 0.3
    - >90 days: 0.0
    """
    signals = candidate.get("redrob_signals", {})
    days = signals.get("notice_period_days", 0)
    
    if days <= ideal_days:
        return 1.0
    elif days <= 60:
        return 0.6
    elif days <= 90:
        return 0.3
    else:
        return 0.0


def check_salary_sanity_fit(candidate: dict) -> float:
    """
    Check expected salary.
    For a senior role in India (Lakhs Per Annum):
    - Min expected <= 45 LPA: 1.0
    - Min expected 46-90 LPA: decays linearly from 1.0 to 0.5
    - Min expected > 90 LPA: 0.0 (high budget risk)
    """
    salary = candidate.get("redrob_signals", {}).get("expected_salary_range_inr_lpa", {})
    if not isinstance(salary, dict):
        return 1.0
    min_val = salary.get("min")
    if min_val is None:
        return 1.0
        
    if min_val <= 45:
        return 1.0
    elif min_val <= 90:
        return 1.0 - 0.5 * ((min_val - 45) / 45)
    else:
        return 0.0


def check_experience_band_score(candidate: dict, band: dict) -> float:
    """
    Evaluate experience band fit:
    - soft_min to soft_max (5 to 9 years): 1.0
    - < hard_floor (3 years): 0.0
    - hard_floor to soft_min: linear scaling from 0.0 to 1.0
    - soft_max to hard_ceiling: linear scaling from 1.0 to 0.8
    - > hard_ceiling (16 years): 0.7
    """
    profile = candidate.get("profile", {})
    years = profile.get("years_of_experience", 0)
    
    soft_min = band.get("soft_min", 5)
    soft_max = band.get("soft_max", 9)
    hard_floor = band.get("hard_floor", 3)
    hard_ceiling = band.get("hard_ceiling", 16)
    
    if years < hard_floor:
        return 0.0
    elif years < soft_min:
        return (years - hard_floor) / (soft_min - hard_floor)
    elif years <= soft_max:
        return 1.0
    elif years <= hard_ceiling:
        return 1.0 - 0.2 * ((years - soft_max) / (hard_ceiling - soft_max))
    else:
        return 0.7


def check_production_evidence_score(candidate: dict) -> float:
    """
    Scan resume text for keywords indicating production/shipped experience:
    - matched >= 3: 1.0
    - matched >= 1: 0.8
    - matched 0: 0.5
    """
    profile = candidate.get("profile", {})
    headline = profile.get("headline", "").lower()
    summary = profile.get("summary", "").lower()
    
    history_desc = []
    for job in candidate.get("career_history", []):
        history_desc.append(job.get("description", "").lower())
        history_desc.append(job.get("title", "").lower())
        
    full_text = f"{headline} {summary} " + " ".join(history_desc)
    
    prod_kws = ["shipped", "deployed", "production", "real users", "productionized", "scale", "infrastructure", "launch"]
    matches = sum(1 for kw in prod_kws if kw in full_text)
    
    if matches >= 3:
        return 1.0
    elif matches >= 1:
        return 0.8
    else:
        return 0.5


def check_irrelevant_role(candidate: dict, role_type: str) -> bool:
    """
    Returns True if the candidate's profile is completely irrelevant to the target role type.
    For example, a Mechanical Engineer or HR Manager applying for a Backend Engineer role.
    """
    profile = candidate.get("profile", {})
    title = str(profile.get("current_title", "")).lower()
    headline = str(profile.get("headline", "")).lower()
    
    # Non-technical roles are always irrelevant to engineering roles
    non_tech_keywords = [
        "mechanical engineer", "civil engineer", "chemical engineer", "aerospace engineer",
        "human resources", "recruiter", "talent acquisition", "hr generalist", "hr manager",
        "sales executive", "sales manager", "account executive", "marketing manager", 
        "seo specialist", "content writer", "copywriter", "graphic designer", 
        "operations manager", "operations associate", "finance analyst", "accountant",
        "qa tester", "manual tester", "qa engineer", "quality assurance", "test engineer",
        "business analyst", "project coordinator", "project manager", "customer support",
        "customer service", "technical support"
    ]
    
    for kw in non_tech_keywords:
        if kw in title or kw in headline:
            return True
            
    # Product Manager is irrelevant for engineering roles
    if role_type in ["backend", "frontend", "fullstack", "devops", "data_engineer", "ml_ai", "mobile"]:
        if "product manager" in title or "product manager" in headline:
            return True
            
    # Mismatches between technical roles
    if role_type in ["backend", "devops", "data_engineer", "ml_ai"]:
        frontend_keywords = ["frontend engineer", "frontend developer", "ui/ux developer", "ui/ux designer", "web designer"]
        for kw in frontend_keywords:
            if kw in title or kw in headline:
                return True
                
    if role_type == "frontend":
        backend_keywords = ["backend engineer", "backend developer", "devops engineer", "sre", "infrastructure engineer", "data engineer"]
        for kw in backend_keywords:
            if kw in title or kw in headline:
                return True
                
    return False



