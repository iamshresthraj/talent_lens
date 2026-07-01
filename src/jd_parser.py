import os
import re
import yaml

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Default weights from config/weights.yaml for the default Senior AI Engineer role
DEFAULT_LINEAR_WEIGHTS = {
    "semantic_fit": 0.28,
    "skill_depth_fit": 0.20,
    "lexical_fit": 0.12,
    "structural_fit": 0.30,
    "logistics_fit": 0.10
}

# Role-specific default linear weights adjustments
ROLE_WEIGHTS = {
    "ml_ai": {
        "semantic_fit": 0.28,
        "skill_depth_fit": 0.20,
        "lexical_fit": 0.12,
        "structural_fit": 0.30,
        "logistics_fit": 0.10
    },
    "backend": {
        "semantic_fit": 0.15,
        "skill_depth_fit": 0.25,
        "lexical_fit": 0.10,
        "structural_fit": 0.35,  # Emphasize backend structure/practices
        "logistics_fit": 0.15
    },
    "frontend": {
        "semantic_fit": 0.15,
        "skill_depth_fit": 0.25,
        "lexical_fit": 0.15,
        "structural_fit": 0.30,
        "logistics_fit": 0.15
    },
    "fullstack": {
        "semantic_fit": 0.15,
        "skill_depth_fit": 0.25,
        "lexical_fit": 0.15,
        "structural_fit": 0.30,
        "logistics_fit": 0.15
    },
    "devops": {
        "semantic_fit": 0.10,
        "skill_depth_fit": 0.30,  # Highly specific infra toolkits
        "lexical_fit": 0.10,
        "structural_fit": 0.35,
        "logistics_fit": 0.15
    },
    "data_engineer": {
        "semantic_fit": 0.15,
        "skill_depth_fit": 0.25,
        "lexical_fit": 0.10,
        "structural_fit": 0.35,
        "logistics_fit": 0.15
    },
    "tpm": {
        "semantic_fit": 0.10,
        "skill_depth_fit": 0.20,
        "lexical_fit": 0.10,
        "structural_fit": 0.40,  # Prioritize trajectory/seniority fit
        "logistics_fit": 0.20
    },
    "mobile": {
        "semantic_fit": 0.15,
        "skill_depth_fit": 0.25,
        "lexical_fit": 0.15,
        "structural_fit": 0.30,
        "logistics_fit": 0.15
    }
}

def parse_jd(jd_text: str) -> dict:
    """
    Parses a job description to determine:
    1. Role type
    2. Must-have and nice-to-have skills mapping from the taxonomy
    3. Experience band
    4. Seniority / Context
    5. Scoring weights
    """
    if not jd_text:
        jd_text = ""
        
    jd_lower = jd_text.lower()
    
    # 1. Detect Role Type
    role_type = "ml_ai" # Default
    if "frontend" in jd_lower or "react engineer" in jd_lower:
        role_type = "frontend"
    elif "backend" in jd_lower or "go engineer" in jd_lower or "python api" in jd_lower:
        role_type = "backend"
    elif "full stack" in jd_lower or "full-stack" in jd_lower or "fullstack" in jd_lower:
        role_type = "fullstack"
    elif "devops" in jd_lower or "platform engineer" in jd_lower or "sre" in jd_lower or "infrastructure" in jd_lower:
        role_type = "devops"
    elif "data engineer" in jd_lower or "etl" in jd_lower or "data warehouse" in jd_lower:
        role_type = "data_engineer"
    elif "product manager" in jd_lower or "tpm" in jd_lower or "technical product" in jd_lower:
        role_type = "tpm"
    elif "mobile" in jd_lower or "ios" in jd_lower or "android" in jd_lower or "react native" in jd_lower:
        role_type = "mobile"
    elif "ai engineer" in jd_lower or "machine learning" in jd_lower or "ml" in jd_lower or "nlp" in jd_lower:
        role_type = "ml_ai"
        
    # Check if this matches the default Senior AI Engineer - Founding Team JD text/context
    is_default_ai_role = "redrob" in jd_lower and ("founding team" in jd_lower or "founding senior ai" in jd_lower)
    
    # Load Taxonomy
    taxonomy_path = os.path.join(base_dir, "config", "skill_taxonomy.yaml")
    taxonomy = {}
    if os.path.exists(taxonomy_path):
        with open(taxonomy_path, "r", encoding="utf-8") as f:
            taxonomy = yaml.safe_load(f)
            
    # Extract skills for this role type
    role_skills = taxonomy.get(role_type, {})
    
    must_have_skills = {}
    nice_to_have_skills = {}
    
    # Parse skills from JD
    # If it is the default AI role, load default must_haves and nice_to_haves from config/jd_requirements.yaml
    if is_default_ai_role:
        jd_req_path = os.path.join(base_dir, "config", "jd_requirements.yaml")
        if os.path.exists(jd_req_path):
            with open(jd_req_path, "r", encoding="utf-8") as f:
                jd_req = yaml.safe_load(f)
                must_have_skills = jd_req.get("must_have_skills", {})
                nice_to_have_skills = jd_req.get("nice_to_have_skills", {})
    
    # If not loaded or empty, detect from taxonomy
    if not must_have_skills:
        # Match taxonomy terms against JD text
        for skill_key, skill_def in role_skills.items():
            terms = skill_def.get("terms", [])
            weight = skill_def.get("weight", 1.0)
            
            # Simple substring matching
            matched = False
            for term in terms:
                if term.lower() in jd_lower:
                    matched = True
                    break
            
            if matched:
                must_have_skills[skill_key] = {
                    "weight": weight,
                    "terms": terms
                }
                
        # If still empty (e.g. custom JD with no matching skills), fall back to a generic skill set
        if not must_have_skills:
            must_have_skills = {
                "general_software": {
                    "weight": 1.0,
                    "terms": ["software", "engineering", "coding", "development", "programming"]
                }
            }

    # 2. Parse Experience Band
    soft_min, soft_max = 5, 9
    # Matches "X to Y years", "X-Y years", "X - Y years"
    match = re.search(r'(\d+)\s*(?:to|-)\s*(\d+)\s*years?', jd_lower)
    if match:
        soft_min = int(match.group(1))
        soft_max = int(match.group(2))
    else:
        # Matches "X+ years", "X + years"
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
    
    # 3. Weights selection
    weights = ROLE_WEIGHTS.get(role_type, DEFAULT_LINEAR_WEIGHTS).copy()
    if is_default_ai_role:
        # Load weights from config/weights.yaml if it's the official run
        weights_path = os.path.join(base_dir, "config", "weights.yaml")
        if os.path.exists(weights_path):
            with open(weights_path, "r", encoding="utf-8") as f:
                w_config = yaml.safe_load(f)
                weights = w_config.get("linear_weights", DEFAULT_LINEAR_WEIGHTS)
                
    return {
        "role_type": role_type,
        "must_have_skills": must_have_skills,
        "nice_to_have_skills": nice_to_have_skills,
        "experience_band": exp_band,
        "dimension_weights": weights
    }
