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
    # Score each role by counting word-boundaried keyword hits rather than taking the
    # first substring match in a fixed if/elif order. The old ordering let generic terms
    # (e.g. "infrastructure" -> devops, "frontend" mentioned in passing) win over the
    # role's actual, more specific signals, causing many different JDs to collapse onto
    # the same role_type (and therefore the same weights/skill taxonomy/output).
    ROLE_KEYWORDS = {
        "mobile": [r"\bmobile\b", r"\bios\b", r"\bandroid\b", r"react native"],
        "data_engineer": [r"data engineer", r"\betl\b", r"data warehouse", r"data pipeline"],
        "tpm": [r"product manager", r"\btpm\b", r"technical product"],
        "devops": [r"\bdevops\b", r"platform engineer", r"\bsre\b", r"\binfrastructure\b", r"site reliability"],
        "fullstack": [r"full[\s-]?stack"],
        "frontend": [r"\bfrontend\b", r"front-end", r"react engineer", r"\bui engineer\b"],
        "backend": [r"\bbackend\b", r"back-end", r"go engineer", r"python api"],
        "ml_ai": [r"ai engineer", r"machine learning", r"\bnlp\b", r"\bml engineer\b", r"\bmlops\b", r"\bllm\b"],
    }
    # Tie-break order: more specific role types first, generic "ml_ai" catch-all last.
    ROLE_PRIORITY = ["mobile", "data_engineer", "tpm", "devops", "fullstack", "frontend", "backend", "ml_ai"]

    role_scores = {role: 0 for role in ROLE_KEYWORDS}
    for role, patterns in ROLE_KEYWORDS.items():
        for pattern in patterns:
            role_scores[role] += len(re.findall(pattern, jd_lower))

    best_role = max(ROLE_PRIORITY, key=lambda r: (role_scores[r], -ROLE_PRIORITY.index(r)))
    role_type = best_role if role_scores[best_role] > 0 else "ml_ai"
        
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
