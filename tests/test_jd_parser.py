import os
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from src.jd_parser import parse_jd

def test_parse_jd_default_role():
    jd_text = "Hiring Founding Senior AI Engineer at Redrob AI, Pune/Noida. Deployed embeddings-based retrieval systems."
    parsed = parse_jd(jd_text)
    assert parsed["role_type"] == "ml_ai"
    assert "embeddings_retrieval" in parsed["must_have_skills"]
    assert parsed["experience_band"]["soft_min"] == 5
    assert parsed["experience_band"]["soft_max"] == 9
    assert abs(parsed["dimension_weights"]["semantic_fit"] - 0.28) < 1e-5

def test_parse_jd_backend():
    jd_text = "Hiring Senior Backend Engineer with 4 to 8 years experience. Need strong SQL query optimization, REST APIs, and message queues like Kafka."
    parsed = parse_jd(jd_text)
    assert parsed["role_type"] == "backend"
    assert parsed["experience_band"]["soft_min"] == 4
    assert parsed["experience_band"]["soft_max"] == 8
    assert "sql" in parsed["must_have_skills"]
    assert "rest_api" in parsed["must_have_skills"]
    assert "message_queue" in parsed["must_have_skills"]
    # Verify weights are adjusted
    assert parsed["dimension_weights"]["structural_fit"] == 0.35

def test_parse_jd_frontend():
    jd_text = "React Frontend Specialist. Need 3+ years experience in React, TypeScript, CSS layout models, and responsive web design."
    parsed = parse_jd(jd_text)
    assert parsed["role_type"] == "frontend"
    assert parsed["experience_band"]["soft_min"] == 3
    assert parsed["experience_band"]["soft_max"] == 7
    assert "react" in parsed["must_have_skills"]
    assert "typescript" in parsed["must_have_skills"]
    assert "css" in parsed["must_have_skills"]

def test_parse_jd_devops():
    jd_text = "Hiring DevOps/Platform Engineer. Managed Kubernetes clusters and Terraform infrastructure as code."
    parsed = parse_jd(jd_text)
    assert parsed["role_type"] == "devops"
    assert "kubernetes" in parsed["must_have_skills"]
    assert "terraform" in parsed["must_have_skills"]
