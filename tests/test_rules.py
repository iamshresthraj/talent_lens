import os
import sys
import pytest

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from src.rules import detect_honeypot

def make_base_candidate():
    """Returns a valid baseline candidate that triggers no honeypot rules."""
    return {
        "candidate_id": "CAND_1234567",
        "profile": {
            "years_of_experience": 5.0,
            "current_title": "AI Engineer",
            "current_company": "Redrob",
            "location": "Pune, India",
            "country": "India"
        },
        "career_history": [
            {
                "title": "Software Engineer",
                "company": "Redrob",
                "start_date": "2021-01-01",
                "end_date": "2026-01-01",
                "duration_months": 60,
                "is_current": True,
                "industry": "HR-tech",
                "company_size": "11-50",
                "description": "Building stuff"
            }
        ],
        "education": [
            {
                "institution": "IIT Delhi",
                "degree": "B.Tech",
                "field_of_study": "Computer Science",
                "start_year": 2016,
                "end_year": 2020
            }
        ],
        "skills": [
            {
                "name": "Python",
                "proficiency": "expert",
                "endorsements": 10,
                "duration_months": 48
            }
        ],
        "redrob_signals": {
            "expected_salary_range_inr_lpa": {"min": 20, "max": 30},
            "last_active_date": "2026-06-01",
            "willing_to_relocate": True
        }
    }


def test_rule1_only():
    # Trigger 1: expert skill with short duration
    cand = make_base_candidate()
    cand["skills"].append({
        "name": "TensorFlow",
        "proficiency": "expert",
        "endorsements": 1,
        "duration_months": 3
    })
    # Should not flag because only 1 rule is triggered
    assert detect_honeypot(cand) is False


def test_rule1_strong():
    # Trigger 1 strong: 2 expert skills with short duration
    cand = make_base_candidate()
    cand["skills"] = [
        {"name": "PyTorch", "proficiency": "expert", "endorsements": 2, "duration_months": 2},
        {"name": "Transformers", "proficiency": "expert", "endorsements": 5, "duration_months": 3}
    ]
    # Should flag since 2+ expert short-duration skills counts as 2+ triggers
    assert detect_honeypot(cand) is True


def test_rule2_only():
    # Trigger 2: experience mismatch (claimed 10 yrs, history is 1 yr)
    cand = make_base_candidate()
    cand["profile"]["years_of_experience"] = 10.0
    cand["career_history"][0]["duration_months"] = 12
    assert detect_honeypot(cand) is False


def test_rule3_only():
    # Trigger 3: multiple current roles
    cand = make_base_candidate()
    # Adjust first job duration so total is 60 months (5 years) to avoid Rule 2
    cand["career_history"][0]["duration_months"] = 20
    cand["career_history"].append({
        "title": "Machine Learning Engineer",
        "company": "Other Corp",
        "start_date": "2023-01-01",
        "end_date": None,
        "duration_months": 40,
        "is_current": True,
        "industry": "Tech",
        "company_size": "51-200",
        "description": "Doing ML"
    })
    assert detect_honeypot(cand) is False


def test_rule4_only():
    # Trigger 4: job end date before start date
    cand = make_base_candidate()
    cand["career_history"][0]["end_date"] = "2020-01-01"
    cand["career_history"][0]["start_date"] = "2021-01-01"
    assert detect_honeypot(cand) is False


def test_rule5_only():
    # Trigger 5: education finished after work starts, no prior degree explaining
    cand = make_base_candidate()
    # first job starts in 2021. Degree ends in 2025.
    cand["career_history"][0]["start_date"] = "2021-01-01"
    cand["education"][0]["end_year"] = 2025
    assert detect_honeypot(cand) is False


def test_rule6_only():
    # Trigger 6: min salary > max
    cand = make_base_candidate()
    cand["redrob_signals"]["expected_salary_range_inr_lpa"] = {"min": 50, "max": 40}
    assert detect_honeypot(cand) is False


def test_rule7_only():
    # Trigger 7: high experience (>10), but only recent degree (<3 yrs)
    cand = make_base_candidate()
    cand["profile"]["years_of_experience"] = 15.0
    cand["career_history"][0]["duration_months"] = 180  # 15 years
    cand["career_history"][0]["start_date"] = "2021-01-01"  # avoid Rule 5 (ends <= start + 3)
    # Current relative year: 2026. Recent degree: 2024.
    cand["education"][0]["end_year"] = 2024
    cand["education"][0]["start_year"] = 2020
    assert detect_honeypot(cand) is False


def test_two_rules_triggered():
    # Trigger 1 and 6
    cand = make_base_candidate()
    # expert skill with short duration
    cand["skills"].append({
        "name": "Keras",
        "proficiency": "expert",
        "endorsements": 0,
        "duration_months": 2
    })
    # min salary > max
    cand["redrob_signals"]["expected_salary_range_inr_lpa"] = {"min": 60, "max": 40}
    
    assert detect_honeypot(cand) is True


def test_three_rules_triggered():
    cand = make_base_candidate()
    # Rule 3: Multiple current
    cand["career_history"].append({
        "title": "MLE", "company": "Other", "start_date": "2023-01-01", "end_date": None,
        "duration_months": 12, "is_current": True, "industry": "IT", "company_size": "1-10", "description": ""
    })
    # Rule 4: end < start
    cand["career_history"][0]["end_date"] = "2019-01-01"
    cand["career_history"][0]["start_date"] = "2020-01-01"
    # Rule 6: salary min > max
    cand["redrob_signals"]["expected_salary_range_inr_lpa"] = {"min": 90, "max": 50}
    
    assert detect_honeypot(cand) is True
