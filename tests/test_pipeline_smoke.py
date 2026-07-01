import os
import sys
import subprocess
import pandas as pd
import pytest

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

def test_end_to_end_pipeline():
    """Runs rank.py on a 500-candidate slice and verifies results."""
    candidates_path = os.path.join(base_dir, "data", "candidates.jsonl")
    test_dir = os.path.join(base_dir, "tests")
    os.makedirs(test_dir, exist_ok=True)
    
    slice_path = os.path.join(test_dir, "test_candidates_slice.jsonl")
    submission_path = os.path.join(test_dir, "test_submission.csv")
    
    # 1. Create a slice of first 500 candidates
    print("Creating 500 candidate slice...")
    count = 0
    with open(candidates_path, "r", encoding="utf-8") as fin:
        with open(slice_path, "w", encoding="utf-8") as fout:
            for line in fin:
                if count >= 500:
                    break
                fout.write(line)
                count += 1
                
    assert count == 500, f"Expected 500 candidates, only loaded {count}"
    
    # 2. Run rank.py
    print("Running rank.py on test slice...")
    rank_script = os.path.join(base_dir, "rank.py")
    cmd = [
        sys.executable,
        rank_script,
        "--candidates", slice_path,
        "--out", submission_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print("rank.py stdout:")
    print(result.stdout)
    print("rank.py stderr:")
    print(result.stderr)
    
    assert result.returncode == 0, f"rank.py failed with return code {result.returncode}"
    
    # 3. Validate output file existence and format
    assert os.path.exists(submission_path), "Submission file not created"
    
    df = pd.read_csv(submission_path)
    assert len(df) == 100, f"Expected exactly 100 rows, got {len(df)}"
    assert list(df.columns) == ["candidate_id", "rank", "score", "reasoning"], "Columns mismatch"
    
    # 4. Run validate_submission.py
    validate_script = os.path.join(base_dir, "validate_submission.py")
    cmd_val = [
        sys.executable,
        validate_script,
        submission_path
    ]
    result_val = subprocess.run(cmd_val, capture_output=True, text=True)
    print("validate_submission.py stdout:")
    print(result_val.stdout)
    print("validate_submission.py stderr:")
    print(result_val.stderr)
    
    assert result_val.returncode == 0, "Submission validation failed"
    
    # 5. Clean up test files
    try:
        os.remove(slice_path)
        os.remove(submission_path)
    except OSError:
        pass
