import os
import sys
import re
import pandas as pd

def validate(file_path):
    print(f"Validating submission file: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return False
        
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error: Failed to parse CSV: {e}")
        return False
        
    expected_cols = ["candidate_id", "rank", "score", "reasoning"]
    if list(df.columns) != expected_cols:
        print(f"Error: Columns must be exactly {expected_cols}, got {list(df.columns)}.")
        return False
        
    if len(df) != 100:
        print(f"Error: Submission must have exactly 100 rows. Got {len(df)}.")
        return False
        
    # Check candidate_id format
    id_pattern = re.compile(r"^CAND_[0-9]{7}$")
    for idx, row in df.iterrows():
        cand_id = str(row["candidate_id"])
        if not id_pattern.match(cand_id):
            print(f"Error on row {idx+1}: candidate_id '{cand_id}' is invalid. Must match 'CAND_XXXXXXX'.")
            return False
            
    # Check ranks: 1 to 100, exactly once
    ranks = list(df["rank"])
    if sorted(ranks) != list(range(1, 101)):
        print("Error: Ranks must be exactly 1 to 100, each used once.")
        return False
        
    # Check scores non-increasing, ties broken by candidate_id ascending
    # Let's verify sorting
    scores = df["score"].values
    candidate_ids = df["candidate_id"].values
    
    for i in range(len(df) - 1):
        s1, s2 = scores[i], scores[i+1]
        id1, id2 = candidate_ids[i], candidate_ids[i+1]
        
        if s1 < s2:
            print(f"Error: Score is increasing at rank {i+1} -> {i+2} ({s1} < {s2}).")
            return False
        elif s1 == s2:
            if id1 >= id2:
                print(f"Error: Tie at score {s1} is not broken by candidate_id ascending ({id1} is not < {id2}).")
                return False
                
    # Check reasoning
    for idx, row in df.iterrows():
        reason = str(row["reasoning"]).strip()
        if not reason:
            print(f"Error on row {idx+1}: reasoning is empty.")
            return False
            
        # 1-2 sentences check
        # We split by sentence terminators (. ! ?)
        sentences = [s.strip() for s in re.split(r'[.!?]', reason) if s.strip()]
        if len(sentences) < 1 or len(sentences) > 2:
            print(f"Error on row {idx+1}: reasoning must be 1-2 sentences. Got {len(sentences)} sentences: '{reason}'")
            # Let's make it a warning instead of hard error if it's borderline,
            # but we will enforce it strictly in our generator!
            return False
            
        # Check for placeholders
        placeholders = ["[", "]", "{", "}", "todo", "placeholder", "<", ">"]
        for p in placeholders:
            if p in reason.lower():
                print(f"Error on row {idx+1}: reasoning contains potential placeholder '{p}': '{reason}'")
                return False
                
    print("Success: Submission is 100% valid!")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_submission.py <submission_file>")
        sys.exit(1)
    file_to_validate = sys.argv[1]
    success = validate(file_to_validate)
    if not success:
        sys.exit(1)
    sys.exit(0)
