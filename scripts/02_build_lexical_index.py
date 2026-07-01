import os
import json
import sys
import pickle

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates_path = os.path.join(base_dir, "data", "candidates.jsonl")
    artifacts_dir = os.path.join(base_dir, "artifacts")
    
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Import source modules
    sys.path.append(base_dir)
    from src.text_build import build_candidate_doc
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        print("Required packages (scikit-learn) not found.")
        sys.exit(1)
        
    print(f"Reading candidates from {candidates_path}...")
    texts = []
    
    count = 0
    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                cand = json.loads(line)
                texts.append(build_candidate_doc(cand))
                count += 1
                if count % 20000 == 0:
                    print(f"Loaded {count} candidates...")
            except Exception as e:
                print(f"Error parsing candidate on line {count}: {e}")
                
    print(f"Finished loading {len(texts)} candidate texts for indexing.")
    
    print("Fitting TF-IDF Vectorizer...")
    # Token pattern that preserves technical symbols like C++, C#, .NET, A/B
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        token_pattern=r"(?u)\b[\w\-\#\+\.]+\b",
        norm="l2"
    )
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    print(f"TF-IDF Matrix shape: {tfidf_matrix.shape}")
    
    vec_file = os.path.join(artifacts_dir, "tfidf_vectorizer.pkl")
    matrix_file = os.path.join(artifacts_dir, "tfidf_matrix.pkl")
    
    print(f"Saving vectorizer to {vec_file}...")
    with open(vec_file, "wb") as f:
        pickle.dump(vectorizer, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print(f"Saving TF-IDF matrix to {matrix_file}...")
    with open(matrix_file, "wb") as f:
        pickle.dump(tfidf_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print("Lexical index creation completed successfully.")

if __name__ == "__main__":
    main()
