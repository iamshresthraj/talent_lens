import os
import json
import sys
import numpy as np

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "all-MiniLM-L6-v2")
    candidates_path = os.path.join(base_dir, "data", "candidates.jsonl")
    artifacts_dir = os.path.join(base_dir, "artifacts")
    
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Import source modules
    sys.path.append(base_dir)
    from src.text_build import build_candidate_doc
    
    # Import sentence-transformers and torch
    try:
        import torch
        # Use all 12 cores for parallelized tensor ops
        torch.set_num_threads(12)
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Required packages (sentence-transformers or torch) not found.")
        sys.exit(1)
        
    if not os.path.exists(model_path):
        print(f"Embedding model not found at {model_path}. Please run setup_models.py first.")
        sys.exit(1)
        
    print(f"Loading SentenceTransformer model from {model_path}...")
    model = SentenceTransformer(model_path)
    model.max_seq_length = 128
    
    print(f"Reading candidates from {candidates_path}...")
    candidate_ids = []
    texts = []
    
    count = 0
    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                cand = json.loads(line)
                candidate_ids.append(cand["candidate_id"])
                texts.append(build_candidate_doc(cand))
                count += 1
                if count % 20000 == 0:
                    print(f"Loaded {count} candidates...")
            except Exception as e:
                print(f"Error parsing candidate on line {count}: {e}")
                
    print(f"Finished loading {len(texts)} candidates.")
    
    print("Generating embeddings (running on CPU with 12 threads)...")
    embeddings = model.encode(
        texts,
        batch_size=1024,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    emb_file = os.path.join(artifacts_dir, "embeddings.npy")
    ids_file = os.path.join(artifacts_dir, "ids.npy")
    
    print(f"Saving embeddings (shape: {embeddings.shape}) to {emb_file}...")
    np.save(emb_file, embeddings)
    
    print(f"Saving candidate IDs to {ids_file}...")
    np.save(ids_file, np.array(candidate_ids))
    
    print("Embedding generation completed successfully.")

if __name__ == "__main__":
    main()
