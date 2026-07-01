import os
import sys
import yaml
import numpy as np

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    jd_config_path = os.path.join(base_dir, "config", "jd_requirements.yaml")
    model_path = os.path.join(base_dir, "models", "all-MiniLM-L6-v2")
    artifacts_dir = os.path.join(base_dir, "artifacts")
    
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Load JD Config
    print(f"Loading JD requirements from {jd_config_path}...")
    with open(jd_config_path, "r", encoding="utf-8") as f:
        jd_config = yaml.safe_load(f)
        
    ideal_text = jd_config.get("ideal_candidate_text", "")
    if not ideal_text:
        print("Error: ideal_candidate_text not found in jd_requirements.yaml.")
        sys.exit(1)
        
    # Import sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Required packages (sentence-transformers) not found.")
        sys.exit(1)
        
    if not os.path.exists(model_path):
        print(f"Embedding model not found at {model_path}. Please run setup_models.py first.")
        sys.exit(1)
        
    print(f"Loading SentenceTransformer model from {model_path}...")
    model = SentenceTransformer(model_path)
    
    print("Encoding ideal candidate text...")
    jd_vector = model.encode(ideal_text, convert_to_numpy=True)
    
    vector_file = os.path.join(artifacts_dir, "jd_vector.npy")
    print(f"Saving JD vector (shape: {jd_vector.shape}) to {vector_file}...")
    np.save(vector_file, jd_vector)
    
    print("JD vector precomputation completed successfully.")

if __name__ == "__main__":
    main()
