import os
import sys

def main():
    try:
        from huggingface_hub import snapshot_download, hf_hub_download
    except ImportError:
        print("Required packages (huggingface_hub) not found. Please run: pip install huggingface_hub")
        sys.exit(1)
        
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Download Sentence-Transformer model
    emb_model_dir = os.path.join(models_dir, "all-MiniLM-L6-v2")
    print(f"Downloading sentence-transformer 'all-MiniLM-L6-v2' to {emb_model_dir}...")
    try:
        snapshot_download(
            repo_id="sentence-transformers/all-MiniLM-L6-v2",
            local_dir=emb_model_dir,
            local_dir_use_symlinks=False
        )
        print("Sentence-transformer downloaded and cached successfully.")
    except Exception as e:
        print(f"Error downloading embedding model: {e}")
        sys.exit(1)
        
    # 2. Download tiny GGUF model for Gradio reasoning
    print(f"Downloading Qwen2.5-0.5B-Instruct-GGUF to {models_dir}...")
    try:
        # We download Qwen2.5-0.5B-Instruct-GGUF Q4_K_M which is extremely tiny (~397MB)
        hf_hub_download(
            repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            filename="qwen2.5-0.5b-instruct-q4_k_m.gguf",
            local_dir=models_dir,
            local_dir_use_symlinks=False
        )
        print("GGUF model downloaded and cached successfully.")
    except Exception as e:
        print(f"Error downloading GGUF model: {e}")
        # Note: GGUF download failure shouldn't crash setup completely if the embedding model worked,
        # but let's notify the user.
        print("Continuing setup, GGUF model could not be downloaded. Local LLM will not be available in Gradio app.")

if __name__ == "__main__":
    main()
