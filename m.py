import os
import modal
from huggingface_hub import hf_hub_download

app = modal.App("sam-vit")
volume = modal.Volume.from_name("sam-weights-volume", create_if_missing=True)

@app.function(
    image=modal.Image.debian_slim().pip_install("huggingface-hub>=0.16.0"),
    volumes={"/weights": volume}
)
def download_model_weights():
    repo_id = "astle/sam"
    filename = "sam_vit_h_4b8939.pth"

    os.makedirs("/weights", exist_ok=True)
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir="/weights",
        local_dir_use_symlinks=False
    )
    
    print(f"Downloaded model weights to {model_path}")
    return os.path.basename(model_path)


@app.local_entrypoint()
def main():
    download_model_weights.remote()