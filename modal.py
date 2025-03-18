import os
import modal
from huggingface_hub import hf_hub_download

# =====================================================================
# SETUP SECTION
# =====================================================================
# Initialize the Modal application and storage volume
# The volume persists between runs and stores model weights
app = modal.App("sam-vit")
volume = modal.Volume.from_name("sam-weights-volume", create_if_missing=True)

# =====================================================================
# MODEL WEIGHTS DOWNLOAD
# =====================================================================
# Function to download the SAM (Segment Anything Model) weights from Hugging Face
@app.function(
    image=modal.Image.debian_slim().pip_install("huggingface-hub>=0.16.0"),
    volumes={"/weights": volume}
)
def download_model_weights():
    """
    Downloads the SAM model weights from Hugging Face and stores them in the volume.
    Returns the filename of the downloaded weights.
    """
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


# =====================================================================
# LOCAL ENTRY POINT
# =====================================================================
# Main function to run when executing this script directly
@app.local_entrypoint()
def main():
    """
    Entry point for local execution.
    Triggers the remote download of model weights.
    """
    download_model_weights.remote()


# =====================================================================
# ENVIRONMENT CONFIGURATION
# =====================================================================
# Define container image with all necessary dependencies
image = modal.Image.debian_slim().pip_install(
    "torch>=2.0.0",
    "numpy>=1.24.0",
    "segment-anything>=1.0",
    "opencv-python-headless>=4.7.0",
    "pillow>=9.0.0",
    "pydantic>=2.10.6",
    "torchvision>=0.21.0"
)

# =====================================================================
# DATA MODELS
# =====================================================================
# Note: Missing import for BaseModel and Field
# Should be: from pydantic import BaseModel, Field
class Point(BaseModel):
    """
    Represents a point in normalized coordinates (0-1) for input to the model.
    """
    x: float = Field(..., description="Normalized x-coordinate (0-1)")
    y: float = Field(..., description="Normalized y-coordinate (0-1)")

class SegmentationRequest(BaseModel):
    """
    Request model for image segmentation.
    Contains the image data and point coordinates to segment around.
    """
    image: str = Field(..., description="Base64 encoded image")
    point: Point = Field(..., description="Point coordinates (normalized)")
    mask_it: bool = Field(False, description="Whether to return masked image")

# =====================================================================
# MODEL PREDICTOR CLASS
# =====================================================================
# Note: Missing imports for base64, io, np, Image, Dict, Any
# Should include: import base64, import io, import numpy as np, from PIL import Image, from typing import Dict, Any
@app.cls(
    image=image,
    volumes={"/weights": volume},
    gpu="T4",  # Specifies GPU requirement for inference
)
class SamPredictor:
    """
    Class for running SAM model inference.
    Handles model loading and mask generation.
    """

    @modal.enter()
    def enter(self):
        """
        Initializes the model when the container starts.
        This method runs once when the container spins up, providing a warm start.
        """
        from segment_anything import sam_model_registry, SamPredictor
        checkpoint = "/weights/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        
        # Load the model and move it to GPU
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        self.predictor = SamPredictor(sam)
        self.predictor.model.to('cuda')

    @modal.method()
    def generate_mask(self, image_data: str, x: float, y: float, mask_it: bool = False) -> Dict[str, Any]:
        """
        Generates a segmentation mask based on the input image and point.
        
        Args:
            image_data: Base64 encoded image
            x: Normalized x-coordinate (0-1) of the point
            y: Normalized y-coordinate (0-1) of the point
            mask_it: Whether to return the masked image (True) or just the mask (False)
            
        Returns:
            Dictionary containing either the mask or masked image, and a success flag
        """
        # Decode the base64 image
        decoded_image = base64.b64decode(image_data)
        image = np.array(Image.open(io.BytesIO(decoded_image)))

        # Convert normalized coordinates to pixel coordinates
        height, width = image.shape[:2]
        original_x = int(x * width)
        original_y = int(y * height)
        point = np.array([[original_x, original_y]])
        
        # Generate the mask
        input_label = np.array([1])  # 1 indicates a foreground point
        self.predictor.set_image(image)
        masks, _, _ = self.predictor.predict(
            point_coords=point,
            point_labels=input_label,
            multimask_output=False,  # Only return the best mask
        )
        
        # Process the output based on mask_it flag
        if mask_it:
            # Apply the mask to the original image
            masked_image = np.copy(image)
            masked_image[masks.squeeze() == 0] = 0  # Set background pixels to black
            
            # Convert the masked image to base64
            pil_img = Image.fromarray(masked_image)
            buffered = io.BytesIO()
            pil_img.save(buffered, format="PNG")
            masked_img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            return {"masked_image": masked_img_str, "success": True}
        else:
            # Convert the binary mask to an image
            mask_img = (masks.squeeze() * 255).astype(np.uint8)  # Convert to 8-bit grayscale
            pil_mask = Image.fromarray(mask_img)
            buffered = io.BytesIO()
            pil_mask.save(buffered, format="PNG")
            mask_img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            return {"mask": mask_img_str, "success": True}
        
# =====================================================================
# API ENDPOINT FUNCTION
# =====================================================================
@app.function(
    image=image,
    volumes={"/weights": volume},
    gpu="T4",
)
def segment_image(request_data: SegmentationRequest) -> Dict[str, Any]:
    """
    Main API endpoint function for image segmentation.
    
    Args:
        request_data: The segmentation request containing image and point data
        
    Returns:
        Dictionary containing the segmentation result or error information
    """
    try:
        # Call the predictor class to generate the mask
        result = SamPredictor().generate_mask.remote(
            image_data=request_data.image,
            x=request_data.point.x,
            y=request_data.point.y,
            mask_it=request_data.mask_it
        )

        return result
    
    except Exception as e:
        return {"success": False, "error": str(e)}

# =====================================================================
# EXAMPLE USAGE (COMMENTED OUT)
# =====================================================================
# def segmentation (request) -> SegmentationResponse:
#     """
#     Example function showing how to use the segment_image function from another service.
#     
#     Args:
#         request: HTTP request containing segmentation parameters
#         
#     Returns:
#         HTTP response with the segmentation result or error information
#     """
#     try:
#         data = SegmentationRequest(**request)
#         segFunc = modal.Function.from_name('sam-vit', 'segment_image')
#         result = SegmentationResponse(**segFunc.remote(data))
#         if result.success:
#             return jsonify(result.model_dump_json()), 200
#         else:
#             return jsonify({'error': 'Internal server'}), 400
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400
