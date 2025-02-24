import base64
import requests
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

def Init_predictor () -> SamPredictor:
    """
    Initialize the model
    """
    checkpoint: str = "models/sam_vit_h_4b8939.pth"
    model_type: str = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    return SamPredictor(sam)

def normalized_to_original(width: float, height: float, x: float, y: float) -> np.array:
    """
    The moondream API generally returns a normalized point.
    Therefore it needs to be converted to the original dimensions
    """
    original_x = int(x * width)
    original_y = int(y * height)
    return np.array([[original_x, original_y]])

def Masked (
        predictor: SamPredictor,
        image: np.ndarray,
        point: np.array,
        normalize: bool = False,
        mask_it: bool = False
    ) -> np.ndarray:
   """
   Returns a mask of the image. If mask_it is set to true, this functions applies the
   mask to the image, ans returns the new image.
   """
   input_point = normalized_to_original(image.shape[0], image.shape[1], point[0][0], point[0][1]) if normalize else point
   input_label = np.array([1])

   predictor.set_image(image)
   masks, _, _ = predictor.predict(
      point_coords=input_point,
      point_labels=input_label,
      multimask_output=False,
    )

   if mask_it:
      masked_image = np.copy(image)
      masked_image[masks.squeeze() == 0] = 0
      return masked_image
   else:
       return masks


def image_array_to_base64 (arr: np.array) -> str:
    """
    Converts a numpy array to base64 string
    """
    pil_image = Image.fromarray(arr)
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')
