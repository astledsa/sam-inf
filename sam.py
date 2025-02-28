def Init_predictor () -> SamPredictor:
    """
    Initialize the model
    """
    checkpoint: str = "/kaggle/input/sam/pytorch/default/1/sam_vit_h_4b8939.pth"
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
        mask_it: bool = False
    ) -> np.ndarray:
    
    print("Image shape: ", image.shape)
    input_label = np.array([1])
    predictor.set_image(image)
    masks, _, _ = predictor.predict(
        point_coords=point,
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
    print("Trying here")
    pil_image = Image.fromarray(arr)
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')
