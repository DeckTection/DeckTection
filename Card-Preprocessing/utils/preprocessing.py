from perspective_utils import four_point_transform
from clahe_utils import apply_clahe

def preprocess_card(image, corners):
    """
    Full preprocessing pipeline:
    1. Applies perspective shift based on provided corners.
    2. Enhances contrast using CLAHE.
    
    Params:
        image (np.array): Input image (BGR).
        corners (np.array): 4x2 array of card corner coordinates.
    Returns:
        preprocessed_card (np.array): Preprocessed image.
    """
    warped = four_point_transform(image, corners)
    enhanced = apply_clahe(warped)
    return enhanced
