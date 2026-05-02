import cv2
import numpy as np
import torch

# class to initiate and do depth esimation via MiDaS

class DepthEstimator:
    # initialize the device and model type
    def __init__(self, device: str | None = None, model_type: str = "DPT_Hybrid"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # load the model and transforms
        self.model = torch.hub.load("intel-isl/MiDaS", model_type).to(self.device).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        # if the model type is hybrid or large, use dpt_transform otherwise use small_transform
        if model_type in ("DPT_Hybrid", "DPT_Large"):
            self.transform = transforms.dpt_transform
        else:
            self.transform = transforms.small_transform

    @torch.inference_mode()
    def predict_depth(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Predict the depth of an image using the MiDaS model.
        image_rgb: np.ndarray - The input image in RGB format.
        returns: np.ndarray - The predicted depth map.
        """
        # keep same format as segmentation.py
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("Image must be a 3D array with shape (H, W, 3)")
        
        inp = self.transform(image_rgb).to(self.device)

        pred = self.model(inp)
        # if the pred depth map is 3D, unsqueeze it to 2D
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        # interpolate the predicted depth map to the original image size
        pred = torch.nn.functional.interpolate(pred, size=image_rgb.shape[:2], mode="bilinear", align_corners=False)
        depth = pred.squeeze().detach().float().cpu().numpy()
        return depth