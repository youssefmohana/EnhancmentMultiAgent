"""Photometric agent - handles color, lighting, blur, noise augmentations."""
import time
import cv2
import numpy as np
from .base import BaseAgent, AgentResult


class PhotometricAgent(BaseAgent):
    """Specialised agent for photometric augmentations targeting lighting/color weaknesses."""

    def __init__(self):
        super().__init__(name="photometric", description="Photometric augmentations (brightness, contrast, color, blur, noise)")
        self.capabilities = ["brightness", "contrast", "gamma", "color_jitter", "blur", "noise", "hsv_shift", "clahe"]

    def get_available_operations(self):
        return [
            {"op": "brightness", "params": {"gamma": "0.4..2.5"}, "use_case": "low-light / overexposure robustness"},
            {"op": "contrast", "params": {"clip_limit": "1.0..4.0"}, "use_case": "low-contrast scenes"},
            {"op": "color_jitter", "params": {"hue": "-10..10", "saturation": "0.5..1.5"}, "use_case": "color cast / white balance"},
            {"op": "blur", "params": {"kernel": "3..11"}, "use_case": "defocus / motion blur robustness"},
            {"op": "noise", "params": {"sigma": "5..35"}, "use_case": "sensor noise robustness"},
            {"op": "hsv_shift", "params": {"h_shift": "-5..5"}, "use_case": "illumination color variation"},
            {"op": "clahe", "params": {"clip_limit": "2.0"}, "use_case": "enhance local contrast"},
        ]

    def augment(self, input_path: str, output_path: str, operation: str = "brightness", **kwargs) -> AgentResult:
        start = time.time()
        try:
            img = self._load_image(input_path)

            if operation == "brightness":
                gamma = float(kwargs.get("gamma", 1.5))
                inv = 1.0 / max(gamma, 0.1)
                table = np.array([((i/255.0)**inv)*255 for i in range(256)]).astype(np.uint8)
                result = cv2.LUT(img, table)

            elif operation == "contrast":
                # CLAHE based
                clip = float(kwargs.get("clip_limit", 2.0))
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l,a,b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8))
                l = clahe.apply(l)
                result = cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR)

            elif operation == "color_jitter":
                hue_shift = int(kwargs.get("hue", 5))
                sat_scale = float(kwargs.get("saturation", 1.2))
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:,:,0] = (hsv[:,:,0] + hue_shift) % 180
                hsv[:,:,1] = np.clip(hsv[:,:,1]*sat_scale, 0, 255)
                result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

            elif operation == "blur":
                k = int(kwargs.get("kernel", 5))
                if k % 2 == 0: k += 1
                result = cv2.GaussianBlur(img, (k,k), 0)

            elif operation == "noise":
                sigma = float(kwargs.get("sigma", 15))
                noise = np.random.normal(0, sigma, img.shape)
                result = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            elif operation == "hsv_shift":
                h_shift = int(kwargs.get("h_shift", 3))
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
                hsv[:,:,0] = (hsv[:,:,0] + h_shift) % 180
                result = cv2.cvtColor(np.clip(hsv,0,255).astype(np.uint8), cv2.COLOR_HSV2BGR)

            elif operation == "clahe":
                clip = float(kwargs.get("clip_limit", 2.0))
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l,a,b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8))
                l = clahe.apply(l)
                result = cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR)

            elif operation == "sharpen":
                kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
                result = cv2.filter2D(img, -1, kernel)

            elif operation == "denoise":
                result = cv2.fastNlMeansDenoisingColored(img, None, 10,10,7,21)

            else:
                return AgentResult(False, output_path, self.name, operation, kwargs, time.time()-start, f"Unknown operation: {operation}")

            self._save_image(result, output_path)
            return AgentResult(True, output_path, self.name, operation, kwargs, time.time()-start, f"{operation} -> {output_path}")

        except Exception as e:
            return AgentResult(False, output_path, self.name, operation, kwargs, time.time()-start, str(e))
