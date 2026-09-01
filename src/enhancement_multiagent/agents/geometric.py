"""Geometric agent - handles spatial augmentations."""
import time
import cv2
import numpy as np
from .base import BaseAgent, AgentResult


class GeometricAgent(BaseAgent):
    """Specialised agent for geometric transformations: flip, rotate, crop, affine, perspective."""

    def __init__(self):
        super().__init__(name="geometric", description="Geometric augmentations (flip, rotate, scale, crop, perspective)")
        self.capabilities = ["flip", "rotate", "scale", "crop", "affine", "perspective", "shear"]

    def get_available_operations(self):
        return [
            {"op": "flip", "params": {"direction": "horizontal|vertical|both"}, "use_case": "symmetry, doubling dataset"},
            {"op": "rotate", "params": {"angle": "-45..45", "border_mode": "reflect"}, "use_case": "rotation invariance"},
            {"op": "scale", "params": {"factor": "0.5..2.0"}, "use_case": "scale invariance"},
            {"op": "crop", "params": {"ratio": "0.7..1.0"}, "use_case": "focus on regions, small object simulation"},
            {"op": "affine", "params": {"shear": "-20..20"}, "use_case": "viewpoint robustness"},
            {"op": "perspective", "params": {"distortion": "0.0..0.5"}, "use_case": "camera angle variation"},
        ]

    def augment(self, input_path: str, output_path: str, operation: str = "flip", **kwargs) -> AgentResult:
        start = time.time()
        try:
            img = self._load_image(input_path)
            h, w = img.shape[:2]
            result = img

            if operation == "flip":
                direction = kwargs.get("direction", "horizontal")
                if direction == "horizontal":
                    result = cv2.flip(img, 1)
                elif direction == "vertical":
                    result = cv2.flip(img, 0)
                else:
                    result = cv2.flip(img, -1)

            elif operation == "rotate":
                angle = float(kwargs.get("angle", 15))
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                result = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

            elif operation == "scale":
                factor = float(kwargs.get("factor", 1.2))
                new_w, new_h = int(w * factor), int(h * factor)
                result = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                # crop or pad back to original size for consistency
                if factor > 1:
                    # center crop
                    cx, cy = new_w // 2, new_h // 2
                    result = result[cy - h//2:cy + h//2, cx - w//2:cx + w//2]
                else:
                    # pad
                    pad_w = (w - new_w)//2
                    pad_h = (h - new_h)//2
                    result = cv2.copyMakeBorder(result, pad_h, h - new_h - pad_h, pad_w, w - new_w - pad_w, cv2.BORDER_REFLECT_101)

            elif operation == "crop":
                ratio = float(kwargs.get("ratio", 0.85))
                crop_h, crop_w = int(h * ratio), int(w * ratio)
                y1 = (h - crop_h)//2
                x1 = (w - crop_w)//2
                cropped = img[y1:y1+crop_h, x1:x1+crop_w]
                result = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

            elif operation == "affine":
                shear = float(kwargs.get("shear", 10))
                # shear matrix
                M = np.float32([[1, np.tan(np.radians(shear)), 0], [0, 1, 0]])
                result = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

            elif operation == "perspective":
                distortion = float(kwargs.get("distortion", 0.2))
                # randomize corners slightly
                dw, dh = int(w * distortion), int(h * distortion)
                pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
                # fixed small perspective for reproducibility
                pts2 = np.float32([[dw, dh], [w-dw, dh//2], [dw//2, h-dh], [w-dw//2, h-dh//2]])
                M = cv2.getPerspectiveTransform(pts1, pts2)
                result = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

            else:
                return AgentResult(False, output_path, self.name, operation, kwargs, time.time()-start, f"Unknown operation: {operation}")

            self._save_image(result, output_path)
            return AgentResult(True, output_path, self.name, operation, kwargs, time.time()-start, f"{operation} -> {output_path}", {"input_shape": (h,w)})

        except Exception as e:
            return AgentResult(False, output_path, self.name, operation, kwargs, time.time()-start, str(e))
