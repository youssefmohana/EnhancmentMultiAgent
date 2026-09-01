"""Generative agent - placeholder for GAN/Diffusion based augmentations, with OpenCV fallbacks."""
import time
import cv2
import numpy as np
from .base import BaseAgent, AgentResult


class GenerativeAgent(BaseAgent):
    """Generative augmentations (synthetic data, inpainting, style). Extensible to Stable Diffusion."""

    def __init__(self):
        super().__init__(name="generative", description="Generative augmentations (synthetic, inpainting, upscale)")
        self.capabilities = ["upscale", "inpaint", "synthetic_occlusion", "style_transfer_stub", "cutmix", "mixup_stub"]

    def get_available_operations(self):
        return [
            {"op": "upscale", "params": {"factor": "1.5..4.0"}, "use_case": "super-resolution robustness"},
            {"op": "inpaint", "params": {"mask_ratio": "0.1..0.3"}, "use_case": "occlusion handling"},
            {"op": "synthetic_occlusion", "params": {"count": "1..5"}, "use_case": "simulate missing parts"},
            {"op": "cutmix", "params": {"alpha": "0.4"}, "use_case": "regularization via patch mixing (needs pair)"},
            {"op": "elastic", "params": {"alpha": "30", "sigma": "4"}, "use_case": "deformation robustness"},
        ]

    def augment(self, input_path: str, output_path: str, operation: str = "upscale", **kwargs) -> AgentResult:
        start = time.time()
        try:
            img = self._load_image(input_path)
            h, w = img.shape[:2]

            if operation == "upscale":
                factor = float(kwargs.get("factor", 2.0))
                new_w, new_h = int(w*factor), int(h*factor)
                # High-quality Lanczos, placeholder for ESRGAN/Real-ESRGAN hook
                upscaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                # Downscale back to original to simulate SR training pair if needed
                if kwargs.get("keep_original_size", False):
                    result = cv2.resize(upscaled, (w, h), interpolation=cv2.INTER_LINEAR)
                else:
                    result = upscaled

            elif operation == "inpaint":
                # random mask inpainting
                mask = np.zeros((h,w), dtype=np.uint8)
                ratio = float(kwargs.get("mask_ratio", 0.15))
                # create random rectangles
                num_rects = max(1, int(ratio*5))
                for _ in range(num_rects):
                    x1 = np.random.randint(0, w//2)
                    y1 = np.random.randint(0, h//2)
                    rw = np.random.randint(w//8, w//3)
                    rh = np.random.randint(h//8, h//3)
                    cv2.rectangle(mask, (x1,y1), (x1+rw, y1+rh), 255, -1)
                result = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

            elif operation == "synthetic_occlusion":
                result = img.copy()
                count = int(kwargs.get("count", 2))
                for _ in range(count):
                    x = np.random.randint(0, w-20)
                    y = np.random.randint(0, h-20)
                    rw = np.random.randint(20, w//4)
                    rh = np.random.randint(20, h//4)
                    cv2.rectangle(result, (x,y), (x+rw, y+rh), (0,0,0), -1)

            elif operation == "cutmix":
                # Single-image cutmix placeholder - black patch mixing demonstration
                # Real cutmix requires second image; here we simulate with cropped patch from same image
                result = img.copy()
                ratio = float(kwargs.get("ratio", 0.3))
                cut_w, cut_h = int(w*ratio), int(h*ratio)
                x1 = np.random.randint(0, w-cut_w)
                y1 = np.random.randint(0, h-cut_h)
                x2 = np.random.randint(0, w-cut_w)
                y2 = np.random.randint(0, h-cut_h)
                patch = img[y2:y2+cut_h, x2:x2+cut_w]
                result[y1:y1+cut_h, x1:x1+cut_w] = patch

            elif operation == "elastic":
                # Simple elastic-like displacement via remap
                alpha = float(kwargs.get("alpha", 30))
                sigma = float(kwargs.get("sigma", 4))
                # generate displacement fields
                dx = cv2.GaussianBlur((np.random.rand(h,w)*2-1).astype(np.float32), (0,0), sigma) * alpha
                dy = cv2.GaussianBlur((np.random.rand(h,w)*2-1).astype(np.float32), (0,0), sigma) * alpha
                x, y = np.meshgrid(np.arange(w), np.arange(h))
                map_x = (x + dx).astype(np.float32)
                map_y = (y + dy).astype(np.float32)
                result = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

            else:
                return AgentResult(False, output_path, self.name, operation, kwargs, time.time()-start, f"Unknown operation: {operation}")

            self._save_image(result, output_path)
            return AgentResult(True, output_path, self.name, operation, kwargs, time.time()-start, f"{operation} -> {output_path}")

        except Exception as e:
            return AgentResult(False, output_path, self.name, operation, kwargs, time.time()-start, str(e))
