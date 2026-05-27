import os
import json
import base64
import io
import time
from typing import Any, Dict, Optional, Tuple
import numpy as np

def _normalize_vla_backend(backend: Optional[str]) -> str:
    """Map .env backend names to internal modes."""
    raw = (backend or os.getenv("VLA_BACKEND", "heuristic")).strip().lower()
    aliases = {
        "simulation": "heuristic",
        "sim": "heuristic",
        "local_heuristic": "heuristic",
        "closed_loop": "heuristic",
        "openvla": "openvla_local",
        "openvla-api": "openvla_api",
        "api": "openvla_api",
        "local": "neural_local",
        "neural": "neural_local",
        "neural_local": "neural_local",
        "openvla_local": "openvla_local",
        "openvla_api": "openvla_api",
    }
    return aliases.get(raw, raw)


class VLARecoveryAgent:
    def __init__(
        self,
        backend: Optional[str] = None,
        api_url: Optional[str] = None,
        model_path: Optional[str] = None,
    ):
        self.backend = _normalize_vla_backend(backend)
        self.api_url = api_url or os.getenv("VLA_API_URL", "http://localhost:8000/predict")
        self.model_path = (model_path or os.getenv("VLA_MODEL_PATH", "")).strip()
        self.allow_openvla = os.getenv("VLA_USE_OPENVLA", "0") == "1"

        print(f"[VLA AGENT] Initializing VLARecoveryAgent with backend: {self.backend}")

        self.model = None
        self.processor = None

        if self.backend == "heuristic":
            print(
                "[VLA AGENT] Local closed-loop heuristic (overhead vision + world error). "
                "No OpenVLA weights loaded."
            )
        elif self.backend == "openvla_local":
            if not self.allow_openvla:
                print(
                    "[VLA AGENT] OpenVLA local requested but VLA_USE_OPENVLA=0. "
                    "Using heuristic closed-loop instead."
                )
                self.backend = "heuristic"
            else:
                if not self.model_path:
                    self.model_path = "openvla/openvla-7b"
                self._init_local_model()
        elif self.backend == "neural_local":
            if not self.model_path:
                print(
                    "[VLA AGENT] neural_local requires VLA_MODEL_PATH "
                    "(HuggingFace id or local checkpoint directory)."
                )
                self.backend = "heuristic"
            else:
                self._init_local_model()
        elif self.backend == "openvla_api":
            if not self.allow_openvla:
                print(
                    "[VLA AGENT] OpenVLA API requested but VLA_USE_OPENVLA=0. "
                    "Using heuristic closed-loop instead."
                )
                self.backend = "heuristic"
            else:
                print(f"[VLA AGENT] OpenVLA API endpoint: {self.api_url}")
        elif self.backend not in ("heuristic",):
            print(f"[VLA AGENT] Unknown backend '{self.backend}'. Using heuristic closed-loop.")
            self.backend = "heuristic"

    def _init_local_model(self):
        """Initializes the local VLA model on GPU if possible."""
        print(f"[VLA AGENT] Attempting to load local VLA model from path: {self.model_path} on GPU...")
        try:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor
            
            # Check for CUDA availability
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[VLA AGENT] PyTorch device detected: {device}")
            
            # Load weights (using bfloat16 or float16 if GPU is available to save memory)
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            
            # Note: We load in 4-bit/8-bit or with low-cpu-mem-usage if running locally
            print("[VLA AGENT] Loading AutoProcessor...")
            self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
            
            print(f"[VLA AGENT] Loading model weights (dtype={torch_dtype})...")
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_path,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).to(device)
            print("[VLA AGENT] Local VLA model loaded successfully!")
        except ImportError as e:
            print(f"[VLA AGENT WARNING] Missing transformers/torch dependencies: {e}. Falling back to heuristic mode.")
            self.backend = "heuristic"
        except Exception as e:
            print(f"[VLA AGENT WARNING] Failed to load neural VLA model: {e}. Falling back to heuristic mode.")
            self.backend = "heuristic"

    def predict_action(
        self,
        rgb: Optional[np.ndarray],
        text_instruction: str,
        relative_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Predicts the next action step to correct robot alignment.
        Returns:
            dict: {"dx": float, "dy": float, "dz": float, "gripper_close": bool, "terminate": bool}
        """
        if self.backend == "heuristic":
            return self._predict_heuristic(relative_state)
        if self.backend == "openvla_api":
            return self._predict_api(rgb, text_instruction, relative_state)
        if self.backend in ("openvla_local", "neural_local"):
            return self._predict_local(rgb, text_instruction, relative_state)
        return self._predict_heuristic(relative_state)

    def _predict_heuristic(self, relative_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Local closed-loop recovery (default): overhead camera + world-frame error.
        No OpenVLA / HuggingFace model required.
        """
        if not relative_state:
            return {"dx": 0.0, "dy": 0.0, "dz": 0.0, "gripper_close": False, "terminate": True}
            
        pixel_err_x = float(relative_state.get("pixel_error_x", 0.0))
        pixel_err_y = float(relative_state.get("pixel_error_y", 0.0))

        # Prefer live world-frame error when available (avoids camera/world Y sign mistakes).
        gripper_pos = relative_state.get("gripper_pos")
        cube_pos = relative_state.get("cube_pos")
        if gripper_pos is not None and cube_pos is not None:
            phys_dx = float(cube_pos[0]) - float(gripper_pos[0])
            phys_dy = float(cube_pos[1]) - float(gripper_pos[1])
        else:
            try:
                from perception.segmentation import get_overhead_pixels_per_meter

                ppm = get_overhead_pixels_per_meter()
            except Exception:
                ppm = 500.0
            phys_dx = pixel_err_x / ppm
            phys_dy = -pixel_err_y / ppm
        true_xy_error_mag = float(np.sqrt(phys_dx**2 + phys_dy**2))

        # Scale step with distance (faster convergence when far; cap for stability)
        max_step = float(np.clip(0.35 * true_xy_error_mag, 0.012, 0.022))
        dx = float(np.clip(phys_dx, -max_step, max_step))
        dy = float(np.clip(phys_dy, -max_step, max_step))
        dz = 0.0
        gripper_close = False
        terminate = False

        tcp_pos = relative_state.get("gripper_pos", [0.0, 0.0, 0.1])
        cube_pos = relative_state.get("cube_pos", [0.0, 0.0, 0.02])
        z_error = float(tcp_pos[2] - cube_pos[2])

        if z_error > 0.03:
            dz = -0.008
        elif true_xy_error_mag > 0.012:
            dz = 0.0
        elif z_error > 0.002:
            dz = -0.008
        elif true_xy_error_mag <= 0.012:
            gripper_close = True
            
        return {
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "gripper_close": gripper_close,
            "terminate": terminate,
            "explanation": (
                f"Local heuristic VLA: XY error={true_xy_error_mag * 1000:.1f}mm, "
                f"Z error={z_error * 1000:.1f}mm. dx={dx:.4f}, dy={dy:.4f}, dz={dz:.4f}"
            ),
        }

    def _predict_api(
        self,
        rgb: Optional[np.ndarray],
        text_instruction: str,
        relative_state: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Queries a hosted FastAPI/Docker OpenVLA endpoint."""
        if rgb is None:
            print("[VLA AGENT] No visual input provided for API query. Falling back to heuristic.")
            return self._predict_heuristic(relative_state)
            
        try:
            from PIL import Image
            import urllib.request
            import urllib.error
            
            # Encode image to Base64 JPEG
            pil_img = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
            # Downscale slightly to improve query speed
            pil_img.thumbnail((256, 256))
            buffer = io.BytesIO()
            pil_img.save(buffer, format="JPEG", quality=60)
            img_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
            
            payload = {
                "image": img_b64,
                "instruction": text_instruction,
                "relative_state": relative_state or {}
            }
            
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.api_url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            
            # Set a 5-second timeout for rapid loop response
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                resp_text = resp.read().decode("utf-8")
                data = json.loads(resp_text)
                
            return {
                "dx": float(data.get("dx", 0.0)),
                "dy": float(data.get("dy", 0.0)),
                "dz": float(data.get("dz", 0.0)),
                "gripper_close": bool(data.get("gripper_close", False)),
                "terminate": bool(data.get("terminate", False)),
                "explanation": str(data.get("explanation", "API prediction"))
            }
            
        except Exception as e:
            print(f"[VLA AGENT WARNING] API query failed: {e}. Falling back to heuristic.")
            return self._predict_heuristic(relative_state)

    def _predict_local(
        self,
        rgb: Optional[np.ndarray],
        text_instruction: str,
        relative_state: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Performs forward pass on the local GPU-loaded model."""
        if self.model is None or rgb is None:
            return self._predict_heuristic(relative_state)
            
        try:
            import torch
            from PIL import Image
            
            pil_img = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
            
            # Format model input prompt
            inputs = self.processor(text=text_instruction, images=pil_img, return_tensors="pt")
            device = next(self.model.parameters()).device
            
            # Move tensors to the correct device
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)
                    
            start_t = time.perf_counter()
            with torch.no_grad():
                # Perform model prediction
                outputs = self.model.generate(**inputs, max_new_tokens=20)
                
            decoded = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            elapsed = time.perf_counter() - start_t
            print(f"[VLA AGENT] Local GPU forward pass completed in {elapsed:.3f}s. Response: {decoded}")
            
            # Parse decoded action tokens (depends on VLA tokenization, e.g. OpenVLA outputs actions wrapped in brackets)
            # Here we parse action tokens: e.g. "Action: [0.01, -0.02, -0.005, 0]"
            # If parsing fails, fall back gracefully to heuristic relative corrections.
            action_vector = self._parse_vla_text_response(decoded)
            if action_vector:
                dx, dy, dz, gripper_close = action_vector
                return {
                    "dx": dx,
                    "dy": dy,
                    "dz": dz,
                    "gripper_close": gripper_close,
                    "terminate": False,
                    "explanation": f"Local VLA Output: {decoded}"
                }
                
        except Exception as e:
            print(f"[VLA AGENT WARNING] Local forward pass failed: {e}. Falling back to heuristic.")
            
        return self._predict_heuristic(relative_state)

    def _parse_vla_text_response(self, text: str) -> Optional[Tuple[float, float, float, bool]]:
        """Parses model action outputs from raw generated text."""
        try:
            import re
            # Search for numbers in bracket lists like [0.01, -0.02, -0.005, 1.0]
            match = re.search(r"\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]", text)
            if match:
                vals = [float(match.group(i)) for i in range(1, 5)]
                # Map to [dx, dy, dz, gripper_close]
                # Scale from normalized VLA space [-1, 1] to physical joint/TCP steps (max 1.2cm step)
                scale_factor = 0.012
                dx = vals[0] * scale_factor
                dy = vals[1] * scale_factor
                dz = vals[2] * scale_factor
                gripper_close = vals[3] > 0.5
                return dx, dy, dz, gripper_close
        except Exception:
            pass
        return None
