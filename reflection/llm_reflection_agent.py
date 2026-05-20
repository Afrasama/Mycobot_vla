import base64
import io
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib import error, request

import numpy as np

from reflection.reflection_agent import ReflectionAgent
from utils.logger import log_failure


DEFAULT_POLICY_LIMITS = {
    "x_offset": (-0.02, 0.02),
    "y_offset": (-0.02, 0.02),
    "grasp_height": (-0.04, 0.06),
    "approach_height": (0.06, 0.20),
    "lift_height": (0.10, 0.30),
    "release_delay": (0, 180),
}

DEFAULT_STEP_LIMITS = {
    "x_offset": 0.02,
    "y_offset": 0.02,
    "grasp_height": 0.015,
    "approach_height": 0.03,
    "lift_height": 0.04,
    "release_delay": 45,
}

DECISION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string"},
        "updates": {
            "type": "object",
            "properties": {
                "x_offset": {"type": "number"},
                "y_offset": {"type": "number"},
                "grasp_height": {"type": "number"},
                "approach_height": {"type": "number"},
                "lift_height": {"type": "number"},
                "release_delay": {"type": "number"},
            },
            "additionalProperties": False,
        },
        "terminate": {"type": "boolean"},
        "confidence": {"type": "number"},
    },
    "required": ["explanation", "updates", "terminate"],
    "additionalProperties": False,
}


@dataclass
class LLMDecision:
    explanation: str
    updates: Dict[str, float]
    terminate: bool = False
    confidence: Optional[float] = None
    mode: str = "fallback"
    raw_text: Optional[str] = None


def apply_policy_updates(
    policy: Dict[str, float],
    updates: Dict[str, float],
    limits: Optional[Dict[str, tuple]] = None,
) -> Dict[str, float]:
    limits = limits or DEFAULT_POLICY_LIMITS
    new_policy = dict(policy)

    for key, delta in updates.items():
        if key not in new_policy:
            continue

        value = new_policy[key] + delta
        if key in limits:
            lo, hi = limits[key]
            value = min(max(value, lo), hi)

        if key == "release_delay":
            value = int(round(value))

        new_policy[key] = value

    return new_policy


class LLMReflectionAgent:
    def __init__(
        self,
        model: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        backend: Optional[str] = None,
        timeout_s: Optional[float] = None,
        use_vision: Optional[bool] = None,
        policy_limits: Optional[Dict[str, tuple]] = None,
        step_limits: Optional[Dict[str, float]] = None,
        url: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_AGENT_API_KEY")
        self.backend = (backend or os.getenv("LLM_AGENT_BACKEND") or self._default_backend()).lower()
        self.vision_model = "llama3.2-vision"
        self.reasoning_model = "llama3"
        self.model = model or os.getenv("LLM_AGENT_MODEL") or self.vision_model
        self.endpoint = url or endpoint or os.getenv("LLM_AGENT_ENDPOINT") or self._default_endpoint()
        self.timeout_s = self._resolve_timeout(timeout_s)
        self.use_vision = self._resolve_use_vision(use_vision)
        self.policy_limits = policy_limits or DEFAULT_POLICY_LIMITS
        self.step_limits = step_limits or DEFAULT_STEP_LIMITS
        # Swap axes = True because image X corresponds to World Y in some PyBullet setups, but actually we use vision LM now.
        self.fallback_agent = ReflectionAgent(scale=0.00015, max_step=0.02, swap_axes=False)

    @staticmethod
    def _resolve_timeout(timeout_s: Optional[float]) -> float:
        if timeout_s is not None:
            return float(timeout_s)
        env_value = os.getenv("LLM_AGENT_TIMEOUT_S")
        if env_value:
            try:
                return float(env_value)
            except ValueError:
                pass
        return 20.0

    @staticmethod
    def _resolve_use_vision(use_vision: Optional[bool]) -> bool:
        if use_vision is not None:
            return bool(use_vision)
        return os.getenv("LLM_AGENT_USE_VISION", "1") == "1"

    def _default_backend(self) -> str:
        return "ollama"

    def _default_model(self) -> str:
        return "llama3.2-vision"

    def _default_endpoint(self) -> str:
        return "http://localhost:11434/api/chat"

    def is_configured(self) -> bool:
        if not self.endpoint:
            return False
        if self.backend == "ollama":
            return True
        return False

    def reflect(
        self,
        scene_info: Dict[str, Any],
        policy: Dict[str, float],
        rgb: Optional[np.ndarray] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMDecision:
        if self.is_configured():
            try:
                decision = self._query_llm(scene_info, policy, rgb=rgb, history=history or [])
                decision.updates = self._sanitize_updates(decision.updates)
                return decision
            except Exception as exc:
                fallback = self._fallback(scene_info)
                fallback.explanation = (
                    f"LLM call failed, using fallback heuristic: {exc}. "
                    f"{fallback.explanation}"
                )
                return fallback

        return self._fallback(scene_info)

    def _fallback(self, scene_info: Dict[str, Any]) -> LLMDecision:
        pixel_error_x = float(scene_info.get("pixel_error_x", 0.0))
        pixel_error_y = float(scene_info.get("pixel_error_y", 0.0))
        result = self.fallback_agent.reflect(
            {
                "pixel_error_x": pixel_error_x,
                "pixel_error_y": pixel_error_y,
                "retry_count": int(scene_info.get("retry_count", 0)),
            }
        )

        updates = {
            "x_offset": float(result["action"]["adjust_x"]),
            "y_offset": float(result["action"]["adjust_y"]),
        }

        failure_type = scene_info.get("failure_type", "unknown")
        if failure_type == "placement_failure":
            updates["release_delay"] = 15

        return LLMDecision(
            explanation=result["explanation"],
            updates=self._sanitize_updates(updates),
            terminate=False,
            confidence=None,
            mode="fallback",
            raw_text=None,
        )

    def _query_llm(
        self,
        scene_info: Dict[str, Any],
        policy: Dict[str, float],
        rgb: Optional[np.ndarray],
        history: List[Dict[str, Any]],
    ) -> LLMDecision:
        if self.backend == "ollama":
            return self._query_ollama(scene_info, policy, rgb, history)
        raise RuntimeError(f"Unsupported backend: {self.backend}")

    def _query_ollama(self, scene_info, policy, rgb, history) -> LLMDecision:
        # Step 1: Visual Analysis (if vision is enabled and RGB is provided)
        vision_analysis = "No visual data available."
        if self.use_vision and rgb is not None:
            vision_analysis = self._analyze_vision_ollama(rgb)
            print(f"Vision Analysis: {vision_analysis}")

        # Step 2: Reasoning & Policy Update
        prompt = self._build_prompt(scene_info, policy, history, vision_analysis)
        user_message = {"role": "user", "content": prompt}

        payload = {
            "model": self.reasoning_model,
            "stream": False,
            "format": DECISION_JSON_SCHEMA,
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                user_message,
            ],
        }
        headers = {"Content-Type": "application/json"}
        response_text = self._post_json(self.endpoint, payload, headers, label="ollama reasoning")
        data = json.loads(response_text)
        decision = self._parse_decision(data["message"]["content"], mode="ollama")
        self._log_llm_decision(scene_info, policy, decision)
        return decision

    def _analyze_vision_ollama(self, rgb) -> str:
        image_b64 = self._rgb_to_base64_jpeg(rgb)
        if not image_b64:
            return "Failed to encode image."

        prompt = (
            "You are a robotic vision system analyzing an overhead camera view of a robotic arm grasping a cube. "
            "Describe the position of the gripper relative to the cube. Is it to the left, right, above, or below the cube? "
            "Is the gripper open or closed? Keep it to 2 sentences max."
        )
        
        payload = {
            "model": self.vision_model,
            "stream": False,
            "options": {"temperature": 0},
            "messages": [
                {
                    "role": "user", 
                    "content": prompt,
                    "images": [image_b64]
                }
            ],
        }
        headers = {"Content-Type": "application/json"}
        try:
            response_text = self._post_json(self.endpoint, payload, headers, label="ollama vision")
            data = json.loads(response_text)
            return data["message"]["content"]
        except Exception as e:
            return f"Vision model failed: {e}"

    def _log_llm_decision(
        self,
        scene_info: Dict[str, Any],
        policy: Dict[str, float],
        decision: LLMDecision,
    ) -> None:
        strategy_chosen = "abort" if decision.terminate else "retry_with_policy_update"
        robot_state = {
            "scene_info": scene_info,
            "current_policy": policy,
        }
        llm_response = {
            "mode": decision.mode,
            "explanation": decision.explanation,
            "updates": decision.updates,
            "terminate": decision.terminate,
            "confidence": decision.confidence,
            "raw_text": decision.raw_text,
        }
        log_failure(
            failure_type=str(scene_info.get("failure_type", "unknown")),
            robot_state=robot_state,
            llm_response=llm_response,
            strategy_chosen=strategy_chosen,
        )

    def _parse_decision(self, raw_text: str, mode: str) -> LLMDecision:
        parsed = json.loads(raw_text)
        return LLMDecision(
            explanation=str(parsed.get("explanation", "")).strip() or "No explanation provided",
            updates=dict(parsed.get("updates", {})),
            terminate=bool(parsed.get("terminate", False)),
            confidence=self._to_optional_float(parsed.get("confidence")),
            mode=mode,
            raw_text=raw_text,
        )

    def _post_json(self, base_url, payload, headers, label: str, endpoint: str = "") -> str:
        full_url = base_url + endpoint
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(full_url, data=body, headers=headers, method="POST")
        start_time = time.perf_counter()
        try:
            with request.urlopen(req, timeout=self.timeout_s) as resp:
                response_text = resp.read().decode("utf-8")
                elapsed = time.perf_counter() - start_time
                print(f"{label} call completed in {elapsed:.2f}s (timeout={self.timeout_s:.1f}s, vision={self.use_vision})")
                return response_text
        except error.HTTPError as exc:
            elapsed = time.perf_counter() - start_time
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{label} call failed after {elapsed:.2f}s with HTTP {exc.code}: {details}") from exc
        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            raise RuntimeError(f"{label} call failed after {elapsed:.2f}s: {exc}") from exc

    def _system_prompt(self) -> str:
        return (
            "You are a robot recovery agent. "
            "You MUST respond with ONLY valid JSON. No explanations, no text, just JSON. "
            "Format: {\"explanation\": \"brief reason\", \"updates\": {\"param\": delta}, \"terminate\": false, \"confidence\": 0.8} "
            "You may change x_offset, y_offset, grasp_height, approach_height, lift_height, and release_delay. "
            "Use small deltas (0.01-0.05 range), not absolute values."
        )

    def _build_prompt(self, scene_info, policy, history, vision_analysis) -> str:
        compact_history = history[-3:]
        request_json = {
            "scene_info": scene_info,
            "current_policy": policy,
            "policy_limits": self.policy_limits,
            "max_delta_per_step": self.step_limits,
            "recent_attempts": compact_history,
            "visual_analysis": vision_analysis,
            "required_output_schema": DECISION_JSON_SCHEMA,
        }
        return (
            "Analyze the robot failure and propose the next retry policy.\n\n"
            "CRITICAL: You MUST respond with ONLY JSON. No extra text, explanations, or formatting.\n"
            "Example response: {\"explanation\": \"gripper too high\", \"updates\": {\"grasp_height\": -0.02}, \"terminate\": false, \"confidence\": 0.7}\n\n"
            "Rules:\n"
            "- Response must be valid JSON only\n"
            "- All update values are deltas (changes), not absolute values\n"
            "- Use very small deltas: 0.005-0.02 range for gentle adjustments\n"
            "- If no changes needed, use empty updates: {}\n"
            "- Set terminate=true only if retrying won't help\n"
            "- confidence should be 0.1-1.0\n\n"
            f"Context:\n{json.dumps(request_json, indent=2)}\n\n"
            "JSON Response:"
        )

    def _sanitize_updates(self, updates: Dict[str, Any]) -> Dict[str, float]:
        clean_updates: Dict[str, float] = {}
        for key, raw_value in updates.items():
            if key not in self.step_limits:
                continue
            value = self._to_optional_float(raw_value)
            if value is None:
                continue
            step_limit = float(self.step_limits[key])
            value = float(np.clip(value, -step_limit, step_limit))
            if key == "release_delay":
                value = int(round(value))
            clean_updates[key] = value
        return clean_updates

    @staticmethod
    def _to_optional_float(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _rgb_to_base64_jpeg(rgb: Optional[np.ndarray], max_size=(256, 256)) -> Optional[str]:
        if rgb is None:
            return None
        try:
            from PIL import Image
        except Exception:
            return None
        image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
        # Use bilinear resampling for speed and further reduce the image size/quality
        # to ensure LLM processing is extremely fast
        image.thumbnail(max_size, Image.Resampling.BILINEAR)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=50)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _rgb_to_openai_image_url(self, rgb: Optional[np.ndarray]) -> Optional[str]:
        image_b64 = self._rgb_to_base64_jpeg(rgb)
        if image_b64 is None:
            return None
        return f"data:image/jpeg;base64,{image_b64}"
