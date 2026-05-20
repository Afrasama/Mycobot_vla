# Project Report: Multimodal Agentic "Reflection" for Robotics Error Recovery

## 1. Title
**Multimodal Agentic "Reflection" for Robotics Error Recovery**

---

## 2. Problem Statement
Despite significant advancements in industrial automation, robotic systems operating in unstructured or semi-structured environments remain highly susceptible to execution errors. Traditional error-handling mechanisms rely on rigid, pre-defined exception rules that fail to account for the nuanced physical realities of a failure (e.g., a slight slippage of a workpiece, lighting variations, or kinematic inaccuracies). When a robot fails a task, it typically requires human intervention or a blind restart, neither of which facilitates long-term autonomy. There is a critical need for a system that can "self-reflect"—autonomously analyzing visual and tactile feedback to understand *why* a failure occurred and dynamically adjusting its control parameters to ensure success in subsequent attempts.

---

## 3. Objectives
The primary objectives of this project are:
- **Autonomous Error Diagnostics**: To develop a multimodal agent capable of interpreting visual data (overhead camera) and internal state logs to diagnose failure types (e.g., grasp failure, collision, or unstable placement).
- **Closed-Loop Reflection**: To implement a "Reflection" loop where a Large Language Model (LLM) or Vision-Language Model (VLM) acts as a high-level reasoning agent to propose corrective policy updates.
- **Dynamic Parameter Optimization**: To enable the robot to adjust its kinematics parameters (offsets, grasp heights) in real-time based on agentic feedback.

---

## 4. Motivation
The motivation behind this work stems from the desire to bridge the gap between high-level cognitive reasoning and low-level robotic execution. By imbuing robots with the ability to "think" about their mistakes, we move closer to truly resilient autonomous systems. This reduces downtime in manufacturing, lowers the cost of programming complex tasks, and allows robots to learn from experience in a manner similar to human practitioners.

---

## 5. Proposed Methodology
The project implements a cyclical **Trial-Reflect-Retry** architecture:
1.  **Execution Phase**: The robot (MyCobot in PyBullet simulation) attempts a pick-and-place task using a baseline kinematic policy.
2.  **Perception & Failure Detection**: If the task fails, an overhead camera captures an RGB image of the scene, and the system logs the failure type and pixel error.
3.  **Multimodal Reflection**: 
    - The scene info and visual data are passed to the **LLMReflectionAgent**.
    - The agent supports **Ollama** (for local Llama 3.2) as the primary backend for local LLM execution.
    - The LLM performs "Reflection," generating a natural language explanation and a structured JSON update for the robot's policy.
4.  **Policy Adaptation**: The robot applies the suggested updates (e.g., adjusting `x_offset` or `grasp_height`).
5.  **Recovery Phase**: The robot retries the task with the refined parameters, closing the loop.

---

## 6. Results
The system was evaluated using the **Improved Kinematics Reflection** experiment. 
- **Baseline Performance**: Initial attempts often failed due to fixed kinematic errors, with an average distance to goal of ~0.12m.
- **Reflection Performance**: Upon engaging the LLM agent, the system successfully identified grasp failures. 
- **Key Outcome**: In a representative trial using the Ollama/Llama 3.2 backend, the robot reduced its final distance to goal from **0.112m** in the first attempt to **0.066m** by the fifth attempt, successfully completing the placement task.

---

## 7. Conclusion
This project demonstrates that integrating Multimodal LLMs into the robotic control loop through an "Agentic Reflection" mechanism significantly enhances error recovery. By moving away from hard-coded recovery paths and towards reasoned adaptations, the system exhibits higher resilience and a capacity for self-improvement. Future work will focus on integrating real-time tactile feedback and expanding the reflection agent to handle more complex multi-step manipulation tasks.

---

## 8. References
1.  **PyBullet**: A Python module for physics simulation for games, robotics, and machine learning.
2.  **Ollama**: Local execution framework for Large Language Models.
3.  **Ollama**: Local execution framework for Large Language Models.
