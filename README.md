# Posture Coach — Real-Time AI Posture Correction

A real-time posture coaching agent built with Vision Agents SDK for the Vision Possible: Agent Protocol hackathon.

## What it does
Uses YOLO pose detection + Gemini Live to watch your posture via webcam and give instant voice corrections. 
The agent analyzes your skeleton keypoints in real-time and speaks up when it detects slouching, 
forward head position, or uneven shoulders.

## Tech Stack
- Vision Agents SDK (Stream) — WebRTC edge network, real-time video
- YOLO11n-pose — skeleton/pose keypoint detection
- Gemini Realtime — multimodal LLM with voice output
- Python / uv

## Setup
1. Clone this repo
2. Add your `.env` file with `STREAM_API_KEY`, `STREAM_API_SECRET`, `GOOGLE_API_KEY`
3. `uv add "vision-agents[getstream,gemini,ultralytics]" python-dotenv`
4. `uv run main.py run`

## Use Case
Millions of desk workers develop chronic back and neck pain from poor posture. 
This agent acts as a persistent, real-time coach — no app to open, no manual checks. 
Just turn it on and it watches.