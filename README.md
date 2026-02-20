# VisualMemory

Visual memory backend for blind users. Detects, remembers, and locates objects with distance + direction narration.

## Setup

```bash
pip install -e .
python setup_weights.py
```

## Modes

- **Remember Mode** — detect + store an object from a text prompt
- **Scan Mode** — find all remembered objects in a new image, return distance + clock position
