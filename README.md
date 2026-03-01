# Setup
**Prerequisites:** Python >= 3.10, Hugging Face account

**Request access:**
- [DINOv3 model](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
- [Grounding DINO model](https://huggingface.co/IDEA-Research/grounding-dino-base)

```bash
git clone https://github.com/tsa-softwaredev-26/TSA-soft-dev-2026.git
cd TSA-soft-dev-2026
pip install --upgrade pip
pip install -e .
python setup_weights.py
huggingface-cli login
```

## Modes

- **Remember Mode** — detect + store an object from a text prompt
- **Scan Mode** — find all remembered objects in a new image, return distance + relative position

## Running tests

```bash
# Full integration test (run after any engine/pipeline change)
python -m visual_memory.tests.scripts.run_tests

# Show OCR text vs ground truth
VERBOSE=1 python -m visual_memory.tests.scripts.run_tests
```
