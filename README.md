# Setup
**Prerequisites:** Python >= 3.10, Hugging Face account

**Request access:**
- [Grounding DINO model](https://huggingface.co/IDEA-Research/grounding-dino-base)


```bash
git clone https://github.com/tsa-softwaredev-26/TSA-soft-dev-backend-2026.git
cd TSA-soft-dev-backend-2026
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
hf auth login
python setup_weights.py
```
## Modes

- **Remember Mode** - detect + store an object from a text prompt
- **Scan Mode** - find all remembered objects in a new image, return distance + relative position

## Running tests

```bash
# Full integration test (run after any engine/pipeline change)
# Most models download on first use, expect longer first run
python -m visual_memory.tests.scripts.run_tests

# Show OCR text vs ground truth
VERBOSE=1 python -m visual_memory.tests.scripts.run_tests
```
