# Setup to run locally (temporary)
**Prerequisites:** Python >= 3.8, Hugging Face account 

**Request access:**  
- [Dinov3 model](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
- [Grounding DINO model](https://huggingface.co/docs/transformers/en/model_doc/grounding-dino)

**Paste into terminal:**
```bash
git clone https://github.com/tsa-softwaredev-26/TSA-soft-dev-2026.git
cd TSA-soft-dev-26
pip install --upgrade pip
pip install -e .
huggingface-cli login
