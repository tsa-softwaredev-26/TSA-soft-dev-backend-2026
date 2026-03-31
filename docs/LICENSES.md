# Third-Party Model Attributions and Licenses

This project uses third-party ML models and model-adjacent assets. This file records attribution, source links, citation links, and license terms for each model currently wired in code.

Scope:
- Core backend models loaded from Hugging Face or local checkpoints
- OCR microservice model stack
- Ollama LLM used for query parsing and fallback prompt generation

If model IDs or checkpoints change, update this file and `CITATIONS.bib` in the same change.

## Model inventory

| Component | Model or Asset | Where used | Source repo | Model page | License |
|---|---|---|---|---|---|
| Prompt-based detection | `IDEA-Research/grounding-dino-base` | `src/visual_memory/engine/object_detection/prompt_based.py` | https://github.com/IDEA-Research/GroundingDINO | https://huggingface.co/IDEA-Research/grounding-dino-base | Apache-2.0 |
| Prompt-free detection | `yoloe-26l-seg-pf.pt` | `src/visual_memory/engine/object_detection/detect_all.py` | https://github.com/ultralytics/ultralytics | https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26l-seg-pf.pt | AGPL-3.0 (Ultralytics project license) |
| Image embedding | `facebook/dinov3-vitl16-pretrain-lvd1689m` | `src/visual_memory/engine/embedding/embed_image.py` | https://github.com/facebookresearch/dinov3 | https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m | DINOv3 license (custom Meta license) |
| OCR text embedding | `openai/clip-vit-base-patch32` (text encoder only) | `src/visual_memory/engine/embedding/embed_text.py` | https://github.com/openai/CLIP | https://huggingface.co/openai/clip-vit-base-patch32 | MIT |
| Depth estimation | `apple/DepthPro` (`depth_pro.pt`) | `src/visual_memory/engine/depth/estimator.py` | https://github.com/apple/ml-depth-pro | https://huggingface.co/apple/DepthPro | Apple Machine Learning Research Model License |
| Speech recognition | `openai/whisper-large-v3-turbo` | `src/visual_memory/engine/speech_recognition/whisper_recognizer.py` | https://github.com/openai/whisper | https://huggingface.co/openai/whisper-large-v3-turbo | MIT |
| OCR service | PaddleOCR runtime models via `paddleocr` | `services/ocr/app.py` | https://github.com/PaddlePaddle/PaddleOCR | https://pypi.org/project/paddleocr/ | Apache-2.0 (framework) |
| Query parsing LLM | `llama3.2:1b` via Ollama | `src/visual_memory/utils/ollama_utils.py` | https://github.com/ollama/ollama | https://ollama.com/library/llama3.2 | Meta Llama 3.2 license and policy terms |

## Citations

The BibTeX entries used by this project are in:
- `CITATIONS.bib`

## License and usage notes

1) GroundingDINO
- License: Apache-2.0.
- Model card includes citation for arXiv:2303.05499.

2) YOLOE
- The project code license is AGPL-3.0.
- The weight file used by this project is downloaded from Ultralytics assets release URL in `setup_weights.py`.
- For downstream redistribution or deployment terms, review Ultralytics licensing pages and current release terms.

3) DINOv3
- Hugging Face card marks this model as `license: other` with `license_name: dinov3-license`.
- License link: https://ai.meta.com/resources/models-and-libraries/dinov3-license/
- This is a custom license, not Apache/MIT.
- The current DINOv3 license text includes attribution requirements such as displaying "Built with DINOv3" when redistributing products or derivatives.

4) CLIP
- OpenAI CLIP repo license is MIT.
- Model card references the original CLIP paper.

5) Depth Pro
- Hugging Face license id: `apple-amlr`.
- License text: https://huggingface.co/apple/DepthPro/raw/main/LICENSE
- License is research-focused and not equivalent to permissive OSS licenses.
- Redistribution requires preserving attribution text from the license.

6) Whisper
- OpenAI Whisper repo license is MIT.
- Model card references arXiv:2212.04356.

7) PaddleOCR
- PaddleOCR repository is Apache-2.0.
- OCR runtime may fetch model artifacts used by PaddleOCR; verify any model-specific notices when pinning or redistributing OCR weights.

8) Llama 3.2 via Ollama
- Ollama provides packaging and serving.
- Underlying model terms are governed by Meta Llama license and policy pages:
  - https://www.llama.com/llama3/license/
  - https://www.llama.com/llama3/use-policy/
- Ensure use stays within those terms.

## Canonical license URLs

- Apache-2.0: https://www.apache.org/licenses/LICENSE-2.0
- AGPL-3.0: https://www.gnu.org/licenses/agpl-3.0.html
- MIT: https://opensource.org/license/mit
- DINOv3 license: https://ai.meta.com/resources/models-and-libraries/dinov3-license/
- Apple AMLR license text: https://huggingface.co/apple/DepthPro/raw/main/LICENSE
- Meta Llama 3.2 license page: https://www.llama.com/llama3/license/

