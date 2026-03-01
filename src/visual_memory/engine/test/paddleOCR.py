"""
The following command installs the PaddlePaddle version for CUDA 12.6. 
For other CUDA versions and the CPU version, refer to:
https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/linux-pip_en.html

python -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
python -m pip install -U "paddleocr[doc-parser]"
"""

from paddleocr import PaddleOCRVL
import os

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

# Initialize pipeline
pipeline = PaddleOCRVL()

# Images to test
image_files = [
    "malarkey.jpeg",
    "marker.jpeg",
    "pen.jpeg",
    "pencil.jpeg",
    "typed.jpeg"
]

outputs = []

# Run inference on each image
for image_path in image_files:
    print(f"\nProcessing {image_path} ...")

    results = pipeline.predict(image_path)  # returns a LIST
    outputs.append(results)

    for i, result in enumerate(results):
        result.print()
        result.save_to_json(save_path=f"output/{image_path}_{i}.json")
        result.save_to_markdown(save_path=f"output/{image_path}_{i}.md")

print("\nDone.")