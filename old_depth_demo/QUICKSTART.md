# Quick Start: Depth Estimation Testing

## Install Dependencies

```bash
# Make sure you have the required packages
pip install transformers torch pillow pillow-heif

# Or from project root
pip install -r requirements.txt
```

## Run the Tests

```bash
cd depth_demo
python test_depth_anything.py
```

## Understanding the Output

### Input (what you pass from Kotlin/Swift):
```python
{
    "image_path": "/path/to/image.heic",
    "focal_length_mm": 6.86,        # From Camera2 API
    "sensor_width_mm": 9.8,         # From Camera2 API  
    "image_width_px": 3024          # From bitmap.width
}
```

### Output (what you get back):
```python
{
    "depth_map": numpy_array,          # Full depth map
    "mean_depth_meters": 1.82,         # Average depth in image
    "mean_depth_feet": 5.97,           # Converted to feet
    "focal_length_px": 2115.8,         # Calculated focal length
    "scale_factor": 4.079,             # Calibration multiplier
    "calibrated": True                 # Whether calibration was applied
}
```

### For Object Detection:
```python
# After you get a bounding box from Grounding DINO
bbox = [x1, y1, x2, y2]  # From Grounding DINO

# Get depth at that object
object_depth = tester.get_depth_at_bbox(depth_map, bbox)

# Returns:
{
    "mean_depth_meters": 1.95,
    "mean_depth_feet": 6.4,
    "min_depth_meters": 1.88,
    "max_depth_meters": 2.01,
    "bbox": [1000, 1500, 2000, 2500]
}
```

## Troubleshooting

### Model Download
First run will download ~400MB model from HuggingFace. This is normal.

### GPU Not Found
Tests will run on CPU if no GPU. Expect ~2-3x slower but still works.

### Import Error
Make sure you're running from the depth_demo directory or have the parent in sys.path.

## Next: Integration

Once tests pass, the integration pattern is:

1. **Kotlin** → captures image, gets camera metadata
2. **Kotlin** → sends to Python server with focal length
3. **Python** → runs Grounding DINO (gets bbox)
4. **Python** → runs Depth Anything (gets depth map)
5. **Python** → extracts depth at bbox
6. **Python** → returns: "Your wallet is 6.4 feet away"
