# Depth Estimation Demo

Isolated testing environment for Depth Anything V2 before integration into the main pipeline.

## Purpose

Test and understand Depth Anything V2's:
- Input requirements (image, focal length, sensor dimensions)
- Output format (depth maps, metric depth values)
- Calibration accuracy (with vs without focal length)
- Integration with bounding boxes (from Grounding DINO)

## Running Tests

```bash
cd depth_demo
python test_depth_anything.py
```

## What the Tests Do

### Test 1: Basic Usage (Uncalibrated)
- Runs depth estimation WITHOUT focal length calibration
- Shows what you get "out of the box"
- May be inaccurate for cameras different from training data

### Test 2: With Calibration
- Uses simulated iPhone 15 Pro camera parameters
- Shows how to pass focal length from Kotlin/Swift
- Demonstrates calibration scaling

### Test 3: Bounding Box Depth
- Simulates getting depth at an object location
- This is what you'll integrate with Grounding DINO
- Returns average depth in the bounding box region

### Test 4: Comparison
- Compares calibrated vs uncalibrated at the same point
- Shows the difference calibration makes
- Helps validate if calibration is working correctly

## Expected Output

```
🧪 Depth Anything V2 Test Suite

Loading Depth Anything V2 Metric Indoor Base...
Using device: GPU
✓ Model loaded successfully!

============================================================
TEST 1: Basic Usage (No Calibration)
============================================================
[UNCALIBRATED] Processing: ../remember_mode/demo_images/Wallet.heic

Results:
  Image size: (3024, 4032)
  Depth range: 0.52m - 3.45m
  Average depth: 1.82m (5.97 feet)

... etc
```

## Key Learnings

After running these tests, you'll know:
1. ✅ How to pass focal length from Kotlin → Python
2. ✅ What format depth maps come in (numpy arrays)
3. ✅ How to extract depth at object locations (bbox)
4. ✅ Whether calibration is making a meaningful difference

## Next Steps

Once tests pass:
1. Integrate into `remember_mode/` as `depth_estimator.py`
2. Connect with Grounding DINO bounding boxes
3. Add to main pipeline API
4. Test with Kotlin client passing real camera metadata
