# Bug Fix: Missing Normalization in Preprocessing

## Problem Description

When running YOLO predict directly with the `.engine` file, face detections were correct (all detections above 0.7 confidence were actual faces). However, when using the FaceStream C++ code, many false positives appeared - detections with high confidence (>0.7) that were NOT faces.

## Root Cause

**Missing pixel normalization by 255.0 in C++ preprocessing!**

### Python (Ultralytics) - CORRECT
```python
input_data = np.ascontiguousarray(input_data).astype(np.float32) / 255.0
```

### C++ (Bug) - INCORRECT
```cpp
inputBlob[...] = (float)pixel[...];  // Values: 0-255
```

### C++ (Fixed) - CORRECT
```cpp
inputBlob[...] = (float)pixel[...] / 255.0f;  // Values: 0.0-1.0
```

## Why This Caused False Positives

1. **Neural networks expect normalized inputs** in range [0.0, 1.0] or [-1.0, 1.0]
2. When feeding [0-255] values instead of [0.0-1.0]:
   - Pixel values are **255x larger** than expected
   - Feature activations become abnormally high
   - Model produces incorrect confidence scores
   - Background/non-face regions get high confidence (>0.7)

## The Fix

**File:** `pkg/cpp_processor/processor.cpp`

**Location:** Lines 492-494 (inside the preprocessing loop)

**Change:**
```cpp
// BEFORE (Bug):
inputBlob[offset + r*640 + c] = (float)pixel[2];           // R
inputBlob[offset + 640*640 + r*640 + c] = (float)pixel[1]; // G
inputBlob[offset + 2*640*640 + r*640 + c] = (float)pixel[0]; // B

// AFTER (Fix):
inputBlob[offset + r*640 + c] = (float)pixel[2] / 255.0f;           // R
inputBlob[offset + 640*640 + r*640 + c] = (float)pixel[1] / 255.0f; // G
inputBlob[offset + 2*640*640 + r*640 + c] = (float)pixel[0] / 255.0f; // B
```

## Verification

To verify the fix is applied:
```bash
grep -A5 "Normalize by 255" pkg/cpp_processor/processor.cpp
```

Should show:
```
// [FIX] Normalize by 255.0 to match Python/Ultralytics preprocessing
inputBlob[offset + r*640 + c] = (float)pixel[2] / 255.0f;           // R
inputBlob[offset + 640*640 + r*640 + c] = (float)pixel[1] / 255.0f; // G
inputBlob[offset + 2*640*640 + r*640 + c] = (float)pixel[0] / 255.0f; // B
```

## Rebuild Instructions

```bash
# Rebuild the C++ library
cd pkg/cpp_processor
make clean && make

# Or from project root
make cpp_lib

# The library is now fixed and ready to use
```

## Expected Results After Fix

- Face detections with confidence >0.7 should be actual faces
- False positives (non-face regions with high confidence) should be eliminated
- Detection quality should match the Python YOLO predict results

## Technical Details

### Input Value Ranges
- **Without normalization:** 0-255 (uint8 range)
- **With normalization:** 0.0-1.0 (normalized range)

### Impact on Model
- YOLO models are trained with normalized inputs
- Feeding unnormalized inputs is like multiplying brightness by 255
- Causes activations to saturate and produces spurious detections

### Why It Wasn't Obvious
- The code ran without errors (no crash)
- Detections were still produced (just incorrect ones)
- Confidence scores looked reasonable (0.7-0.9)
- Only comparison with Python revealed the discrepancy
