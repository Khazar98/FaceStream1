# Best Shot Selection Pipeline

This document describes how FaceStream selects and sends the highest-quality face crop ("best shot") for each detected person.

---

## Overview

For every tracked person, the system continuously evaluates incoming face crops and keeps the single best one. When the track ends (person leaves the scene), the stored best shot is sent to Kafka.

```
RTSP Frame
   │
   ▼
TensorRT Inference (YOLOv12n-face)
   │  conf_threshold = 0.72  ← only high-confidence detections enter tracker
   ▼
BoTSORT Tracker  (returns only current-frame-matched tracks)
   │
   ▼
Per-Detection Filters (see below)
   │
   ▼
Quality Score → keep best per track
   │
   ▼
Kafka (on track timeout)
```

---

## Filters (in order of execution)

| # | Filter | Threshold | Reason |
|---|--------|-----------|--------|
| 1 | **Tracker input confidence** | `≥ 0.72` | Only strong detections reach the tracker |
| 2 | **Minimum face size** | `50 × 50 px` | Rejects very distant/small faces |
| 3 | **Aspect ratio** | `1.0 – 1.8` (h/w) | Portrait orientation only |
| 4 | **Minimum track confirmations** | `3 frames` | Eliminates single-frame ghost detections |
| 5 | **Best-shot confidence** | `≥ 0.72` | Only confident detection stored as best shot |
| 6 | **Dark pixel ratio** | `≥ 2 %` | Face must have eyes/eyebrows (dark pixels); rejects uniform walls |
| 7 | **Dynamic range** | `≥ 40` (lum max − min) | Face must have internal contrast; rejects flat textures |

### Quality Score Formula

When a detection passes all filters, a quality score is computed:

```
quality_score = (confidence × 10) + √sharpness
```

- `confidence` — model output score (0–1)
- `sharpness` — Sobel gradient mean over the crop (higher = sharper)

The crop with the **highest quality score** across all frames of a track is stored and ultimately sent.

---

## Filters Removed / Relaxed (history)

| Filter | Old value | Current | Reason for change |
|--------|-----------|---------|-------------------|
| Min confidence | `0.80` | `0.72` | Captures more valid faces without increasing FP |
| Sharpness gate | `≥ 2.0` | *(removed)* | Redundant with quality score; was rejecting valid faces |
| Color stddev | `≥ 15.0` | *(removed)* | Dark pixel + dynamic range checks are more reliable |
| Min face size | `60 px` | `50 px` | Allows slightly smaller but still valid faces |
| Quality multiplier | `× 20` | `× 10` | Balances confidence vs sharpness contribution |

---

## False Positive Fixes

The following root causes of false positives were identified and fixed:

### 1. Stale tracker bbox (main bug)
**`tracker.cpp`** — `update()` was returning **all** tracked objects, including those not matched in the current frame. Their bounding boxes were from previous frames, so cropping the current frame with old coordinates produced wrong image regions (wall, door, floor).

**Fix:** Return only tracks with `time_since_update == 0`.

### 2. Low tracker input threshold
`conf_threshold = 0.45` was flooding the tracker with noise detections, creating many short-lived tracks from background textures.

**Fix:** Raised to `0.72`.

### 3. Texture uniform surfaces (wall/floor hallucination)
The model can occasionally predict wall or floor textures as faces at 0.80–0.87 confidence. Two pixel-level checks now catch these:
- **Dark pixel ratio < 2%** → no eyes/eyebrows present → rejected
- **Dynamic range < 40** → pixel values too uniform → rejected

---

## Kafka Payload

Best shots are sent to Kafka topic `detectiontopic` (and `detectionMega`) as a JSON message containing:

- `camera_id` — source camera UUID
- `track_id` — unique person track ID
- `confidence` — detection confidence score
- `sharpness` — √(Sobel mean) of the best crop
- `face_image` — base64-encoded JPEG (tight crop)
- `face_image_padded` — base64-encoded JPEG (50% padding around face)
- `timestamp` — detection time (ISO 8601)
