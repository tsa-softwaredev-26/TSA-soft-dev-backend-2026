# Spaitra Frontend Integration Guide

> For the mobile team. Everything needed to integrate with the backend API.
> Backend details: see ARCHITECTURE.md.

---

## Base URL and Auth

```
Base URL: http://<host>:5000
Default local: http://127.0.0.1:5000
```

Auth is opt-in. If `API_KEY` is set on the server, all routes except `/health` require:
```
X-API-Key: <secret>
```
If the key is missing or wrong, the server returns `401`. `/health` is always open.

Upload cap: 50MB per request.

---

## Endpoints

### GET /health

Health check. No auth required.

```json
{ "status": "ok" }
```

---

### POST /remember

Teach the app a new object. Detects the object in the image, embeds it, and saves it to the database. The new item is immediately visible to `/scan` after this returns.

**Request:** `multipart/form-data`

| Field | Type | Required |
|-------|------|----------|
| `image` | file | yes |
| `prompt` | string | yes - e.g. `"wallet"`, `"house keys"` |

**Response (success):**
```json
{
  "success": true,
  "message": "Object detected and embedded successfully.",
  "result": {
    "label": "wallet",
    "confidence": 0.87,
    "box": [120, 200, 340, 410],
    "ocr_text": "RFID Blocking",
    "ocr_confidence": 0.91
  }
}
```

**Response (no detection):**
```json
{ "success": false, "message": "No object detected.", "result": null }
```

Notes:
- `ocr_text` and `ocr_confidence` are only present when OCR found text on the object.
- `box` is `[x1, y1, x2, y2]` in pixels.
- If detection fails, prompt the user to move closer or try again.

---

### POST /scan

Scan the current scene for previously taught objects. Returns matches sorted left-to-right by spatial position.

**Request:** `multipart/form-data`

| Field | Type | Required |
|-------|------|----------|
| `image` | file | yes |
| `focal_length_px` | float | no - pass for calibrated depth; omit or `0` to disable depth |

**Focal length formula (for calibrated depth):**
```
focal_length_px = (focalLengthMm / sensorWidthMm) * imageWidthPx
```
iPhone 15 Plus reference: `f_mm=6.24`, `sensor_width=8.64mm` -> `focal_length_px = 3094.0`

**Response:**
```json
{
  "scan_id": "a3f9c2b1...",
  "count": 2,
  "matches": [
    {
      "label": "wallet",
      "similarity": 0.72,
      "confidence": "high",
      "direction": "to your left",
      "narration": "Wallet to your left, 2.3 feet away.",
      "distance_ft": 2.3,
      "box": [12, 44, 210, 380],
      "crop_b64": "<base64-encoded JPEG>",
      "ocr_text": "RFID Blocking"
    },
    {
      "label": "house keys",
      "similarity": 0.38,
      "confidence": "medium",
      "direction": "to your right",
      "narration": "May be house keys to your right, focus to verify.",
      "distance_ft": 4.1,
      "box": [640, 100, 820, 300],
      "crop_b64": "<base64-encoded JPEG>",
      "ocr_text": null
    }
  ]
}
```

**Field reference:**

| Field | Always present | Notes |
|-------|---------------|-------|
| `scan_id` | yes | UUID hex - pass to `/feedback` |
| `label` | yes | Object name as taught |
| `similarity` | yes | Cosine similarity score, 0-1 |
| `confidence` | yes | `"high"` / `"medium"` / `"low"` - see thresholds below |
| `direction` | yes | One of 5 zones - see below |
| `narration` | yes | Ready-to-speak string |
| `box` | yes | `[x1, y1, x2, y2]` in pixels |
| `crop_b64` | yes | Base64 JPEG of the detected object - decode and display/share directly |
| `distance_ft` | only with depth | Metric depth in feet |
| `ocr_text` | only when found | Text extracted from the object |

**Confidence thresholds:**

| Label | Similarity range | Narration style |
|-------|-----------------|-----------------|
| `"high"` | >= 0.5 | `"Wallet to your left, 2.3 feet away."` |
| `"medium"` | 0.35 - 0.5 | `"May be a wallet to your left, focus to verify."` |
| `"low"` | 0.2 - 0.35 | `"May be a wallet to your left, focus to verify."` |

**Direction zones:**

| Value | Meaning |
|-------|---------|
| `"to your left"` | Far left (< 25% of frame) |
| `"slightly left"` | Left of center |
| `"ahead"` | Center |
| `"slightly right"` | Right of center |
| `"to your right"` | Far right (> 75% of frame) |

**Matches are always sorted left-to-right.** Read them out in array order for spatial narration.

**When depth is disabled** (`ENABLE_DEPTH=0` server-side or `focal_length_px=0`): `distance_ft` is absent, narration omits distance but keeps direction. All other fields are still present.

---

### POST /feedback

Record whether a scan result was correct or wrong. This powers the personalization system - the projection head adapts over time to be more accurate for this specific user.

**Request:** `application/json`

```json
{
  "scan_id": "a3f9c2b1...",
  "label": "wallet",
  "feedback": "correct"
}
```

`feedback` must be `"correct"` or `"wrong"`.

**Response:**
```json
{
  "recorded": true,
  "label": "wallet",
  "feedback": "correct",
  "triplets": 12,
  "min_for_training": 10
}
```

**404** if `scan_id` is not found. The backend caches the last 50 scan results. Don't delay feedback more than 50 scans.

When `triplets >= min_for_training`, call `/retrain` to apply the accumulated feedback.

---

### POST /retrain

Retrain the personalization model on accumulated feedback. Weights are hot-loaded into the scan pipeline immediately - no restart needed.

**Request:** no body

**Response (not enough data):**
```json
{
  "trained": false,
  "reason": "insufficient_data",
  "triplets": 3,
  "min_required": 10
}
```

**Response (success):**
```json
{
  "trained": true,
  "triplets": 15,
  "final_loss": 0.0421,
  "head_weight": 0.30
}
```

Note: training is synchronous and may take several seconds. Plan for a loading state. (Async retrain is a planned backend improvement.)

---

### GET /settings

Read ML runtime state. Useful for showing personalization progress in settings UI.

**Response:**
```json
{
  "enable_learning": true,
  "min_feedback_for_training": 10,
  "projection_head_weight": 1.0,
  "projection_head_ramp_at": 50,
  "projection_head_epochs": 20,
  "head_trained": false,
  "triplet_count": 0,
  "feedback_counts": {
    "positives": 5,
    "negatives": 3,
    "triplets": 3
  }
}
```

`head_trained` = whether the personalization model has weights loaded (false = no feedback yet).
`triplet_count` = total triplets used in last training run.

---

### PATCH /settings

Update ML runtime parameters. All fields optional. Returns the full settings state.

**Request:** `application/json` (any subset)

```json
{
  "enable_learning": true,
  "min_feedback_for_training": 10,
  "projection_head_weight": 1.0,
  "projection_head_ramp_at": 50,
  "projection_head_epochs": 20
}
```

Note: these are not persisted across restarts (planned fix).

---

### GET /user-settings

User preferences. These are persisted in the database.

**Response:**
```json
{
  "performance_mode": "balanced",
  "voice_speed": 1.0,
  "auto_update_location": false,
  "learning_enabled": true,
  "button_layout": "default",
  "performance_config": {
    "depth_enabled": true,
    "target_latency": 3.0
  }
}
```

**Performance modes:**

| Mode | Depth enabled | Target latency |
|------|--------------|----------------|
| `"fast"` | no | ~1s |
| `"balanced"` | yes | ~3s |
| `"accurate"` | yes | ~5s |

`voice_speed` range: 0.5 - 2.0. Apply this to your TTS engine when reading narrations.

---

### PATCH /user-settings

Update user preferences. Persisted to the database. All fields optional.

**Request:** `application/json` (any subset)

```json
{
  "performance_mode": "fast",
  "voice_speed": 1.2,
  "auto_update_location": false,
  "learning_enabled": true,
  "button_layout": "default"
}
```

`learning_enabled` is propagated to the scan pipeline immediately (no restart needed).

**400** on type or range errors. Response includes `{"errors": {...}}` with per-field messages.

---

## Scan -> Feedback -> Retrain Flow

The full personalization loop:

```
1. POST /scan
   - Store scan_id and the matches array

2. User swipes through matches
   - For each match, surface confirm/deny UI
   - POST /feedback with scan_id + label + "correct"/"wrong"
   - Response tells you triplets and min_for_training

3. When triplets >= min_for_training
   - Show a "Ready to improve" prompt
   - POST /retrain on user confirmation (or automatically)
   - On success, next /scan uses updated model weights
```

The model starts as identity (no effect) and gradually adapts. `head_weight` in the retrain response shows how much influence the personalized model currently has (ramps up as more feedback accumulates, capped at 1.0).

---

## Image Tips

- Send JPEG or HEIC. Both are supported.
- For `/remember`: get close (1-3ft). 6ft is unreliable for teach mode.
- For `/scan`: wider shots work; the model crops detected regions internally.
- EXIF orientation is automatically corrected server-side.
- If `/remember` returns `"No object detected."`, the object is probably too small or occluded.

---

## Error Handling

| Status | Meaning |
|--------|---------|
| 400 | Bad request - missing field, wrong type, or invalid value |
| 401 | Missing or wrong `X-API-Key` |
| 404 | `scan_id` expired or not found (feedback only) |
| 500 | Server error - log and surface a generic retry message |

All error responses include `{"error": "<message>"}` or `{"errors": {...}}` (PATCH routes).

---

## Notes for TTS

- Read `narration` directly. It is phrased for a blind user.
- Apply `voice_speed` from `/user-settings` to your TTS rate.
- Read matches in array order - they are already left-to-right spatially.
- For low/medium confidence matches, the narration already includes `"May be a..."` - do not add additional hedging in the UI.
- `ocr_text` is raw extracted text. Read it verbatim if the user requests it.
