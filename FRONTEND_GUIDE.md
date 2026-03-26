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

**Single image - request:** `multipart/form-data`

| Field | Type | Required |
|-------|------|----------|
| `image` | file | yes (single frame) |
| `prompt` | string | yes - e.g. `"wallet"`, `"house keys"` |

**Multi-image (recommended) - request:** `multipart/form-data`

Send 2-5 frames captured in quick succession (~500ms apart). Backend picks the frame with the highest-confidence detection and returns a single result.

| Field | Type | Required |
|-------|------|----------|
| `images[]` | file[] | yes (2-5 frames) |
| `prompt` | string | yes |

**Response (success):**
```json
{
  "success": true,
  "message": "Object detected and embedded successfully.",
  "images_tried": 4,
  "images_with_detection": 3,
  "result": {
    "label": "wallet",
    "confidence": 0.87,
    "detection_quality": "high",
    "detection_hint": "Detection confidence is high.",
    "blur_score": 143.7,
    "is_blurry": false,
    "is_dark": false,
    "darkness_level": 112.4,
    "replaced_previous": false,
    "second_pass": false,
    "second_pass_prompt": null,
    "text_likelihood": 0.62,
    "box": [120, 200, 340, 410],
    "ocr_text": "RFID Blocking",
    "ocr_confidence": 1.0
  }
}
```

**Response (failure - too dark):**
```json
{
  "success": false,
  "message": "Image is too dark for detection. Enable the flashlight and retry.",
  "is_dark": true,
  "darkness_level": 12.4,
  "blur_score": 5.1,
  "is_blurry": false,
  "result": null
}
```

**Response (failure - no detection):**
```json
{
  "success": false,
  "message": "No object detected.",
  "is_dark": false,
  "darkness_level": 87.3,
  "blur_score": 48.2,
  "is_blurry": true,
  "result": null
}
```

**Result field reference:**

| Field | Notes |
|-------|-------|
| `detection_quality` | `"low"` / `"medium"` / `"high"` based on GroundingDINO confidence |
| `detection_hint` | Human-readable advice string - surface verbatim when quality is low or medium |
| `blur_score` | Laplacian variance; > 100 is sharp |
| `is_blurry` | true when `blur_score < 100` |
| `is_dark` | true when mean luminance < 30 - check this FIRST before other signals |
| `darkness_level` | Mean grayscale 0-255 |
| `replaced_previous` | true if a prior teach for this label was overwritten - say "Updated [label]" |
| `second_pass` | true if detection only succeeded with a reformulated prompt |
| `text_likelihood` | 0.0-1.0 heuristic estimate of text on the object; > 0.3 = OCR was likely run |
| `ocr_text` | Raw extracted text (empty string when none found or OCR skipped) |
| `images_tried` | Only present in multi-image mode |
| `images_with_detection` | Only present in multi-image mode |

**Frontend signals:**

| Signal | Recommended UX |
|--------|---------------|
| `is_dark: true` | PRIORITY: enable flashlight; speak "It's too dark - turning on flashlight" |
| `detection_quality: "low"` | Show warning; speak `detection_hint` verbatim |
| `detection_quality: "medium"` | Soft prompt; surface `detection_hint` |
| `is_blurry: true` + `is_dark: false` | "Hold steady and try again" |
| `replaced_previous: true` | Say "Updated [label]" not "[label] saved" |
| `text_likelihood > 0.3` | Show "Text label detected on this item" |
| `success: false` + `is_dark: false` + `is_blurry: false` | "Try a different angle, better lighting, or a more specific label" |

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
      "ocr_text": "RFID Blocking"
    },
    {
      "label": "house keys",
      "similarity": 0.38,
      "confidence": "medium",
      "direction": "to your right",
      "narration": "May be house keys to your right, focus to verify.",
      "distance_ft": 4.1,
      "text_likelihood": 0.05,
      "box": [640, 100, 820, 300]
    }
  ]
}
```

**Dark scene response** (check `is_dark` at top level, not inside matches):
```json
{
  "matches": [],
  "count": 0,
  "is_dark": true,
  "darkness_level": 8.2,
  "message": "Image is too dark for detection. Enable the flashlight and retry."
}
```

**Field reference:**

| Field | Always present | Notes |
|-------|---------------|-------|
| `scan_id` | yes | UUID hex - pass to `/feedback` and `/crop` |
| `is_dark` | top level only | true when scene is too dark; check before processing matches |
| `label` | yes | Object name as taught - pass to `DELETE /items/<label>` to forget it |
| `similarity` | yes | Cosine similarity score, 0-1 |
| `confidence` | yes | `"high"` / `"medium"` / `"low"` - see thresholds below |
| `direction` | yes | One of 5 zones - see below |
| `narration` | yes | Ready-to-speak string |
| `text_likelihood` | yes | 0.0-1.0 heuristic; > 0.3 means the item has a text label and OCR ran |
| `box` | yes | `[x1, y1, x2, y2]` in pixels |
| `distance_ft` | only with depth | Metric depth in feet; absent when `ENABLE_DEPTH=0` or `focal_length_px=0` |
| `ocr_text` | only when found | Text extracted from the object |

**Confidence thresholds:**

| Label | Similarity range | Narration style |
|-------|-----------------|-----------------|
| `"high"` | >= 0.35 | `"Wallet to your left, 2.3 feet away."` |
| `"medium"` | 0.25 - 0.35 | `"May be a wallet to your left, focus to verify."` |
| `"low"` | 0.2 - 0.25 | `"May be a wallet to your left, focus to verify."` |

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

### GET /crop

Fetch the cropped image for a specific match by its index in the scan result array. Returns a raw JPEG. Call this only when the user navigates to that match - do not prefetch all crops.

**Request:** query params

| Param | Type | Required |
|-------|------|----------|
| `scan_id` | string | yes |
| `index` | integer | yes - 0-based position in the `matches` array |

**Response:** `image/jpeg` - the cropped region of the detected object.

**404** if `scan_id` has expired (last 50 scans cached) or `index` is out of range.

Example: `GET /crop?scan_id=a3f9c2b1...&index=1`

---

### GET /items

List all taught objects. Use this to fetch item details before confirming a deletion - see voice delete flow below.

**Request:** query params

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `label` | string | no | Filter to a single label. Omit to return all items. |

**Response:**
```json
{
  "items": [
    { "id": 3, "label": "wallet", "confidence": 0.87, "ocr_text": "RFID Blocking", "timestamp": 1742700000.0 },
    { "id": 1, "label": "house keys", "confidence": 0.91, "ocr_text": "", "timestamp": 1742600000.0 }
  ],
  "count": 2
}
```

Results are ordered most-recently-taught first. Multiple rows with the same label means the user taught the object more than once - all rows are removed by `DELETE /items/<label>`.

---

### DELETE /items/<label>

Permanently remove a taught object from memory. After deletion the object will no longer be detected in future scans.

**Request:** no body

**Response:**
```json
{ "deleted": true, "label": "wallet", "count": 1 }
```

`count` is the number of database rows removed. **404** if no item with that label exists.

The scan pipeline reloads immediately - no restart needed.

---

### GET /find

Ask Mode - retrieve the last known location of a taught object. Searches by label with fuzzy semantic matching (so `"my wallet"` will match a label stored as `"wallet"`).

**Request:** query params

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `label` | string | no | Object to search for. Omit to get last sighting of every known object. |
| `limit` | integer | no | Max sightings to return (1-100, default 1) |
| `since` | float | no | Unix timestamp - only return sightings after this time |
| `before` | float | no | Unix timestamp - only return sightings before this time |

**Response (label found - exact or fuzzy match):**
```json
{
  "label": "my wallet",
  "found": true,
  "matched_label": "wallet",
  "last_sighting": {
    "id": 41,
    "label": "wallet",
    "timestamp": 1742860412.3,
    "last_seen": "3 minutes ago",
    "direction": "to your left",
    "distance_ft": 2.3,
    "similarity": 0.72,
    "crop_path": "/path/to/scans/abc123_0.jpg"
  },
  "sightings": [...]
}
```

`matched_label` is only present when a fuzzy match was used (query did not exactly match any stored label).

**Response (no label param - overview of all known objects):**
```json
{
  "results": [
    { "id": 41, "label": "wallet", "last_seen": "3 minutes ago", ... },
    { "id": 38, "label": "house keys", "last_seen": "2 hours ago", ... }
  ],
  "count": 2
}
```

Results are ordered most-recently-seen first.

**Response (not found):**
```json
{ "label": "wallet", "found": false, "sightings": [] }
```

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
  "performance_mode": "balanced", # fast = no depth, no ocr, balanced = depth, ocr included w/ base models, accurate = base + better models
  "voice_speed": 1.0,
  "scan_update_location": true, #asks user where they are after a scan mode detection, updates all
  "learning_enabled": true, # train personalized model from user feedback
  "button_layout": "default", # default or swapped
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
  "scan_update_location": false,
  "learning_enabled": true,
  "button_layout": "default"
}
```

`learning_enabled` is propagated to the scan pipeline immediately (no restart needed).

**400** on type or range errors. Response includes `{"errors": {...}}` with per-field messages.

---

### POST /sightings

Record the location of detected objects. Dual-use: called from remember mode (just taught - where is it?) and scan mode (objects detected - what room?). `room_name` is normalized before storage ("In The Kitchen" -> "kitchen").

**Request:** `application/json`

```json
{
  "room_name": "kitchen",
  "sightings": [
    { "label": "wallet", "direction": "to your left", "distance_ft": 3.2, "similarity": 0.61 }
  ]
}
```

`direction`, `distance_ft`, and `similarity` are optional - omit them in remember mode.
`room_name` is optional - omit when room is unknown.

**Response:**
```json
{ "saved": 1, "labels": ["wallet"], "room_name": "kitchen" }
```

`room_name` in the response is the normalized form stored in the database.

**Remember mode flow:**
```
POST /remember  ->  success  ->  ask "Where is this?"
User says "in the kitchen"
POST /sightings  {"room_name": "kitchen", "sightings": [{"label": "wallet"}]}
```

**Scan mode flow:**
```
POST /scan  ->  matches  ->  ask "Update location? What room?"
User says "bedroom"
POST /sightings  {"room_name": "bedroom", "sightings": [all matches + spatial data]}
```

Auto-sightings: /scan automatically records a sighting (direction only, no room_name) for every match. GET /find works without a POST /sightings call; POST /sightings enriches it with the room name.

---

### POST /items/<label>/rename

Rename a stored memory. Two-step flow to handle conflicts.

**Step 1 - without force:**
```
POST /items/wallet/rename
Content-Type: application/json
{"new_label": "My Wallet"}
```

**Response (200 - success):**
```json
{"renamed": true, "old_label": "wallet", "new_label": "My Wallet", "count": 1, "replaced": 0}
```

**Response (409 - conflict):**
```json
{"conflict": true, "message": "A memory named \"My Wallet\" already exists (1 item(s)). Send with force=true to overwrite it.", "existing_count": 1}
```

**Step 2 - user confirms overwrite, send with force:**
```json
{"new_label": "My Wallet", "force": true}
```

**Other error responses:**
- `404` - old label not found
- `400` - `new_label` missing or same as current label

**UX flow:**
1. User swipes left on a memory - reveal Rename / Delete
2. Tap Rename -> show text input pre-filled with current label
3. User edits -> POST without force
4. If 409: alert "Replace existing '[new_label]'?" -> resend with force=true
5. On success: update the list in-place

---

### POST /retrain

Start personalization model training in the background. Returns immediately; poll `/retrain/status` for results.

**Response (training started):**
```json
{"started": true, "triplets": 15}
```

**Response (already running):**
```json
{"started": false, "reason": "already_running"}
```

**Response (not enough feedback):**
```json
{"started": false, "reason": "insufficient_data", "triplets": 3, "min_required": 10}
```

---

### GET /retrain/status

Poll training progress.

**Response:**
```json
{
  "running": false,
  "last_result": {
    "trained": true,
    "triplets": 15,
    "final_loss": 0.012345,
    "head_weight": 0.3
  },
  "error": null
}
```

`last_result` is null until the first successful training run. `error` is null unless training failed with an exception. `running: true` means training is currently in progress.

**Typical poll flow:**
```
POST /retrain  ->  {"started": true}
poll GET /retrain/status every 5s until running: false
show result or error
```

---

## Scan -> Feedback Flow

After each scan the user can optionally confirm or deny matches. This powers the personalization system - the backend retrains automatically overnight once enough feedback accumulates.

```
1. POST /scan
   - Store scan_id and the matches array

2. User swipes through matches
   - For each match, allow confirm/deny (voice or gesture)
   - POST /feedback with scan_id + label + "correct"/"wrong"

3. Retraining - POST /retrain to start; GET /retrain/status to poll
   - Next /scan uses updated model weights transparently
```

The model starts as identity (no effect on scan accuracy) and gradually adapts as feedback accumulates. Training is async - POST /retrain starts it in the background, poll GET /retrain/status for results. The frontend does not need to trigger retraining automatically; it can be surfaced as a manual "Improve accuracy" action once enough feedback exists.

---

## Scan Mode UX

The `matches` array is ordered left-to-right spatially. Navigate it like a paged list:

```
POST /scan -> matches: [wallet, keys, charger]
                         ^
                    user swipes left/right through indexes

Read narration[i] aloud as user lands on each item.
```

**Voice: "delete this"**
```
1. User is on match[i] (label = "wallet")
2. GET /items?label=wallet -> returns item details (taught date, confidence)
3. Announce: "Are you sure you want to forget wallet? Taught 3 days ago."
4. User confirms -> DELETE /items/wallet
5. Item permanently removed; will not appear in future scans
```

---

## Scan -> Find (Ask Mode)

Every successful scan match is automatically recorded. Ask Mode queries that history:

```
1. POST /scan
   - Matches are saved to sighting history automatically

2. Later: GET /find?label=wallet
   - Returns last known location with direction, distance, and human-readable time
   - Fuzzy match: "my wallet", "the wallet", "brown wallet" all resolve to "wallet"
   - Build narration from: last_sighting.direction + last_sighting.distance_ft + last_sighting.last_seen

3. GET /find (no label)
   - Returns the most recent sighting per known object - good for "where is everything"
```

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
