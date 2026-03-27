# Spaitra Frontend Integration Guide

> Flow-based integration guide for the mobile team.
> Organized by user action, not by endpoint. Quick reference table at the bottom.
> Backend internals: ARCHITECTURE.md.

---

## Auth and Base URL

```
Base URL: http://<host>:5000
Default local: http://127.0.0.1:5000
```

If `API_KEY` is set on the server, every route except `GET /health` requires:
```
X-API-Key: <secret>
```
Missing or wrong key -> `401`. No key configured on the server -> no header needed.

Upload cap: 50MB per request.

---

## What to Store Client-Side

Here is what state the client needs to track across calls.

**`scan_id`** - returned by every `POST /scan` response. Store it immediately. It is needed for:
- `GET /crop` - fetch the cropped image of a match
- `POST /feedback` - tell the backend whether a match was correct

The server caches the last 50 scans. If the user does 50 more scans before you send feedback, the `scan_id` expires and `/feedback` returns 404. In practice: send feedback during the same scan session, before the user navigates away.

**`voice_speed`** - read once from `GET /user-settings` at app launch. Apply to your TTS engine for every narration.

**Current match index** - 0-based position in the `matches` array from the last scan. The client owns this; the server does not track it.

---

## Teach Mode

The user wants the app to remember a new object. The flow is: capture -> POST /remember -> handle signals -> optionally record location.

### Step 1: Capture

Capture a burst of 3-5 frames in quick succession (~500ms apart) when the user initiates a teach. The backend picks the best frame automatically.

### Step 2: POST /remember

```
POST /remember
Content-Type: multipart/form-data
```

| Field | Type | Notes |
|-------|------|-------|
| `images[]` | file[] | 2-5 frames from the burst (recommended) |
| `image` | file | Single frame (fallback - send `images[]` when possible) |
| `prompt` | string | The user's label, e.g. `"wallet"`, `"house keys"` |

**Success response:**
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
    "blur_score": 312.7,
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

`images_tried` and `images_with_detection` only appear when using `images[]`.

**Failure - too dark:**
```json
{
  "success": false,
  "message": "Image is too dark for detection. Increase lighting or ask me to turn on the flashlight.",
  "is_dark": true,
  "darkness_level": 12.4,
  "blur_score": 5.1,
  "is_blurry": false,
  "result": null
}
```

**Failure - no detection:**
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

### Step 3: Handle the response signals

Check signals in this order:

**1. `is_dark: true`** - highest priority, detection quality drastically worsens in bad lighting. Allow the user to enable the flashlight with a voice command on the front end, or simply turn on the lights
```
speak: "Image is too dark for detection. Increase lighting or ask me to turn on the flashlight.""
```

**2. `success: false` + `is_dark: false`**
```
is_blurry: true  -> "Hold steady and try again"
is_blurry: false -> "Could not detect - try a different angle, better lighting, or a more specific label"
```

**3. `success: true`** - object saved.
```
replaced_previous: true  -> say "Updated [label]"
replaced_previous: false -> say "[label] saved"
```
If `detection_quality` is `"low"` or `"medium"`, announce `detection_hint` as a soft warning. "Detected wallet with low confidence." The object is saved regardless; the hint just tells the user a better photo would improve future scans. They can redo if they want or move on.

### Step 4: Record location (optional)

After a successful teach, ask the user where the object is. If they answer:

```
POST /sightings
Content-Type: application/json
```
```json
{
  "room_name": "kitchen",
  "sightings": [{ "label": "wallet" }]
}
```

`room_name` is normalized server-side ("In The Kitchen" -> "kitchen"). The normalized form is echoed in the response. Omit `room_name` if the user does not name the room.

Response:
```json
{ "saved": 1, "labels": ["wallet"], "room_name": "kitchen" }
```

---

## Scan Mode

The user scans the room to find previously taught objects. The flow is: capture -> POST /scan -> store `scan_id` -> navigate matches -> (optional) crop, feedback, location update, retrain.

### Step 1: POST /scan

```
POST /scan
Content-Type: multipart/form-data
```

| Field | Type | Notes |
|-------|------|-------|
| `image` | file | The scene image |
| `focal_length_px` | float | Enables calibrated depth, drastically improves accuracy. Omit or send `0` to disable. |

**Focal length (for depth):**
```
focal_length_px = (focalLengthMm / sensorWidthMm) * imageWidthPx
```
Example used in docs & benchmarks: iPhone 15 Plus -`f_mm=6.24, sensor_width=8.64mm` -> `focal_length_px = 3094.0`

### Step 2: Check for darkness

Before touching `matches`, check the top-level `is_dark` field:

```json
{
  "matches": [],
  "count": 0,
  "is_dark": true,
  "darkness_level": 8.2,
  "message": "Image is too dark for detection. Increase lighting or ask me to turn on the flashlight."
}
```

`is_dark` is at the top level of the scan response, not inside a match. If true, speak the message and stop.

### Step 3: Store scan_id and navigate matches

A normal scan response looks like:

```json
{
  "scan_id": "a3f9c2b1d4e5f6a7",
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
      "box": [640, 100, 820, 300]
    }
  ]
}
```

**Store `scan_id` immediately** - you need it for `/crop` and `/feedback`.

`matches` is ordered left-to-right spatially. Read `narration` aloud for each match in array order. 

`distance_ft` is only present when depth is enabled. When absent, narration still includes direction.

**Direction zones:** (for debug reference, narration already reflects direction)

| Value | Where in the frame |
|-------|-------------------|
| `"to your left"` | Far left (< 25%) |
| `"slightly left"` | Left of center |
| `"ahead"` | Center |
| `"slightly right"` | Right of center |
| `"to your right"` | Far right (> 75%) |

**Confidence tiers** (for debug reference, narration already reflects confidence):

| Value | Similarity range |
|-------|-----------------|
| `"high"` | >= 0.35 |
| `"medium"` | 0.25 - 0.35 |
| `"low"` | 0.2 - 0.25 |

### Step 4: Fetch a crop (on demand)

The user can swipe left in right to navigate through matches in spatial order. When the user focuses on a specific match, fetch its cropped image:

```
GET /crop?scan_id=a3f9c2b1d4e5f6a7&index=0
```

`index` is the 0-based position in the `matches` array. Returns a raw JPEG. Do not prefetch; call this only when the user navigates to that match.

The user can ask questions about the selected item or give feedback, use the scan_id to create the POST.

Returns `404` if the `scan_id` has expired (older than the last 50 scans) or `index` is out of range.

### Step 5: Collect feedback (optional but important)

After the user reviews each match, record whether it was right or wrong. This is what trains the personalization model over time.

```
POST /feedback
Content-Type: application/json
```
```json
{
  "scan_id": "a3f9c2b1d4e5f6a7",
  "label": "wallet",
  "feedback": "correct"
}
```

`feedback` must be `"correct"` or `"wrong"`. Use the `scan_id` from the scan that produced this match.

Response:
```json
{
  "recorded": true,
  "label": "wallet",
  "feedback": "correct",
  "triplets": 12,
  "min_for_training": 10
}
```

`triplets` is the total training pairs collected so far. 

`404` means the `scan_id` is no longer cached. Send feedback before the user does 50 more scans.

### Step 6: Update location (optional)

After a scan, if the user names the room, record it for all matches at once (except the ones that they gave negative feedback for):

```
POST /sightings
Content-Type: application/json
```
```json
{
  "room_name": "bedroom",
  "sightings": [
    { "label": "wallet",     "direction": "to your left", "distance_ft": 2.3, "similarity": 0.72 },
    { "label": "house keys", "direction": "to your right", "distance_ft": 4.1, "similarity": 0.38 }
  ]
}
```

`direction`, `distance_ft`, and `similarity` come directly from the match objects. Omit `room_name` if the user does not name the room. The scan pipeline already records a basic sighting for every match automatically - this call enriches it with the room name.

**Do NOT update room name for matches the user declined**

---

## Ask Mode

The user asks "where did I leave my wallet?" Query the last known location by label. Fuzzy matching is built in as a fallback - "my wallet", "the wallet", and "brown wallet" all resolve to "wallet".

```
GET /find?label=wallet
```

**Found:**
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
    "room_name": "kitchen",
    "crop_path": null
  },
  "sightings": [...]
}
```

`matched_label` only appears when the query did not exactly match any stored label (fuzzy match was used). Build the narration from `last_sighting.direction`, `distance_ft`, `last_seen`, and `room_name`.

`room_name` is null until the user has confirmed a location via `POST /sightings`. `last_seen` is always a human-readable string ("3 minutes ago", "2 hours ago").

**Not found:**
```json
{ "label": "wallet", "found": false, "sightings": [] }
```

**All objects at once** (omit `label`):
```
GET /find
```
```json
{
  "results": [
    { "label": "wallet", "last_seen": "3 minutes ago", "direction": "to your left", ... },
    { "label": "house keys", "last_seen": "2 hours ago", ... }
  ],
  "count": 2
}
```

Ordered most-recently-seen first. Good for "where is everything I own."

**Optional filters:**

| Param | Type | Notes |
|-------|------|-------|
| `limit` | integer | Max sightings to return (1-100, default 1) |
| `since` | float | Unix timestamp - only return sightings after this time |
| `before` | float | Unix timestamp - only return sightings before this time |

---

## Managing Memories

### List what is stored

```
GET /items
GET /items?label=wallet
```

```json
{
  "items": [
    { "id": 3, "label": "wallet", "confidence": 0.87, "ocr_text": "RFID Blocking", "timestamp": 1742700000.0 },
    { "id": 1, "label": "house keys", "confidence": 0.91, "ocr_text": "", "timestamp": 1742600000.0 }
  ],
  "count": 2
}
```

Ordered most-recently-taught first. Multiple rows with the same label means the user taught it more than once.

### Delete (forget) an object

```
DELETE /items/wallet
```

```json
{ "deleted": true, "label": "wallet", "count": 1 }
```

`count` is the number of rows removed. `404` if the label does not exist. The scan pipeline reloads immediately.

**Recommended UX before deleting:**
1. GET /items?label=wallet -> get `timestamp`
2. Confirm: "Are you sure you want to forget wallet? Taught 3 days ago."
3. User confirms -> DELETE /items/wallet

### Rename an object

Two-step flow to handle conflicts:

**Step 1 - send without force:**
```
POST /items/wallet/rename
Content-Type: application/json
{"new_label": "My Wallet"}
```

Success:
```json
{ "renamed": true, "old_label": "wallet", "new_label": "My Wallet", "count": 1, "replaced": 0 }
```

Conflict (a memory named "My Wallet" already exists):
```json
{ "conflict": true, "message": "A memory named \"My Wallet\" already exists (1 item(s)). Send with force=true to overwrite it.", "existing_count": 1 }
```

**Step 2 - user confirms, send with force:**
```json
{ "new_label": "My Wallet", "force": true }
```

`replaced` in the success response is the number of items deleted from the target label before renaming (0 when no conflict).

Other errors: `404` (old label not found), `400` (new_label missing or same as current).

---

## Settings

### User preferences (persisted, survive restarts)

```
GET /user-settings
PATCH /user-settings
```

```json
{
  "performance_mode": "balanced",
  "voice_speed": 1.0,
  "scan_update_location": true,
  "learning_enabled": true,
  "button_layout": "default"
}
```

| Field | Values | Notes |
|-------|--------|-------|
| `performance_mode` | `"fast"` / `"balanced"` / `"accurate"` | fast = no depth (~1s), balanced = depth (~3s), accurate = depth + better models (~5s) |
| `voice_speed` | 0.5 - 2.0 | Apply to your TTS rate for all narrations |
| `scan_update_location` | bool | Whether to prompt the user for room name after each scan |
| `learning_enabled` | bool | Propagated to the scan pipeline immediately - no restart needed |
| `button_layout` | `"default"` / `"swapped"` | UI control layout preference |

PATCH accepts any subset. `400` on type or range errors with per-field error messages.

### ML parameters (persisted, survive restarts)

```
GET /settings
PATCH /settings
```

Useful for surfacing personalization progress in a settings UI:

```json
{
  "enable_learning": true,
  "min_feedback_for_training": 10,
  "projection_head_weight": 1.0,
  "projection_head_ramp_at": 50,
  "projection_head_epochs": 20,
  "head_trained": false,
  "triplet_count": 0,
  "feedback_counts": { "positives": 5, "negatives": 3, "triplets": 3 }
}
```

`head_trained` = personalization model has been trained at least once. `triplet_count` = pairs used in the last training run.

---

## TTS and Narration

- Read `narration` from scan matches verbatim. It is phrased for a blind user.
- Apply `voice_speed` from `/user-settings` to your TTS rate.
- Read matches in array order - they are already left-to-right spatially.
- `"May be a..."` in narration means medium or low confidence. Do not add additional hedging.
- `ocr_text` is raw extracted text. Read it verbatim only if the user specifically asks.

---

## Image Tips

- Send JPEG or HEIC. Both are supported. EXIF orientation is corrected server-side.
- For `/remember`: 1-3 feet. 6 feet is unreliable.
- For `/scan`: wider shots work; the pipeline crops detected regions internally.
- If `/remember` returns `"No object detected."`, the object is probably too small, occluded, or the label is too vague. Try a more specific label or get closer.

---

## Error Handling

| Status | Meaning |
|--------|---------|
| 400 | Missing field, wrong type, or invalid value |
| 401 | Missing or wrong `X-API-Key` |
| 404 | Label not found, or `scan_id` expired (feedback/crop) |
| 500 | Server error - log and show a generic retry message |

All error responses include `{"error": "..."}`. PATCH routes may return `{"errors": {...}}` with per-field messages.

---

## Debug Endpoints

These are for the mobile team during integration. Do not call them from production code. All require `X-API-Key` if set.

### Diagnosing a 400

Hit `/debug/echo` with the exact same request that is failing. It mirrors back everything the server received - content-type, field names, file sizes - without running any inference.

```
POST /debug/echo
(same headers and body as the failing request)
```

```json
{
  "content_type": "multipart/form-data; boundary=abc123",
  "form_fields": { "prompt": "wallet" },
  "files": [{ "field": "image", "filename": "photo.jpg", "mimetype": "image/jpeg", "size_bytes": 48100 }]
}
```

Common issues: field named `"file"` instead of `"image"`, missing `prompt`, wrong content-type.

### Checking system state

```
GET /debug/state
```

Returns pipeline load status, whether OCR service is reachable (and latency), DB item/sighting counts, and active feature flags. Use this to confirm the server is healthy before debugging anything else.

### Checking image quality

```
POST /debug/image
Content-Type: multipart/form-data
image: <file>
```

Returns `is_dark`, `is_blurry`, `luminance`, `blur_score` - the same values the pipeline uses to reject images. Use this when `/remember` or `/scan` silently rejects an image.

### Auto testing pipelines

```
GET /debug/test-remember
GET /debug/test-scan
```

Runs the remember or scan pipeline on a built-in test image (wallet at 1ft). No DB write for test-remember. Proves models loaded and are working. If `detected: false` from test-remember, GroundingDINO is not working.

### Inspecting the database

```
GET /debug/db
```

Returns all taught items, last 20 sightings, and feedback counts. Use between test runs to verify state.

### Resetting test state

```
POST /debug/wipe
Content-Type: application/json
{"confirm": true, "target": "all"}
```

Wipes items, sightings, feedback, and projection head weights. `target` can be `"all"`, `"items"`, `"sightings"`, `"feedback"`, `"weights"`, `"settings"`, or `"user-settings"` to wipe selectively. The scan pipeline reloads immediately.

### Reading logs

```
GET /debug/logs?n=50&event=scan_text_match
```

Returns recent structured log entries. Useful for seeing exactly what happened during a scan or remember call. Useful `event` filters: `remember_ocr`, `scan_text_match`, `text_recognition`.

### Tweaking thresholds live

```
PATCH /debug/config
Content-Type: application/json
{"similarity_threshold": 0.25, "darkness_threshold": 20.0}
```

Directly patches any primitive field in the backend settings. Not persisted across restarts. GET with no body to see all settable fields.

---


---

## Retraining 

Debug retrain. Retrain normally happens on a cron schedule, run this to get immediate results

```
POST /retrain
```

Response:
```json
{ "started": true, "triplets": 15 }
```

If not enough data yet:
```json
{ "started": false, "reason": "insufficient_data", "triplets": 3, "min_required": 10 }
```

If already running:
```json
{ "started": false, "reason": "already_running" }
```

Poll for completion:
```
GET /retrain/status
```
```json
{
  "running": false,
  "last_result": { "trained": true, "triplets": 15, "final_loss": 0.012, "head_weight": 0.3 },
  "error": null
}
```

Poll every 5 seconds until `running: false`. The next scan after training completes will automatically use the updated model, no restart needed.

---

## Endpoint Quick Reference

| Method | Path | When to call |
|--------|------|-------------|
| GET | /health | App launch health check |
| POST | /remember | User teaches a new object |
| POST | /sightings | User names the room after teach or scan |
| POST | /scan | User scans the room |
| GET | /crop | User focuses on a specific scan match |
| POST | /feedback | User confirms or denies a scan match |
| GET | /find | User asks "where is my [object]?" |
| GET | /items | User opens the memory list |
| DELETE | /items/<label> | User deletes a memory |
| POST | /items/<label>/rename | User renames a memory |
| GET | /settings | Read ML/personalization state |
| PATCH | /settings | Update ML parameters |
| GET | /user-settings | Read user preferences (call at app launch) |
| PATCH | /user-settings | Update user preferences |
| GET | /debug/state | Verify server is healthy |
| POST | /debug/echo | Diagnose a 400 - mirrors your request back |
| POST | /debug/image | Check why an image was rejected |
| GET | /debug/db | Inspect database contents |
| GET | /debug/logs | Read server log entries |
| GET | /debug/test-remember | Smoke test remember pipeline |
| GET | /debug/test-scan | Smoke test scan pipeline |
| POST | /debug/wipe | Reset test state |
| PATCH | /debug/config | Patch settings live |
| POST | /retrain | Immediatly retrain for testing |
| GET | /retrain/status | Poll retraining progress |
