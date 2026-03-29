# Spaitra API Guide

Integration guide for the Spaitra API. Works for frontend, mobile, and backend developers.

---

## The Experience

Spaitra is a spatial memory aid for blind users. The user teaches the app what their things look like; the app tells them where those things are. For someone who cannot quickly re-scan their environment, this replaces the constant mental load of tracking object locations.

The frontend's job is to make this feel immediate. No mode menus, no confirmation dialogs, no help screens. The user speaks or performs a touch gesture, the app responds, and the item is found. Every design decision follows from keeping that loop as tight as possible, following Steve Job's design philosophy.

The first time someone opens the app, they do not see a tutorial. The app has them experience the product: teach two real items from their environment, scan, and hear the app announce what it found with distance and direction. They already know where they placed those items, so when the app names them back, the result is immediately verifiable and immediately impressive. This is designed to immediatly spark the curiosity so they learn the rest of the features naturally. Little voice hints will be planted to nudge the user to use the more specific features that they haven't tried yet, like "try saying 'export this.'" Since they are blind, voice feedback is a necessity with every user interaction. The plan is to have the user learn the app on their own through the quick demo at the start, and little moments of discovery of new uses for the app. The onboarding emphasizes for the user's experience of those moment so it doesn't feel like a tutorial or lesson, but a seamless addition to their lives. 

---

## Setup

### Base URL

Two ways to reach the server - use whichever works for you:

| Method | URL | When to use |
|--------|-----|-------------|
| Public (no VPN) | `https://nre5bjw44wddpu2zjg4fe4iehq.srv.us` | Default for frontend dev |
| Tailscale (VPN) | `http://100.114.39.23:5000` | Team members on the VPN for dev |
| Local dev | `http://127.0.0.1:5000` | When running the server on your own machine |

```bash
# Set once for curl examples below
BASE="https://nre5bjw44wddpu2zjg4fe4iehq.srv.us"
KEY="<your-api-key>"
```

The API key is shared with the team. If you do not have it, ask in the team channel or check
`/opt/spaitra/.env` on the server (`grep ^API_KEY= /opt/spaitra/.env | cut -d= -f2-`).

### Auth

Every request except `GET /health` requires:
```
X-API-Key: <key>
```

Missing or wrong key -> `401`. Upload cap: 50MB per request.

---

## Smoke Tests

Run these first to confirm the server is up and your key works.

```bash
# Health check (no auth needed)
curl "$BASE/health"
# Expected: {"status": "ok"}

# Confirm auth works
curl "$BASE/debug/state" -H "X-API-Key: $KEY" | python3 -m json.tool
# Returns: pipeline load status, OCR health, DB counts, feature flags

# Test remember pipeline end-to-end (uses a built-in test image, no DB write)
curl "$BASE/debug/test-remember" -H "X-API-Key: $KEY" | python3 -m json.tool
# Expected: {"detected": true, ...}

# Test scan pipeline end-to-end
curl "$BASE/debug/test-scan" -H "X-API-Key: $KEY" | python3 -m json.tool
```

If `detected: false` from `test-remember`, GroundingDINO is not loaded. If `debug/state` shows
the OCR service unreachable, the OCR microservice is down (core API still works without it).

---

## Quick Reference

| Method | Path | When to call |
|--------|------|-------------|
| GET | /health | App launch health check |
| POST | /remember | User teaches a new object |
| POST | /sightings | User names the room after teach or scan |
| POST | /scan | User scans the room |
| GET | /crop | User focuses on a specific scan match |
| POST | /feedback | User confirms or denies a scan match |
| POST | /ask | User asks anything in natural language |
| POST | /item/ask | User asks about a specific focused scan item |
| GET | /find | Direct label lookup - "where is my [object]?" |
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
| GET | /debug/perf | Read performance summary (perf/vram/runtime counters) |
| GET | /debug/test-remember | Smoke test remember pipeline |
| GET | /debug/test-scan | Smoke test scan pipeline |
| POST | /debug/wipe | Reset test state |
| PATCH | /debug/config | Patch settings live |
| POST | /retrain | Immediately retrain for testing |
| GET | /retrain/status | Poll retraining progress |

---

## What to Store Client-Side

**`scan_id`** - returned by every `POST /scan` response. Store it immediately. Needed for:
- `GET /crop` - fetch the cropped image of a match
- `POST /feedback` - tell the backend whether a match was correct

The server caches the last 50 scans. Send feedback during the same scan session before the user navigates away.

**`voice_speed`** - read once from `GET /user-settings` at app launch. Apply to your TTS engine.

**Current match index** - 0-based position in the `matches` array from the last scan. Client-owned; server does not track it.

---

## First Launch: Onboarding

Check item count at launch. If items exist, skip onboarding - the user has been here before.

```
GET /items
```

If `count == 0`, start onboarding. There is no separate onboarding screen. The onboarding is the first real Teach + Scan + Ask session. The user teaches actual items from their environment and finds them immediately after.

### Phase 1: Teach two items

App says: "I remember where your things are. Grab two items near you, and let's start."

User points camera at the first item and says the name. Run POST /remember as normal. On success: "Got it. Now teach me the second one."

On second success: "Good. Place both items somewhere in the room."

Brief pause, then: "Tap or say 'Scan' when ready."

### Phase 2: Scan

User initiates scan. Run POST /scan, store scan_id.

Read all narration strings in array order. Example: "Wallet, to your left, 2.3 feet. Keys, straight ahead, 4 feet."

After narration, prompt for room name: "What room is this?" (haptic single pulse, enter listening state)

User says room name. Call POST /sightings with room_name and all matched labels from the scan. This links the room to each detection and is what makes the Ask demo in phase 4 work.

### Phase 3: Swipe introduction

App says: "Swipe right to browse each item."

On swipe, re-read the match narration from the cached scan response. No API call. Format: "[label]. [direction]. [distance] feet." Narration is interruptible - if the user swipes again before it finishes, cut it and start the next item.

After one full cycle: "Swipe left to go back. That is how scan works."

### Phase 4: Ask demo

Return to home listen state. App says: "Now ask me where you left your [label of first item taught], or use specifics about the item or text in it instead."

User asks naturally. Call POST /ask with the spoken query.

Read the narration response. Example: "Your wallet is in the living room, to your left."

Onboarding ends. App says: "You are all set, 'Teach', 'Scan', 'Find', or 'Ask' anything."

Do not collect any feedback during onboarding. 

### Phase 5: Ongoing

Hints for small features like "trying saying export, or holding down when in a scan" appear occasionally for features the user hasn't used

---

## UX Patterns

### Modes

| Voice command | Action |
|---|---|
| "Teach" | Teach mode - app prompts for item label |
| "Scan" | Trigger scan |
| "Find" | App asks "What item?" - direct lookup via GET /find |
| "Ask" | App asks "What do you want to know?" - semantic query via POST /ask |
| "Back" | Return to home from any mode |
| "Settings" | Open settings |

Scan is triggered by voice ("Scan") or by a tap button on the home screen. The tap zone should cover the full lower half of the screen, not a small target - users cannot see where to tap.

### Haptics

| Event | Haptic | Sound |
|---|---|---|
| Command recognized | Single pulse | Beep |
| Mode entered | Single pulse | Mode prompt |
| Listening for input | Continuous soft buzz | (none) |
| Action complete | Double pulse | Confirmation phrase |
| Scan complete | Double pulse | Narration starts |
| Error / no match | Long buzz | Error message |
| Location saved | Double pulse | "Saved to [room]" |

### Scan Mode Gestures

| Gesture | Action |
|---|---|
| Tap | Rescan - interrupts narration, new scan immediately |
| Swipe right | Next detection - re-reads its narration |
| Swipe left | Previous detection - re-reads its narration |
| Double-tap | Save location for current detection |
| Hold ~1 sec | Export current detection |

On swipe, read the narration from the cached matches array. No API call. Narration is interruptible on the next swipe.

Double-tap and hold are power user gestures. Do not mention or demonstrate them during onboarding.

### Feedback Model

Feedback is opt-in and voice-only. The user never needs to confirm a correct detection.

| Scenario | Action |
|---|---|
| Detection is correct | No action. Do not prompt. Do not log. |
| Detection is wrong | User says "wrong" - call POST /feedback with `"wrong"` |

Do not use gesture-based feedback. Tap is already reserved for rescan. Do not prompt for feedback during onboarding. The backend pairs explicit negatives against known positives for triplet training - clean signal matters more than volume.

### Room Naming

Call POST /sightings when the user explicitly names a room:
- After a scan, when the app asks "What room is this?"
- After a double-tap on a scan result and the user speaks a location

Do not call POST /sightings automatically without a user-spoken room name.

---

## Teach Mode

The user wants the app to remember a new object: capture -> POST /remember -> handle signals -> optionally record location.

### Step 1: Capture

Capture 3-5 frames in quick succession (~500ms apart). The backend picks the best frame.

### Step 2: POST /remember

```
POST /remember
Content-Type: multipart/form-data
```

| Field | Type | Notes |
|-------|------|-------|
| `images[]` | file[] | 2-5 frames from the burst (recommended) |
| `image` | file | Single frame (fallback) |
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

Check in this order:

**1. `is_dark: true`** - highest priority.
```
speak: "Image is too dark. Increase lighting or turn on the flashlight."
```

**2. `success: false` + `is_dark: false`**
```
is_blurry: true  -> "Hold steady and try again"
is_blurry: false -> "Could not detect - try a different angle, better lighting, or a more specific label"
```

**3. `success: true`**
```
replaced_previous: true  -> say "Updated [label]"
replaced_previous: false -> say "[label] saved"
```
If `detection_quality` is `"low"` or `"medium"`, announce `detection_hint` as a soft warning. Object is saved regardless.

### Step 4: Record location (optional)

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

`room_name` is normalized server-side ("In The Kitchen" -> "kitchen"). Omit if user did not name the room.

Response: `{ "saved": 1, "labels": ["wallet"], "room_name": "kitchen" }`

---

## Scan Mode

Capture -> POST /scan -> store `scan_id` -> navigate matches -> (optional) crop, feedback, location update.

### Step 1: POST /scan

```
POST /scan
Content-Type: multipart/form-data
```

| Field | Type | Notes |
|-------|------|-------|
| `image` | file | The scene image |
| `focal_length_px` | float | Enables calibrated depth. Omit or send `0` to disable. |

**Focal length formula:**
```
focal_length_px = (focalLengthMm / sensorWidthMm) * imageWidthPx
```
iPhone 15 Plus reference: `f_mm=6.24, sensor_width=8.64mm` -> `focal_length_px = 3094.0`

### Step 2: Check for darkness

Before touching `matches`, check the top-level `is_dark` field:
```json
{
  "matches": [],
  "count": 0,
  "is_dark": true,
  "darkness_level": 8.2,
  "message": "Image is too dark for detection."
}
```
`is_dark` is at the top level, not inside a match. If true, speak the message and stop.

### Step 3: Store scan_id and navigate matches

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

**Store `scan_id` immediately.** `matches` is ordered left-to-right spatially.
Read `narration` aloud for each match in array order.
`distance_ft` is only present when depth is enabled.

**Direction zones** (narration already reflects this):

| Value | Where in the frame |
|-------|-------------------|
| `"to your left"` | Far left (< 25%) |
| `"slightly left"` | Left of center |
| `"ahead"` | Center |
| `"slightly right"` | Right of center |
| `"to your right"` | Far right (> 75%) |

**Confidence tiers:**

| Value | Similarity range |
|-------|-----------------|
| `"high"` | >= 0.35 |
| `"medium"` | 0.25 - 0.35 |
| `"low"` | 0.2 - 0.25 |

### Step 4: Fetch a crop (on demand)

```
GET /crop?scan_id=a3f9c2b1d4e5f6a7&index=0
```

`index` is the 0-based position in `matches`. Returns raw JPEG.
Returns `404` if `scan_id` has expired or `index` is out of range.

### Step 5: Collect feedback (optional but important)

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

`feedback` must be `"correct"` or `"wrong"`.

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

`404` means `scan_id` expired. Send feedback before the user does 50 more scans.

### Step 6: Update location (optional)

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

`direction`, `distance_ft`, and `similarity` come from the match objects. Do NOT include matches the user gave negative feedback for.

---

## Ask Mode

### Open-ended natural language: POST /ask

When the user asks anything in free speech and the frontend cannot cleanly extract a known label, send the raw query here.

```
POST /ask
Content-Type: application/json
```
```json
{ "query": "where did I put my money last week?" }
```

**Found:**
```json
{
  "query": "where did I put my money last week?",
  "search_term": "money",
  "ollama_used": true,
  "found": true,
  "matched_label": "wallet",
  "matched_by": "fuzzy_label",
  "narration": "Your wallet is in the kitchen, to your left. Last seen 3 days ago.",
  "last_sighting": {
    "id": 41,
    "label": "wallet",
    "timestamp": 1742860412.3,
    "last_seen": "3 days ago",
    "direction": "to your left",
    "distance_ft": 2.3,
    "room_name": "kitchen"
  },
  "sightings": [...]
}
```

**Not found:**
```json
{
  "query": "...",
  "search_term": "...",
  "ollama_used": true,
  "found": false,
  "narration": "I couldn't find anything matching that in your memory."
}
```

`matched_by` values:
- `"exact"` - label matched directly
- `"fuzzy_label"` - matched via text embedding similarity on label names
- `"ocr"` - matched via OCR text content of a taught item

`ollama_used: false` means Ollama was unavailable; search ran on raw query text.

**When to call:** any voice query where the user is not tapping a specific item. "Where's my wallet?", "The receipt with the chair", "What did I leave in the kitchen?", anything.

---

### Item-context queries: POST /item/ask

When the user is focused on a specific scan result and asks about it. Frontend already has `scan_id` and `label`.

```
POST /item/ask
Content-Type: application/json
```
```json
{
  "scan_id": "a3f9c2b1d4e5f6a7",
  "label": "wallet",
  "query": "read the text in this"
}
```

**read_ocr / export_ocr response:**
```json
{
  "action": "read_ocr",
  "label": "wallet",
  "ocr_text": "RFID Blocking",
  "narration": "The text says: RFID Blocking.",
  "export": false
}
```
Set `export: true` when the user wants to copy/share the text rather than just hear it.

**rename response:**
```json
{
  "action": "rename",
  "old_label": "wallet",
  "new_label": "my wallet",
  "narration": "Renamed to my wallet.",
  "replaced_existing": false
}
```
Rename auto-replaces any existing memory with the same name - no confirmation step. If the new name is the same as the current name:
```json
{ "action": "rename", "narration": "That's already called wallet.", "unchanged": true }
```

**find response:**
```json
{
  "action": "find",
  "label": "wallet",
  "found": true,
  "narration": "Your wallet is in the kitchen, to your left. Last seen 5 minutes ago.",
  "last_sighting": { ... }
}
```

**describe response (not yet available):**
```json
{ "action": "describe", "narration": "Visual description is not available yet.", "deferred": true }
```

**When to call:** user is scrolling through scan results and asks something about the item currently on screen.

---

### Direct label lookup: GET /find

```
GET /find?label=wallet
```

**Found:**
```json
{
  "label": "my wallet",
  "found": true,
  "matched_label": "wallet",
  "matched_by": "fuzzy_label",
  "narration": "Your wallet is in the kitchen, to your left. Last seen 3 minutes ago.",
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

`matched_label` appears only when fuzzy matching was used. `room_name` is null until the user has confirmed a location via `POST /sightings`.

**Not found:** `{ "label": "wallet", "found": false, "sightings": [] }`

**All objects:** `GET /find` (omit `label`) - ordered most-recently-seen first.

**Room query** - all items last seen in a room:
```
GET /find?room=kitchen
```
```json
{ "room": "kitchen", "results": [...], "count": 2 }
```
Room name is normalized server-side ("In The Kitchen" → "kitchen"). Each result includes a `narration` field.

**Optional filters:** `label` (str), `room` (str), `limit` (int), `since` (Unix timestamp), `before` (Unix timestamp).

---

## Managing Memories

### List

```
GET /items
GET /items?label=wallet
```

```json
{
  "items": [
    { "id": 3, "label": "wallet", "confidence": 0.87, "ocr_text": "RFID Blocking", "timestamp": 1742700000.0 }
  ],
  "count": 1
}
```

### Delete

```
DELETE /items/wallet
```
```json
{ "deleted": true, "label": "wallet", "count": 1 }
```

`count` = rows removed. `404` if label not found.

### Rename (two-step)

**Step 1:**
```
POST /items/wallet/rename
{ "new_label": "My Wallet" }
```

Success: `{ "renamed": true, "old_label": "wallet", "new_label": "My Wallet", "count": 1, "replaced": 0 }`

Conflict (409): `{ "conflict": true, "message": "...", "existing_count": 1 }`

**Step 2 (if conflict, user confirmed):**
```json
{ "new_label": "My Wallet", "force": true }
```

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
  "learning_enabled": true,
  "button_layout": "default"
}
```

| Field | Values | Notes |
|-------|--------|-------|
| `performance_mode` | `"fast"` / `"balanced"` / `"accurate"` | fast = no depth + no LLM query fallback, balanced = depth + LLM query fallback, accurate = depth + LLM query fallback |
| `voice_speed` | 0.5 - 2.0 | Apply to your TTS rate |
| `learning_enabled` | bool | Propagated to scan pipeline immediately |
| `button_layout` | `"default"` / `"swapped"` | UI layout preference |

PATCH accepts any subset. `400` on type or range errors.

### ML parameters

```
GET /settings
PATCH /settings
```

```json
{
  "enable_learning": true,
  "min_feedback_for_training": 10,
  "projection_head_weight": 1.0,
  "head_trained": false,
  "triplet_count": 0,
  "feedback_counts": { "positives": 5, "negatives": 3, "triplets": 3 }
}
```

`head_trained` = personalization model has been trained at least once.

---

## TTS and Narration

- Read `narration` from scan matches verbatim. It is phrased for a blind user.
- Apply `voice_speed` from `/user-settings` to your TTS rate.
- Read matches in array order - they are left-to-right spatially.
- `"May be a..."` in narration means medium or low confidence. Do not add hedging.
- `ocr_text` is raw extracted text. Read it only if the user specifically asks.

---

## Image Tips

- Send JPEG or HEIC. Both supported. EXIF orientation corrected server-side.
- For `/remember`: 1-3 feet. 6 feet is unreliable.
- For `/scan`: wider shots work; the pipeline crops detected regions internally.

---

## Error Handling

| Status | Meaning |
|--------|---------|
| 400 | Missing field, wrong type, or invalid value |
| 401 | Missing or wrong `X-API-Key` |
| 404 | Label not found, or `scan_id` expired |
| 500 | Server error - log and show a generic retry message |

All error responses include `{"error": "..."}`. PATCH routes may return `{"errors": {...}}` with per-field messages.

---

## Debug Endpoints

For integration and debugging. Do not call from production code. All require `X-API-Key`.

### Diagnosing a 400

```
POST /debug/echo
(same headers and body as the failing request)
```

Returns exactly what the server received - content-type, field names, file sizes - without running inference. Common issues: field named `"file"` instead of `"image"`, missing `prompt`, wrong content-type.

### System state

```
GET /debug/state
```

GPU status, model load status, OCR service reachability, DB counts, feature flags.

### Image quality check

```
POST /debug/image
Content-Type: multipart/form-data
image: <file>
```

Returns `is_dark`, `is_blurry`, `luminance`, `blur_score` - same values the pipeline uses.

### Pipeline smoke tests

```
GET /debug/test-remember
GET /debug/test-scan
```

Runs pipeline on a built-in test image. No DB write for test-remember.

### Database inspection

```
GET /debug/db
```

All taught items, last 20 sightings, feedback counts.

### Reset test state

```
POST /debug/wipe
Content-Type: application/json
{"confirm": true, "target": "all"}
```

`target` options: `"all"`, `"items"`, `"sightings"`, `"feedback"`, `"weights"`, `"settings"`, `"user-settings"`.

### Log tail

```
GET /debug/logs?n=50&event=scan_text_match
```

Useful `event` filters: `remember_ocr`, `scan_text_match`, `text_recognition`.

Use `source=important` for warning and error only logs, or `source=crash` to read crash blocks.

### Performance summary

```
GET /debug/perf
```

Use this when tuning scan latency. Key fields include stage timing aggregates (`prepare_ms`, `detect_ms`, `embed_ms`, `ocr_ms`, `match_ms`) and resource peaks (RAM, swap, VRAM, CPU temp). For OCR batching checks, monitor `ocr_batch_requests`, `ocr_batch_items`, and `ocr_batch_non_empty` in recent perf records.

### Live threshold patch

```
PATCH /debug/config
Content-Type: application/json
{"similarity_threshold": 0.25}
```

Not persisted across restarts. GET with no body to see all patchable fields.

---

## Retraining

```
POST /retrain
```

Response:
```json
{ "started": true, "triplets": 15 }
```

If not enough data: `{ "started": false, "reason": "insufficient_data", "triplets": 3, "min_required": 10 }`

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

Poll every 5 seconds until `running: false`.

---

## Server Management

For SSH access, full setup steps, and deployment: see [SERVER_ACCESS.md](SERVER_ACCESS.md).

Quick reference for anyone who needs to restart the server:

```bash
# SSH into the server (Tailscale required)
ssh dev@100.114.39.23

# Restart after a code change
cd /opt/spaitra/TSA-soft-dev-backend-2026
sudo -u spaitra git pull
sudo systemctl restart spaitra-core

# Check service status
systemctl status spaitra-core spaitra-ocr

# Follow logs
journalctl -u spaitra-core -f
```

Server info:
- Host: `Spaitra` / Tailscale IP: `100.114.39.23`
- Public URL: `https://nre5bjw44wddpu2zjg4fe4iehq.srv.us`
- OS: Debian 13 / GPU: NVIDIA GTX 1060 6 GB
- Core API: port 5000 (localhost, exposed via srv.us tunnel)
- OCR service: port 8001 (localhost only)

Environment variables on the server (`/opt/spaitra/.env`):
```dotenv
API_KEY=<secret>              # required for all routes except /health
ENABLE_DEPTH=1                # set to 0 to skip depth estimation (~2 GB VRAM)
ENABLE_OCR=1                  # set to 0 to skip OCR
ENABLE_LEARNING=1             # set to 0 to disable projection head
OCR_SERVICE_URL=http://127.0.0.1:8001/ocr
SAVE_VRAM=1                   # enabled on this server (6 GB GPU)
```
