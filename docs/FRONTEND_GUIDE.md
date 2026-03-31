# Spaitra Frontend Guide

The frontend is a thin I/O layer. It sends audio and images to the backend via WebSocket. The backend decides what to do, sends back narration and a state update. The frontend speaks the narration and updates its UI state. That is the entire loop.

See [UX.md](UX.md) for narration content, onboarding flow phrasing, and hint text.

---

## Server

| Environment | URL |
|---|---|
| Public (no VPN) | `https://nre5bjw44wddpu2zjg4fe4iehq.srv.us` |
| Tailscale VPN | `http://100.114.39.23:5000` |
| Local | `http://127.0.0.1:5000` |

Every request (HTTP and WebSocket) requires the header or query param:
```
X-API-Key: <key>
```

---

## Startup (HTTP, run once)

```
GET /health
```
If not 200, show offline state and retry.

```
GET /user-settings
```
Store `voice_speed` (float, 0.5-2.0). Apply it to your TTS engine for the whole session.

Then open the WebSocket.

---

## WebSocket

Connect:
```
wss://<host>/socket.io/?key=<api-key>&EIO=4&transport=websocket
```

The server sends a `tts` event and a `session_state` event immediately on connect. Wait for them before showing any UI.

---

## Sending Events

### `chat_start` - user taps the Spaitra button

```json
{}
```

Send when the user taps. The server responds with `listening`. Start recording audio after you receive `listening`.

### `chat_stop` - user cancels before speaking

```json
{}
```

Send if the user taps again to cancel before sending audio. Do not send `audio`.

### `audio` - user finished speaking

```json
{
  "audio": "<base64 webm or ogg>",
  "image": "<base64 jpeg, include when camera is open>",
  "focal_length_px": 3094.0
}
```

Send when recording ends. Always include `audio`. Include `image` and `focal_length_px` when the camera is active (scan or teach flows).

Focal length formula: `focal_length_px = (focalLengthMm / sensorWidthMm) * imageWidthPx`

### `navigate` - user swipes through scan results

```json
{ "direction": "next" }
{ "direction": "prev" }
```

Send on swipe left/right in the scan results view. The server responds with `tts` narrating the newly focused item.

---

## Receiving Events

Handle all of these. Ignore anything else.

### `tts` -> speak narration

```json
{ "narration": "...", "next_state": "focused_on_item" }
```

Speak `narration` via TTS at the stored `voice_speed`. If TTS is already playing and the user taps the Spaitra button, stop it and start recording immediately.

### `session_state` -> update UI state

```json
{
  "current_mode": "focused_on_item",
  "context": {
    "scan_id": "abc123",
    "label": "wallet",
    "item_index": 0,
    "onboarding_phase": ""
  }
}
```

This is your authoritative state. Store `current_mode` and `context`. Update UI based on the mode (see State Machine below). This arrives after every user event and on connect.

### `listening` -> show recording UI

```json
{ "state": "idle", "prompt": "Scan, teach something new, or ask me anything." }
```

Set recording state to active. Display `prompt` as a listening hint. The prompt is already phrased for the user's current state - show it as-is.

### `listening_stopped` -> hide recording UI

```json
{ "state": "idle" }
```

User canceled. Reset recording UI.

### `control` -> open camera

```json
{ "action": "request_image", "context": "scan" }
```

When `action == "request_image"`: activate camera, capture a JPEG frame, include it as `image` in the next `audio` event you send.

### `action_result` -> update content views

```json
{ "type": "scan", "data": { "scan_id": "abc123", "matches": [...] } }
```

Store `scan_id` from scan results. You need it for `GET /crop`. The narration for scan results comes separately via `tts` - do not read `matches` yourself.

Other types: `find`, `remember`, `set_location`, `item_focus`, `open_settings`, `navigate_back`. Route these to the appropriate UI transition.

### `transcription` -> live display (optional)

```json
{ "text": "where is my wallet" }
```

Optionally display what was heard before the result arrives. Cosmetic only.

### `error` -> show error

```json
{ "code": "transcription_failed", "message": "..." }
```

Show or speak `message`. Reset recording UI.

---

## State Machine

Track `current_mode` from `session_state`. Show different UI per state.

| State | What to show |
|---|---|
| `idle` | Home screen. Spaitra button visible. |
| `onboarding_teach` | Home screen. Server is prompting for item teaching. |
| `onboarding_await_scan` | Home screen with camera hint. |
| `awaiting_image` | Activate camera. Capture on next audio send. |
| `awaiting_location` | Listening for room name. No camera needed. |
| `awaiting_confirmation` | Listening for yes/no. |
| `focused_on_item` | Scan results overlay. Enable swipe navigation. |

The server drives all transitions. You follow `current_mode`.

---

## Settings Screen (HTTP)

```
GET /user-settings      -> read current settings
PATCH /user-settings    -> update one or more fields
```

Expose two controls:
- Performance mode: `fast` / `balanced` / `accurate`
- Voice speed: 0.5 - 2.0

PATCH with only the changed field. Changes apply immediately server-side.

---

## Memory List Screen (HTTP)

```
GET /items              -> list all taught items
DELETE /items/<label>   -> delete an item
POST /items/<label>/rename  body: { "new_label": "..." }
```

Rename returns 409 if the new name already exists. Resend with `"force": true` to overwrite.

---

## Scan Image (HTTP)

```
GET /crop?scan_id=<id>&index=<0-based>
```

Returns raw JPEG of the matched object. Call this only when displaying a cropped scan result. Scans are cached for the last 50 results.

---

## Debug (HTTP, dev only)

```
GET /debug/state        -> pipeline health, model load status
GET /debug/test-remember
GET /debug/test-scan
POST /debug/wipe        body: { "confirm": true, "target": "all" }
```

Do not call these from production code.
