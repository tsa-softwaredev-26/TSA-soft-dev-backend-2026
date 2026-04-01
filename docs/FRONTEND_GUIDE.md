# Spaitra Frontend Guide

The frontend is an I/O layer. It sends audio and images to the backend via WebSocket. The backend decides what to do, sends back narration and a state update. The frontend speaks the narration and updates its UI state. That is the entire loop.

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

Required query params:
- `key`: same API key used for HTTP requests
- `EIO=4`: socket.io protocol version required by this backend
- `transport=websocket`: force websocket transport

Server sends `tts` and `session_state` immediately on connect. Wait before showing UI.

---

## Sending Events

### `chat_start` - user presses and holds the top-right chat button

```json
{}
```

Send when the user presses and holds the dedicated top-right chat button. The server responds with `listening`. Start recording audio after you receive `listening`.

### `chat_stop` - user releases the chat button or cancels before speaking

```json
{}
```

Send when the user releases the held chat button or cancels before sending audio. Do not send `audio`.

### `audio` - user finished speaking

```json
{
  "audio": "<base64 webm or ogg>",
  "image": "<base64 jpeg, include when camera is open>",
  "focal_length_px": 3094.0
}
```

Send when recording ends. Always include `audio`.

Include `image` and `focal_length_px` when the camera is active or when responding to `control.request_image` for:
- scan
- teach/remember
- idle describe requests (`context: "describe"`)

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

Speak `narration` via TTS at the stored `voice_speed`. If TTS is already playing and the user presses the top-right chat button, stop it and start recording immediately.

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

Set recording state to active. Display `prompt` as a listening hint. Prompt is phrased for current mode; display as-is.

### `listening_stopped` -> hide recording UI

```json
{ "state": "idle" }
```

User canceled. Reset recording UI.

### `control` -> open camera

```json
{ "action": "request_image", "context": "scan" }
{ "action": "request_image", "context": "describe" }
```

When `action == "request_image"`: activate camera, capture a JPEG frame, include it as `image` in the next `audio` event you send.

`context` tells you why image is being requested:
- `scan`: scan flow
- `describe`: idle describe flow ("describe what I am looking at" / "describe this wallet")
- missing or other: treat as generic camera-required step

### `action_result` -> update content views

```json
{ "type": "scan", "data": { "scan_id": "abc123", "matches": [...] } }
```

Store `scan_id` from scan results. You need it for `GET /crop`. The narration for scan results comes separately via `tts` - do not read `matches` yourself.

Other types: `find`, `remember`, `set_location`, `item_focus`, `open_settings`, `navigate_back`, `item_ask`, `ask`.

Route by `type`, not by assumptions about current mode.

### `transcription` -> live display (optional)

```json
{
  "text": "where is my wallet",
  "context_used": true,
  "context_policy": "idle_home",
  "context_state_id": "idle"
}
```

`text` is the only field required for UI.

`context_used`, `context_policy`, and `context_state_id` are diagnostics only (safe to ignore in UI logic).

Current policy ids you may see in diagnostics include:
- `idle_home`
- `focused_item_scan_browse`
- `awaiting_location_capture`
- `awaiting_confirmation_binary`
- `onboarding_teach_step`
- `onboarding_scan_step`
- `awaiting_image_scan`
- `awaiting_image_remember`
- `awaiting_image_describe`
- `awaiting_image_generic`

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

The server drives all transitions. Use `current_mode` exactly as sent in `session_state`; do not remap state names on the client.

During onboarding, also track `context.onboarding_phase` and treat it as authoritative progression:
- `teach_1`: first teach step only
- `teach_2`: second teach step only
- `await_scan`: scan step only
- `ask`: swipe guidance step
- `ask_prompted`: final ask/find completion step

Do not bypass onboarding steps client-side. The backend enforces order and will keep users in-step with narration prompts.

---

## Frontend behavior in focused item mode

When `current_mode == focused_on_item`, keep these behaviors available:
- swipe navigation (`navigate`)
- hold-to-talk item commands

Expected voice examples in focused mode include:
- `wrong` (negative feedback)
- `right` or `correct` (positive feedback)
- item ask commands like read/export/describe/rename/find
- if user says teach/remember while focused, backend will guide them to return home first

No special client parsing is required. Just send `audio` and render returned events.

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
POST /debug/echo
POST /debug/image
GET /debug/db
GET /debug/logs
GET /debug/perf
GET /debug/test-remember
GET /debug/test-scan
POST /debug/wipe        body: { "confirm": true, "target": "all" }
PATCH /debug/config
```

Do not call these from production code.
