<!--
UX implementation surfaces (keep these in sync with this doc):
- docs/FRONTEND_GUIDE.md
- src/visual_memory/api/voice_session.py
- src/visual_memory/api/routes/voice_ws.py
- src/visual_memory/api/routes/voice.py
- src/visual_memory/utils/voice_state_context.py
- src/visual_memory/api/routes/ask.py
- src/visual_memory/api/routes/find.py
- src/visual_memory/api/routes/item_ask.py
- src/visual_memory/api/routes/remember.py
- src/visual_memory/api/routes/scan.py
- src/visual_memory/api/routes/transcribe.py
-->

# Spaitra UX Reference

This file is the UX architecture for voice flow, state behavior, narration style, and user guidance.

If you change UX behavior here, code should be updated to match, and vice versa.

## Design philosophy

- Voice-first, low-friction, one core action per interaction.
- No tutorial walls; onboarding is real usage.
- Backend is authoritative for voice behavior and spoken phrasing.
- Frontend is a thin transport/render layer: capture input, play narration, reflect server state.
- Keep language concise, natural, and accessible for blind users.


## System responsibility split

### Backend owns

- State machine and transitions
- Command classification and mode gating
- Narration wording
- Hint scheduling
- Context contract used for model biasing

### Frontend owns

- Recording and camera capture UX
- Playing `tts.narration` verbatim
- Rendering `session_state.current_mode` and `context`
- Sending interaction events (`chat_start`, `audio`, `navigate`, etc.)


## UX state machine (authoritative)

Canonical states:

- `idle`
- `awaiting_image`
- `awaiting_location`
- `awaiting_confirmation`
- `focused_on_item`
- `onboarding_teach`
- `onboarding_await_scan`

State persistence is per WebSocket session (`VoiceSession`).

### Core transitions

- Connect:
  - empty DB -> `onboarding_teach` (phase `teach_1`)
  - non-empty DB -> `idle`
- Teach flow:
  - teach intent without image -> `awaiting_image`
  - successful remember -> `awaiting_location`
  - location set:
    - onboarding teach 1 -> `onboarding_teach` (phase `teach_2`)
    - onboarding teach 2 -> `onboarding_await_scan` (phase `await_scan`)
    - normal mode -> `idle`
- Scan flow:
  - scan intent without image -> `awaiting_image`
  - scan with matches -> `focused_on_item`
  - scan with no matches -> `idle` (or stay onboarding scan mode when applicable)
- Confirmation flow:
  - pending confirmation -> `awaiting_confirmation`
  - resolved -> `idle`
- Navigation:
  - back/settings from focused or onboarding flows -> `idle`


## Session context model

Runtime context keys include:

- `pending_action` (`scan` or `remember`)
- `label` and/or `current_label`
- `scan_id`
- `scan_matches`
- `item_index`
- `onboarding_phase` (`teach_1`, `teach_2`, `await_scan`, `ask`, `ask_prompted`)
- `onboarding_item_1`, `onboarding_item_2`, `onboarding_ask_label`
- `scan_count`, `teach_count`
- `hints_given`
- optional `pending_confirmation`


## State contract for context biasing

`build_state_contract(...)` normalizes mode + context into a stable payload used by:

- transcription (`transcribe` context hints)
- ask/find/item flows
- LLM extraction helpers

Contract shape (high level):

- `current_mode` / `mode`
- `context.scan_id`
- `context.label`
- `context.item_index`
- `context.onboarding_phase`
- optional `context.pending_action`
- `known_labels` list

This contract is the boundary for context-aware model behavior.


## Interaction protocol (high level)

### Inbound events

- `chat_start` -> server returns `listening` prompt
- `chat_stop` -> cancel recording/listening
- `audio` -> primary command input (plus optional image + focal length)
- `navigate` -> next/previous item focus

### Outbound events

- `tts` -> narration + `next_state`
- `session_state` -> authoritative mode + context snapshot
- `listening` / `listening_stopped`
- `control` (`request_image`) when image is required
- `action_result` for structured outcomes (`scan`, `remember`, `find`, `item_ask`, nav)
- `error`
- optional `transcription`


## Command behavior rules

### Global

- Empty utterance defaults to ask-style fallback.
- `back/home/cancel` maps to navigation back behavior.
- settings keywords map to open settings behavior.
- scan/remember/find/ask are always available in `idle`.

### Mode-specific gating

- `onboarding_teach`: accept teach flow (plus back), reject unrelated commands.
- `onboarding_await_scan`: accept scan/back/settings; reject unrelated commands.
- `awaiting_location`: interpret utterance as room/location.
- `awaiting_confirmation`: interpret utterance as yes/no.
- `focused_on_item`: prioritize item actions and `wrong` feedback; support navigation.

Mode gating should return clear "command unavailable" guidance where applicable.


## Onboarding UX contract

On empty-memory connect:

1. Welcome prompt -> teach first item
2. Teach item 1 -> ask room
3. Teach item 2 -> ask room
4. Prompt scan of both items
5. Successful scan -> focused item browsing intro
6. Ask demo -> completion message -> `idle`

No feedback collection during onboarding.


## Full dialogue script (editable)

This section is the editable UX dialogue source. These lines should stay aligned with runtime behavior.

### 1) Connect and initial readiness

- Empty memory on WebSocket connect:
  - "Welcome to Spaitra. I remember where your things are. Grab two items near you and let's start. Say teach, then the name of the first item."
- Existing memory on WebSocket connect:
  - "Ready."

### 2) Listening prompts (`chat_start`)

- In `onboarding_teach`:
  - "Say teach, then the item name."
- In `onboarding_await_scan`:
  - "Say scan when ready."
- In `awaiting_location`:
  - "Which room is this?"
- In `awaiting_image`:
  - "Hold your phone up."
- In `awaiting_confirmation`:
  - "Say yes to confirm, or no to cancel."
- In `focused_on_item` with label:
  - "Focused on {label}. Ask about it, or say wrong if it was incorrect."
- In `focused_on_item` without label:
  - "Ask about the item, or say wrong if the detection was incorrect."
- Default:
  - "Scan, teach something new, or ask me anything."

### 3) Teach and onboarding narration flow

- Teach success (all modes):
  - "{Label} saved. Which room is this?"
- Awaiting-location reprompt:
  - "What room is this?"
- After location, onboarding phase teach_1:
  - "Got it, {label} in the {location}. Now teach me the second item. Say its name."
- After location, onboarding phase teach_2:
  - "Got it, {label} in the {location}. Great. Place both items somewhere in the room, then say scan."
- After location outside onboarding:
  - "Got it, {label} in the {location}." (with optional appended hint)
- Onboarding scan with no matches:
  - "I did not spot either item. Make sure they are in view and try again."
- Onboarding scan with matches:
  - "{combined_scan_narration}. Now swipe right to browse each item."
- Onboarding swipe demo completion:
  - "{focused_item_narration}. Swipe left to go back. That is how scan works. Now ask me where you left your {onboarding_ask_label_or_first_item}."
- Onboarding ask/find completion:
  - "{answer_narration} You are all set. Say scan, teach, or ask anything."

### 4) Scan, image-request, and navigation narration

- General scan with no matches:
  - "I did not find anything familiar."
- Requesting image for scan/teach:
  - "Hold your phone up."
- While still waiting for image:
  - "Still waiting for an image."
- Remember with missing cached label in awaiting_image:
  - "I lost the item name. Please start again."
- Onboarding scan mode, wrong command:
  - "Say scan when you're ready."
- Teach image guidance:
  - "Point your phone at your {label}."
- Navigate with no scan cache:
  - "No scan results to navigate."
- Focused-item navigation:
  - "{match.narration}" (or label fallback)

### 5) Teach failure narration

- Dark image:
  - "Image is too dark. Increase lighting and try again."
- Blurry image:
  - "Hold steady and try again."
- Generic detection failure:
  - "Could not detect the item. Try a different angle or better lighting."

### 6) Settings/back/feedback/confirmation narration

- Open settings:
  - "Opening settings."
- Navigate back:
  - "Going back."
- Wrong-feedback accepted:
  - "Got it, noted as incorrect."
- Wrong-feedback without active scan:
  - "No active scan to give feedback on."
- Confirmation success:
  - "Done." (WS path) or "Confirmed." (HTTP `/voice` path)
- Confirmation canceled:
  - "Canceled."
- Delete success:
  - "{Label} deleted."
- Delete target missing:
  - "I could not find that item to delete."
- Delete failure:
  - "Could not delete the item."
- Stop action:
  - "Okay, stopping."
- HTTP `/voice` explicit nav actions:
  - "Next item."
  - "Previous item."
  - "Opening settings."
  - "Going back."

### 7) Ask/Find narration

- Unsafe query block:
  - "I can only help with memory-related object lookup requests."
- Ask/find miss:
  - "I couldn't find anything matching that in your memory."
  - "I could not find {search_label} in your memory."
- Ask/find success narration template:
  - "Your {label} is in the {room}, {direction}, {distance:.1f} feet away. Last seen {last_seen}."
  - "Your {label} is {direction}. Last seen {last_seen}."
  - "Your {label} was last seen {last_seen}."

### 8) Item-focused narration (`item_ask`)

- No OCR text:
  - "There's no text stored for {label}."
- OCR read/export:
  - "The text says: {ocr_text}"
- Visual question item missing:
  - "I do not have {label} in memory yet."
- Visual question image missing:
  - "I do not have an image stored for {label}."
- VLM timeout:
  - "That took too long. Please try a shorter question."
- VLM file missing:
  - "I could not find the image file for {label}."
- VLM generic failure:
  - "I could not answer that question about this item right now."
- OCR question no text:
  - "There is no stored text for {label}."
- OCR amount answer:
  - "The amount is {answer}."
- OCR date answer:
  - "The date is {answer}."
- Rename parse failure:
  - "I couldn't understand the new name. Try saying: rename this to [name]."
- Rename unchanged:
  - "That's already called {label}."
- Rename DB sync warning:
  - "Renamed, but I could not sync scan state yet."
- Rename success:
  - "Renamed to {new_label}."
- Item-focused find miss:
  - "I haven't seen {label} in any known location yet."
- Describe minimal fallback:
  - "This is your {label}. I can read text on it: {ocr_snippet}"
  - "This is your {label}."

### 9) Scan-result narration templates (pipeline-generated)

Non-depth mode:

- High confidence:
  - "{Label} look down, {direction}."
  - "{Label} look up, {direction}."
  - "{Label} {direction}."
- Low confidence:
  - "May be a {label} {direction}, focus to verify."

Depth-enabled mode:

- High confidence:
  - "{Label} look down, {direction}, {distance:.1f} feet away."
  - "{Label} look up, {direction}, {distance:.1f} feet away."
  - "{Label} {direction}, {distance:.1f} feet away."
- Low confidence:
  - "May be a {label} {direction}, focus to verify."

### 10) Hint catalog (all hints)

Hints are appended to normal narration by server milestones.

- Navigate hint (`scan_count >= 1`):
  - "Swipe right to browse each detected item one at a time."
- Feedback hint (`scan_count >= 2`):
  - "If a detection was wrong, just say wrong."
- Ask hint (`teach_count >= 2`):
  - "You can ask questions about your items, like what is in a receipt."
- Export hint (`teach_count >= 3`):
  - "Try saying export this when looking at a document."
- Double-tap hint (`scan_count >= 5`):
  - "Double tap to save a location while browsing items."


## Narration policy

- Backend-generated narration is final; frontend does not rewrite.
- Use short, direct phrasing.
- Confidence:
  - low confidence may use hedged wording ("may be...")
  - medium/high should be direct
- Direction and distance language should remain consistent and spatially useful.
- Errors should be neutral/system-focused.


## UX guardrails

- Do not expose internal mode names to users.
- Destructive actions require explicit confirmation.
- Failed teach attempts must not create placeholder memory entries.
- Frontend should always trust `session_state` as canonical.
- Avoid duplicated narration from client and server.


## Change-management checklist (for agents)

When UX.md changes, update and keep aligned:

1. WS state/session logic (`voice_session.py`, `voice_ws.py`)
2. HTTP voice parity (`voice.py`)
3. Context contract (`voice_state_context.py`)
4. Frontend protocol doc (`FRONTEND_GUIDE.md`)
5. Narration examples and UX policy text in docs
6. Relevant tests:
   - `test_voice_ws.py`
   - `test_voice_router.py`
   - `test_websocket_e2e.py`
   - `test_frontend_simulation.py` (if present)
