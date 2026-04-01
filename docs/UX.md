# Spaitra UX Principles and Interaction Trees

This document defines the product UX at a behavioral level:
- What the experience should feel like
- How users move through each state
- Which actions are valid in each state
- What narration should do at each branch

Intentionally behavior-focused; implementation details are in code.

## UX Philosophy

- Voice-first: speaking is the primary interaction.
- Low friction: one clear action at a time.
- Real onboarding: users learn by teaching and scanning real items immediately.
- Calm guidance: short narration, clear next step, no extra words.
- Accessibility-first: language should be direct, concrete, and spatially useful.
- Backend-authoritative: frontend renders backend state and narration exactly.

## Core User Goals

- Teach: save an item so Spaitra can remember it later.
- Scan: find remembered items in the current scene.
- Ask: query memory in natural language.
- Focused follow-up: while browsing detections, ask about the selected item.

## Core Interaction Contract

- User holds the top-right chat button to record, releases to send.
- Frontend emits `chat_start` on press, starts recording after `listening`.
- Frontend emits `audio` on release, with optional `image` when camera is active.
- Backend transcribes speech, dispatches intent based on current mode, and returns:
  - `tts` narration
  - `session_state` (authoritative current mode + context)
  - optional `action_result` for UI data updates
- Frontend may interrupt TTS when user presses chat button again.

## Authoritative Runtime States

- `idle`
- `onboarding_teach`
- `onboarding_await_scan`
- `awaiting_image`
- `awaiting_location`
- `awaiting_confirmation`
- `focused_on_item`

These state names are backend contracts and should not be remapped client-side.

## Listening Prompt Policy

Listening prompts must match state and stay concise:

- `idle`: "Scan, teach something new, or ask me anything."
- `onboarding_teach`: onboarding teach prompt with hold-to-talk guidance.
- `onboarding_await_scan`: onboarding scan prompt, supports bottom scan or voice "scan".
- `awaiting_image`: "Hold your phone up."
- `awaiting_location`: "Which room is this?"
- `awaiting_confirmation`: yes/no confirmation prompt.
- `focused_on_item`: focused label + item-query hint + "say wrong" hint.

## Whisper Context Biasing Policy (Behavior)

Whisper context biasing is always enabled by default and should be treated as mode-aware behavior:

- Base context includes known labels, recent rooms, and common navigation/find terms.
- Additional state hints come from current mode and live session context.
- In focused item browsing, context should prioritize item-scoped verbs and current labels.
- In home/idle usage, context should prioritize broad commands (scan/teach/find/ask/settings).
- In location capture, context should prioritize room-like phrases.

UX requirement: context bias should differ by situation so command likelihood matches user intent by mode.

## Full Action Tree

### 1) Connect

- If memory is empty:
  - Enter `onboarding_teach` with onboarding welcome narration.
- Else:
  - Enter `idle` with short readiness narration.

### 2) Global Input Cycle

- User presses chat:
  - Server emits `listening` prompt for current state.
- User releases:
  - Frontend sends `audio` (+ optional image).
  - Server transcribes and emits optional `transcription`.
  - Server dispatches by state + intent and emits narration + state update.

### 3) State: `idle`

Allowed user intents:

- Scan
  - If no image: transition to `awaiting_image`, request image, narrate hold-phone prompt.
  - If image present: run scan.
    - No matches -> narrate miss, remain/return `idle`.
    - Matches -> cache scan context, move to `focused_on_item`, narrate first result set.

- Teach/Remember
  - Parse/resolve label from utterance.
  - If no image: transition to `awaiting_image`, request image, narrate camera prompt with label.
  - If image present: run teach pipeline.
    - Failure -> narrate quality/detection issue, remain in current flow.
    - Success -> transition to `awaiting_location`, ask room.

- Find
  - Resolve query term (including fallback matching).
  - Narrate found/miss, end in `idle`.

- Ask (general)
  - Run ask pipeline.
  - Narrate answer/miss, end in `idle`.

- Open settings
  - Emit settings action result, narrate opening settings, remain `idle`.

- Navigate back/home
  - Clear focus context, narrate going back, remain `idle`.

### 4) State: `awaiting_image`

Expected input: audio plus image for the pending action (`scan` or `remember`).

- If image missing:
  - Re-request image and remind user server is still waiting.
- If pending action is scan:
  - Execute scan branch (same outcomes as idle->scan).
- If pending action is remember:
  - Execute teach branch (same outcomes as idle->teach).
- If pending action is invalid:
  - Return error and reset toward `idle`.

### 5) State: `awaiting_location`

Expected input: room name.

- If empty/unusable:
  - Ask again: "What room is this?"
- If valid:
  - Save sighting for current label.
  - Normal flow: narrate location saved, return `idle`.
  - Onboarding flow:
    - after first teach -> `onboarding_teach` (teach second item)
    - after second teach -> `onboarding_await_scan` (prompt scan both items)

### 6) State: `awaiting_confirmation`

Expected input: yes/no language.

- Confirm:
  - Execute pending destructive action (for example delete item), narrate result, return `idle`.
- Deny or unclear:
  - Narrate canceled, return `idle`.

### 7) State: `focused_on_item` (scan browsing mode)

Context available: `scan_id`, `scan_matches`, `item_index`, `current_label`.

#### Gesture branch

- Swipe left/right (`navigate` event):
  - Move item focus.
  - Emit `action_result: item_focus`.
  - Narrate newly focused item.

#### Voice branch while focused

- "wrong" style feedback:
  - Record negative feedback for focused item embeddings when cache is valid.
  - Narrate incorrect-noted acknowledgment.

- Item-scoped ask intents:
  - Read text (`read_ocr`)
  - Export text (`export_ocr`)
  - Describe focused item (`describe`)
  - Focused item question (`question`/`ocr_question`)
  - Rename focused item (`rename`)
  - Find focused item location (`find`)
  - For each: emit `item_ask` action result + narration, remain `focused_on_item`.

- Non-item command while focused:
  - Falls through to normal command handling (scan/remember/find/ask/settings/back).
  - If command changes flow, state transitions accordingly.

Important focused-mode UX behavior:

- Holding chat while browsing detections is item-scoped first, global second.
- Focused-mode wording should bias toward operations meaningful on the current detection:
  - read text, export, describe this, wrong feedback, location follow-up.

### 8) State: `onboarding_teach`

Goal: teach two items with rooms.

- Accept teach commands and guide room capture.
- Non-essential commands are discouraged; narration steers back to onboarding objective.

### 9) State: `onboarding_await_scan`

Goal: scan both taught items and practice browsing.

- Accept scan command (or scan gesture per frontend).
- Allow navigation-back and settings.
- Other commands are rejected with onboarding scan prompt.
- On successful scan:
  - transition to `focused_on_item`
  - prompt swipe browsing
  - then prompt one ask/find demo query
- After demo query completes:
  - onboarding closes and returns to `idle`.

## Home vs Focused Context UX Rules

- Home (`idle`) hold-to-talk:
  - User intent is broad and scene-level.
  - Typical asks: scan, teach, find, "describe what I am looking at", "read the text on this receipt".
  - If receipt-level ask runs, response can lead into remember/export follow-up prompts.

- Focused (`focused_on_item`) hold-to-talk:
  - User intent is about the selected detection.
  - Typical asks: "describe this wallet", "what does this say", "export this", "wrong".
  - Export and item-scoped follow-ups should be strongly expected here.

## VLM and Grounding Behavior (Current UX Contract)

- Current implemented item description and item question flows use the stored item image for the focused label.
- Current UX does not claim grounding-crop-by-spoken-noun in home mode as implemented behavior.
- If future implementation adds:
  - "describe what I am looking at" -> full-scene VLM
  - "describe this wallet" -> grounding/crop then VLM
  then this document should be updated to move that behavior into the active contract.

## Narration Guidelines

- Keep prompts short and actionable.
- Use explicit spatial wording for scan/find outputs.
- Use neutral system language for failures.
- Avoid exposing internal system terms.
- Prefer one actionable instruction per line.
- Avoid duplicate narration for the same action across layers.

## Guardrails

- Do not create memory entries on failed teach attempts.
- Destructive actions require explicit confirmation.
- Onboarding keeps flow linear and confidence-building.
- Focused-item commands should prioritize selected-item context before global fallback.

## Change Policy

When UX behavior changes, update product-facing guidance and spoken prompts together so live behavior stays aligned with this document.
