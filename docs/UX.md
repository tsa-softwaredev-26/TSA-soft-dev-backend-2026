# Spaitra UX Reference

Narration content, onboarding flow, hints, and UX decisions. This is the product spec for what the app says and when. The backend produces all narration text - the frontend speaks it verbatim.

---

## Design Philosophy

- Voice-first, low-friction. One core action per interaction.
- No tutorials. Onboarding is the first real use.
- The user learns by doing, not by reading instructions.
- Every spoken output is phrased for a blind user who cannot see the screen.
- Steve Jobs philosophy: fewer controls, more intelligence.

---

## Onboarding Flow

Triggered when the database is empty (first install). The server drives every step. The frontend follows `session_state.current_mode` and speaks `tts.narration` verbatim.

### Welcome

Server speaks on connect:

> "Welcome to Spaitra. I remember where your things are. Grab two items near you and let's start. Say teach, then the name of the first item."

State: `onboarding_teach`

### Teach item 1

User says "teach my keys" via Spaitra button. Server requests camera (`control: request_image`). User sends image. Server saves item.

Server speaks:

> "[Label] saved. Which room is this?"

State: `awaiting_location`

User names room. Server speaks:

> "Got it, [label] in the [room]. Now teach me the second item. Say its name."

State: `onboarding_teach`, phase `teach_2`

### Teach item 2

Same flow as item 1. After room is named:

> "Got it, [label] in the [room]. Great. Place both items somewhere in the room, then say scan."

State: `onboarding_await_scan`

### Scan

User says "scan". Server requests camera. User sends image. Server announces results:

> "[Item 1 narration]. [Item 2 narration]. Now swipe right to browse each item."

State: `focused_on_item`

### Swipe intro

User swipes through items. Server re-reads narration for each item. After the user reaches the last item:

> "[Item narration]. Swipe left to go back. That is how scan works. Now ask me where you left your [first item]."

### Ask demo

User asks via Spaitra button. Server answers. Onboarding completes:

> "[Answer]. You are all set. Say scan, teach, or ask anything."

State: `idle`

**No feedback is collected during onboarding.**

---

## Onboarding Completion Message

Brief readiness message with 2-3 things the user can try next. Natural language, not a list.

> "You are all set. Say scan, teach, or ask anything."

Natural language questions work too. The user can ask "where is my wallet" or "what does the label say" without knowing the exact command name.

---

## Detection Failure Messages

Spoken by server in response to a failed teach attempt. Frontend just speaks whatever the server sends.

| Condition | Narration |
|---|---|
| Image too dark | "Image is too dark. Increase lighting and try again." |
| Image too blurry | "Hold steady and try again." |
| No object detected | "Could not detect the item. Try a different angle or closer distance." |
| Second failure | "Still could not detect it. Make sure the item is in view and well-lit." |

Failed teach attempts do not create placeholders in memory.

---

## Scan Narration Format

The backend generates all scan narration. Frontend reads it verbatim in left-to-right spatial order.

Examples:

> "Wallet look down, to your left, 2.3 feet away."
> "House keys to your right, ahead."
> "May be a receipt slightly right, focus to verify."

`"May be a..."` means low confidence. Do not add extra hedging.
`"look down,"` and `"look up,"` are gaze cues. Read them as-is.

If no items found:

> "I did not find anything familiar."

If no items found during onboarding scan:

> "I did not spot either item. Make sure they are in view and try again."

---

## Ask / Find Narration

Examples:

> "Your wallet is in the kitchen, to your left. Last seen 5 minutes ago."
> "I could not find wallet in your memory."

---

## Error and Timeout Messages

| Scenario | Narration |
|---|---|
| Transcription failed | "I did not catch that. Try again." |
| Server timeout | "That took too long. Try again." |
| Network error | "I lost connection. Reconnecting." |

Errors are neutral and system-focused. Never suggest user error first.

---

## Hints

The server appends hints to normal narration at usage milestones. They are not separate events. The frontend does not need to track hint state - the server handles it.

| Hint | When |
|---|---|
| "Swipe right to browse each detected item one at a time." | After first scan |
| "If a detection was wrong, just say wrong." | After second scan |
| "You can ask questions about your items, like what is in a receipt." | After second teach |
| "Try saying export this when looking at a document." | After third teach |
| "Double tap to save a location while browsing items." | After fifth scan |

Hints are relative to the current state. A hint about scanning will not fire in the middle of a teach flow. The server delays hints if the user is clearly hesitating and the state is ambiguous.

---

## Mode Transition Phrasing

Use verb-first language. No mode names exposed to the user.

| What happens | What server says |
|---|---|
| Switching to listen for room name | "[Label] saved. Which room is this?" |
| Location confirmed, back to idle | "Got it, [label] in the [room]." |
| Returning to idle | (silent, or "Going back.") |
| Settings opened | "Opening settings." |
| Confirmation requested | "[Action] [target]? Say yes to confirm." |
| Confirmation canceled | "Canceled." |
| Confirmation done | "Done." |

---

## Privacy

Mentioned briefly in onboarding. One sentence, not aggressive.

> "Your data is encrypted and stored only on your device."

Do not repeat privacy messaging elsewhere. No privacy mode toggle. The user should trust that privacy is already handled.

---

## Confidence and Uncertainty

Low confidence matches use hedged phrasing in narration:

> "May be a wallet to your left, focus to verify."

Medium and high confidence: direct phrasing, no qualifier.

Confirmations for destructive actions (delete) repeat the target:

> "Delete [label]? Say yes to confirm, or no to cancel."

5-second timeout with one reminder, then cancel safely.

---

## Accessibility

- Speech rate: 1.0x default (moderate). User can adjust via settings.
- All key outcomes use speech plus haptic. No earcons.
- Spatial ordering: matches read left-to-right. Direction words always included.
- Interruption: tap-only. User may be speaking to someone else; do not use voice barge-in detection.
- Concise mode: deferred future feature, not implemented.
