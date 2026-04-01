# Spaitra UX Principles and Flow

This document defines the product UX at a behavioral level:
- What the experience should feel like
- How users move through core flows
- How narration should guide without overwhelming

It is intentionally high level and avoids implementation detail.

## UX Philosophy

- Voice-first: speaking is the primary interaction.
- Low friction: one clear action at a time.
- Real onboarding: users learn by teaching and scanning real items immediately.
- Calm guidance: short narration, clear next step, no extra words.
- Accessibility-first: language should be direct, concrete, and spatially useful.

## Core User Goals

- Teach: save an item so Spaitra can remember it later.
- Scan: find remembered items in the current scene.
- Ask: query memory for where an item was last seen.

## Interaction Model

- User presses and holds the dedicated chat button to speak, then releases to send.
- System listens, interprets intent, and replies with narration plus an updated interaction state.
- Narration is authoritative and should be spoken exactly as provided.
- The interface should always reflect the current state from the backend.

## Onboarding Flow

On first use (empty memory), onboarding is a real workflow, not a tutorial wall:

1. Welcome and ask for the first item.
2. Teach item one and capture its room.
3. Teach item two and capture its room.
4. Prompt user to scan both items.
5. If scan succeeds, guide item-by-item browsing.
6. Prompt one ask/find query.
7. Confirm completion and transition to normal use.

Onboarding should focus on confidence and momentum, not feature depth.

## Runtime Flow (High Level)

- Idle: user can teach, scan, ask, open settings, or go back.
- Awaiting image: system is waiting for camera input tied to teach/scan.
- Awaiting location: system expects a room name after a successful teach.
- Awaiting confirmation: system expects explicit yes/no for destructive actions.
- Focused item: user can browse matches, ask about current item, or flag wrong detection.
- Onboarding modes: restrict commands to keep first-run flow linear and predictable.

## Narration Guidelines

- Keep prompts short and actionable.
- Use explicit spatial wording for scan/find outputs.
- Use neutral system language for failures.
- Avoid exposing internal system terms or mode names.
- Prefer one instruction per line of narration.

## Key Onboarding Prompt Intent

- Teaching prompt should make hold-to-talk explicit and ask for "teach + item name".
- Await-scan prompt should support both quick bottom-area scan action and spoken scan command.

## Guardrails

- Do not create memory entries on failed teach attempts.
- Destructive actions require explicit confirmation.
- Do not collect corrective feedback during onboarding.
- Avoid duplicate narration from multiple layers.

## Change Policy

When UX behavior changes, update product-facing guidance and spoken prompts together so the live behavior remains aligned with this document.
