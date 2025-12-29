# Cursor-Driven Interaction Model

This project uses a cursor-driven interaction model for doors, lights, and camera flips (no CV detection for these controls). The logic lives in `src/game_state.py` within `GameStateExtractor`.

## Overview

- Fixed `UIRegion` bounds define interactive areas for left/right doors, left/right lights, and camera flip.
- Camera flips when the cursor enters the camera region from above, gated by `cam_anim_time` cooldown between flips.
- Doors toggle on click rising edge inside their regions and respect `door_anim_time` cooldown to mimic animation duration.
- Lights toggle on click rising edge and are mutually exclusive; when camera is up, both lights are forced off.
- Duplicate clicks at the exact same pixel within ~300ms are suppressed. A general `click_cooldown` (0.12s) also applies.
- Usage is computed deterministically when interaction mode is enabled: `usage = 1 + doors + lights + camera`.
- Power and usage recognition use smoothing and coherence to block impossible jumps or brief glitches.

## Key Constants

- `CAMERA_FLIP_REGION`: Resolution-specific bounds for camera flip.
- `LEFT_DOOR_REGION` / `RIGHT_DOOR_REGION`: Door click regions.
- `LEFT_LIGHT_REGION` / `RIGHT_LIGHT_REGION`: Light click regions.
- `cam_anim_time`: Minimum time between camera flips.
- `door_anim_time`: Minimum time between door toggles to avoid animation spam.
- `click_cooldown`: Rising-edge debounce for user clicks.

Update regions using `tools/discover_ui_regions.py` for accuracy; avoid manual edits.

## Invariants

- Lights off while camera is up.
- Lights mutually exclusive.
- Usage invariant: `usage - 1 = doors + lights + camera`.

## Notes

- The interaction model is resolution-specific and should be re-discovered if game/window resolution changes.
- The CV-based detection remains for usage and power bars; controls are cursor-driven by design.
