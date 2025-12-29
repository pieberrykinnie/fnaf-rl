# Game State Specification

This document defines the complete game state representation for the FNAF RL agent. The state is extracted from screen captures and encoded as JSON for training and inference.

## Overview

The game state captures all relevant information about the current gameplay situation, including:

- Session state (menu vs active gameplay)
- Time and resource management (power, time elapsed)
- Player actions (doors, lights, camera)
- Animatronic positions and last known locations
- Terminal states (jumpscare, blackout)

## State Schema

### Session State

#### `nightStarted`: boolean

- **Description**: Whether a night is actively being played
- **True**: Night is in progress (gameplay active)
- **False**: Main menu, loading screen, custom night setup, post-jumpscare, or 6 AM win screen
- **Purpose**: Prevents the model from taking actions outside of actual gameplay

---

### Time & Resources

#### `timeElapsed`: float

- **Description**: Seconds elapsed since the night started
- **Range**: 0.0 to ~480.0 (8 minutes for a full night)
- **Resets**: On new night start

#### `power`: integer

- **Description**: Remaining power percentage as displayed on HUD
- **Range**: 0-100
- **Note**: Extracted from the power display in bottom-left corner

#### `usage`: integer

- **Description**: Current power usage rate
- **Range**: 1-5 (bars shown in UI)
- **Factors**: Base (1) + left door (1) + right door (1) + left light (1) + right light (1) + camera (1)

---

### Player Actions

#### `leftLight`: boolean

- **Description**: Whether the left hallway light is currently on

#### `rightLight`: boolean

- **Description**: Whether the right hallway light is currently on

#### `leftDoor`: boolean

- **Description**: Whether the left door is in closed position
- **Note**: Reflects button state, independent of animation progress

#### `rightDoor`: boolean

- **Description**: Whether the right door is in closed position
- **Note**: Reflects button state, independent of animation progress

#### `isOnCamera`: boolean

- **Description**: Whether the camera monitor is up
- **Note**: Reflects intended state, independent of flip animation progress

#### `currentCamera`: integer

- **Description**: Currently selected camera feed
- **Range**: 0-10
- **Note**: Valid regardless of whether camera is currently up or down

---

### Camera Mapping

The following mapping converts game camera names to integer indices:

| Index | Camera ID | Location Name |
|-------|-----------|---------------|
| 0 | CAM 1A | Show Stage |
| 1 | CAM 1B | Dining Area |
| 2 | CAM 1C | Pirate Cove |
| 3 | CAM 2A | West Hall |
| 4 | CAM 2B | West Hall Corner |
| 5 | CAM 3 | Supply Closet |
| 6 | CAM 4A | East Hall |
| 7 | CAM 4B | East Hall Corner |
| 8 | CAM 5 | Backstage |
| 9 | CAM 6 | Kitchen (audio only) |
| 10 | CAM 7 | Restrooms |

---

### Animatronic Tracking

Each animatronic has a `lastSeen[Name]` state that encodes their last known location, plus a corresponding timestamp.

#### `lastSeenBonnie`: integer

- **Description**: Last observed location of Bonnie
- **Encoding**:
  - `0`: Show Stage (CAM 1A) - starting position
  - `1`: Dining Area (CAM 1B)
  - `2`: Backstage (CAM 5)
  - `3`: Supply Closet (CAM 3)
  - `4`: West Hall (CAM 2A)
  - `5`: West Hall Corner (CAM 2B)
  - `6`: Left door blind spot (visible with left light)
  - `7`: Left door jammed (unable to close door)
  - `-1`: Unknown/not recently seen
- **Note**: May encode additional states for realistic player uncertainty (e.g., "just left the door")

#### `lastSeenBonnieTime`: float

- **Description**: Time elapsed (in seconds) since Bonnie was last seen at `lastSeenBonnie` location
- **Range**: 0.0+

#### `lastSeenChica`: integer

- **Description**: Last observed location of Chica
- **Encoding**:
  - `0`: Show Stage (CAM 1A) - starting position
  - `1`: Dining Area (CAM 1B)
  - `2`: Restrooms (CAM 7)
  - `3`: Kitchen (CAM 6) - audio cue only
  - `4`: East Hall (CAM 4A)
  - `5`: East Hall Corner (CAM 4B)
  - `6`: Right door blind spot (visible with right light)
  - `7`: Right door jammed (unable to close door)
  - `-1`: Unknown/not recently seen

#### `lastSeenChicaTime`: float

- **Description**: Time elapsed (in seconds) since Chica was last seen at `lastSeenChica` location
- **Range**: 0.0+

#### `lastSeenFreddy`: integer

- **Description**: Last observed location of Freddy
- **Encoding**:
  - `0`: Show Stage (CAM 1A) - starting position
  - `1`: Dining Area (CAM 1B)
  - `2`: Restrooms (CAM 7)
  - `3`: Kitchen (CAM 6)
  - `4`: East Hall (CAM 4A)
  - `5`: East Hall Corner (CAM 4B)
  - `-1`: Unknown/not recently seen
- **Note**: Freddy never appears at door blind spots, but may encode uncertainty states for realism

#### `lastSeenFreddyTime`: float

- **Description**: Time elapsed (in seconds) since Freddy was last seen at `lastSeenFreddy` location
- **Range**: 0.0+

#### `lastSeenFoxy`: integer

- **Description**: Last observed state of Foxy
- **Encoding**:
  - `0`: Stage 1 - Pirate Cove curtain closed (CAM 1C)
  - `1`: Stage 2 - Pirate Cove curtain slightly open
  - `2`: Stage 3 - Pirate Cove curtain wide open, Foxy peeking out
  - `3`: Stage 4 - Pirate Cove empty (Foxy has left)
  - `4`: Running down West Hall (CAM 2A)
  - `5`: Banging on left door
- **Note**: No "unknown" state needed - stage 0 is the default assumption

#### `lastSeenFoxyTime`: float

- **Description**: Time elapsed (in seconds) since Foxy was last seen at `lastSeenFoxy` state
- **Range**: 0.0+

---

### Special Events

#### `goldenFreddy`: boolean

- **Description**: Whether Golden Freddy is currently visible in the office
- **Note**: Rare easter egg trigger; may or may not be used in final model

#### `jumpscaredBy`: integer

- **Description**: Which animatronic delivered the jumpscare (if any)
- **Encoding**:
  - `0`: No jumpscare
  - `1`: Freddy (office)
  - `2`: Freddy (power out)
  - `3`: Bonnie
  - `4`: Chica
  - `5`: Foxy
  - `6`: Golden Freddy

#### `blackout`: boolean

- **Description**: Whether the power has run out
- **Purpose**: Prevents illegal actions during blackout (doors, lights, cameras non-functional)
- **Note**: Once true, only valid action is waiting for Freddy

---

## Additional Considerations

### Potential Future States

The following states may be considered for future iterations:

1. **Audio cues**: Footstep sounds, laughter, kitchen noises
2. **Fan state**: Whether the office fan is running (always on, but could be encoded)
3. **Phone guy audio**: Whether the phone call is playing (Night 1-4)
4. **Difficulty**: Which night or custom night settings are active
5. **Action cooldowns**: Time since last camera flip, door toggle, etc.
6. **Frame deltas**: Movement/change detection between consecutive frames

### State Reduction Strategies

For training efficiency or to simulate human-like limitations:

- Remove timestamp tracking (forces memory-based reasoning)
- Limit camera history (only track current/previous position)
- Add uncertainty to animatronic positions (probabilistic tracking)
- Remove Golden Freddy tracking entirely
- Merge some animatronic states (e.g., "at door" vs "jammed door")

### Implementation Notes

- All boolean states should be explicitly `true` or `false` (not null/undefined)
- Integer states should use `-1` for truly unknown states (vs `0` for known starting states)
- Float timestamps should be monotonically increasing within a night
- State extraction should be robust to partial occlusion and UI variations
- Consider frame-to-frame validation to prevent spurious state changes

---

## Example JSON State

```json
{
  "nightStarted": true,
  "timeElapsed": 145.6,
  "power": 54,
  "usage": 3,
  "leftLight": false,
  "rightLight": false,
  "leftDoor": true,
  "rightDoor": false,
  "isOnCamera": true,
  "currentCamera": 2,
  "lastSeenBonnie": 4,
  "lastSeenBonnieTime": 12.3,
  "lastSeenChica": 1,
  "lastSeenChicaTime": 45.7,
  "lastSeenFreddy": 0,
  "lastSeenFreddyTime": 145.6,
  "lastSeenFoxy": 2,
  "lastSeenFoxyTime": 8.1,
  "goldenFreddy": false,
  "jumpscaredBy": 0,
  "blackout": false
}
```

---

## Version History

- **v1.0** (2025-12-28): Initial specification
