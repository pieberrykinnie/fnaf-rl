# Project Organization Summary

## Current State ✅

**Structure**:

```
fnaf-rl/
├── src/                    # Core modules
│   ├── observer.py         # Frame capture (24 FPS)
│   └── game_state.py       # State extraction (night detection only)
│
├── tools/                  # Development utilities
│   ├── template_manager.py # Template management
│   ├── frame_capture.py    # Interactive template capture
│   ├── test_game_state.py  # Tests
│   └── cleanup.py          # Cleanup helper
│
├── templates/              # Reference screenshots
│   ├── office/             # Office scenes
│   ├── ui_elements/        # UI components
│   └── animatronics/       # Character appearances
│
├── data/                   # Game data
│   └── recordings/         # Gameplay recordings (future)
│
└── docs/                   # Documentation
    ├── game-state-spec.md  # State specification
    └── dev/                # Development docs
```

## What Works ✅

- Night detection via template matching
- Observer captures at 24 FPS
- Template collection tool (`frame_capture.py`)
- Test harness (`test_game_state.py`)

## What's Next

Implement additional states as needed:

- Time tracking
- Power level
- Player actions (doors, lights)
- Camera state
- Animatronic tracking

Start with whichever is most critical for your RL agent.

## How to Use

```bash
# Test night detection
python -m tools.test_game_state

# Collect templates
python -m tools.frame_capture

# View template inventory
python -m tools.template_manager
```
