# Project Organization Summary

## Current State ✅

**Structure**:

```
fnaf-rl/
├── src/                    # Core modules
│   ├── observer.py         # Frame capture (24 FPS)
│   └── game_state.py       # State extraction (night, time, power %, usage)
│
├── tools/                  # Development utilities
│   ├── template_manager.py # Template management
│   ├── frame_capture.py    # Interactive template capture
│   ├── test_game_state.py  # Live overlay validation
│   └── discover_ui_regions.py # Auto-detect UI ROIs (power/usage)
│
├── templates/              # Reference screenshots
│   ├── office/             # Office scenes
│   ├── ui_elements/        # UI components (power digits, percent, usage bars)
│   └── animatronics/       # Character appearances (future)
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
- Time elapsed tracking
- Power percentage via digit templates (with smoothing and coherence)
- Power usage (1–5) with confidence-aware smoothing and tie-breaks
- Observer captures at 24 FPS
- Template collection and ROI discovery tools
- Live test harness (`tools.test_game_state`)

## How to Use

```bash
# Live overlay test
uv run -m tools.test_game_state

# Collect templates
uv run -m tools.frame_capture

# View/manage templates
uv run -m tools.template_manager

# Discover UI regions (power/usage)
uv run -m tools.discover_ui_regions
```
