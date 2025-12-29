# Game State Detection - Implementation Status

## ✅ Completed

### 1. Architecture & Design

- **Decoupled design**: `src/observer.py` → `src/game_state.py` → application layer
- **Template-based detection**: Pre-recorded screenshot references for reliable matching
- **Organized directory structure**:
  - `src/` - Core modules (observer, game_state)
  - `tools/` - Utilities (frame_capture, template_manager, tests, discover_ui_regions)
  - `templates/` - Reference screenshots (office, ui_elements)
  - `docs/` - Documentation

### 2. Core Components

- **GameState dataclass** - Matches specification (state schema in `docs/game-state-spec.md`)
- **GameStateExtractor** - Main interface for extraction
- **TemplateManager / FrameCapture** - Utilities for collecting and organizing templates
- **UI discovery tool** - `tools/discover_ui_regions.py` outputs ROI constants for stable detection
- **Test harness** - `tools/test_game_state.py` for live validation overlay

### 3. Implemented Detectors

#### Night Detection ✅ **WORKING**

- **Method**: Template matching against pre-recorded office starting frame
- **Behavior**: One-way state transition (False→True on first match)
- **Confidence threshold**: 0.7 (normalized cross-correlation)
- **Performance**: ~5ms per frame

#### Time Tracking ✅ **WORKING**

- **Method**: Wall-clock `perf_counter` from first night start; resets only on manual reset
- **Output**: `timeElapsed` in seconds

#### Usage Bar (power usage 1–5) ✅ **WORKING**

- **ROI**: Fixed region `USAGE_BAR_REGION` (120,657)-(223,689) from discovery tool
- **Method**: Masked template matching (TM_CCOEFF_NORMED) over alpha-masked sprites 1–5
- **Disambiguation**: Area-weighted tie-break + per-level minimum scores + tiny margin (0.001) to separate overlapping cumulative sprites
- **Status**: Correctly distinguishes all five levels in synthetic tests; validated in-game via `tools.test_game_state`

---

## 🔄 Next Steps

- Power percentage detection
- Player actions (doors, lights, camera toggle, current camera)
- Animatronic tracking
- Special events (jumpscare, blackout, Golden Freddy)
- Add smoothing/debouncing where needed once more signals are online

---

## Directory Layout

```
fnaf-rl/
├── src/
│   ├── __init__.py
│   ├── observer.py            # Frame capture (24 FPS)
│   └── game_state.py          # State extractor with night detection
│
├── tools/
│   ├── __init__.py
│   ├── template_manager.py    # Template management utility
│   ├── frame_capture.py       # Interactive capture tool
│   └── test_game_state.py     # Validation script
│
├── templates/
│   ├── README.md              # Collection guidelines
│   ├── office/
│   │   └── starting_frame.png ✅
│   ├── ui_elements/           # usage_1..5.png ✅ collected
│   └── animatronics/          # (empty - for future)
│
├── data/
│   └── recordings/            # For storing gameplay recordings
│
├── docs/
│   ├── game-state-spec.md     # Specification document
│   └── dev/
│       └── IMPLEMENTATION_STATUS.md  # This file
│
├── main.py                    # Entry point (TBD)
├── README.md                  # Project overview
├── LICENSE
└── pyproject.toml             # Project config
```

---

## Usage

### Live test (night + usage)

```bash
uv run -m tools.test_game_state
```

### Capture templates

```bash
uv run -m tools.frame_capture
```

---

## Performance Target

At 24 FPS (41.67ms per frame):

- Frame capture: ~5ms
- Night detection: ~5ms (template match)
- Other states: ~20ms budget remaining
- **Status**: Well under budget

---

## Next Session Checklist

- [ ] Verify imports work from new structure
- [ ] Test night detection still works
- [ ] Plan next state to implement
- [ ] Clean up root directory (remove old files)
