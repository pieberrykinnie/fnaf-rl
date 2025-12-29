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
- Template match on office starting frame; one-way False→True

#### Time Tracking ✅ **WORKING**
- `perf_counter` from night start; resets on manual reset

#### Power Percentage ✅ **WORKING**
- Digit/percent template matching over fixed ROI (183,623, 52x24)
- Smoothing: median over 5 readings; coherence blocks impossible increases and large drops on low-confidence reads (missing digits/percent)

#### Usage Bar (1–5) ✅ **WORKING**
- Fixed ROI from discovery tool; masked template matching with area tie-breaks
- Confidence-aware smoothing: median over recent nonzero reads (size 11); coherence blocks low-confidence jumps >1 unless confidence is high

---

## 🔄 Next Steps

- Player actions (doors, lights, camera toggle, current camera)
- Animatronic tracking
- Special events (jumpscare, blackout, Golden Freddy)

---

## Directory Layout

```
fnaf-rl/
├── src/
│   ├── __init__.py
│   ├── observer.py            # Frame capture (24 FPS)
│   └── game_state.py          # State extractor (night, time, power %, usage)
│
├── tools/
│   ├── __init__.py
│   ├── template_manager.py    # Template management utility
│   ├── frame_capture.py       # Interactive capture tool
│   ├── discover_ui_regions.py # Auto-detect UI ROIs
│   └── test_game_state.py     # Validation script
│
├── templates/
│   ├── README.md              # Collection guidelines
│   ├── office/
│   │   └── starting_frame.png ✅
│   ├── ui_elements/           # power digits/percent, usage_1..5.png ✅ collected
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
