# fnaf-rl

Vision-based RL agent for Five Nights at Freddy's. Current extractor supports:
- Night start detection
- Time elapsed tracking
- Power percent (template-based digits with smoothing)
- Power usage (1–5 bars) with confidence-aware smoothing

Quick run (live overlay test):

```bash
uv run -m tools.test_game_state
```

Template capture / management:

```bash
uv run -m tools.frame_capture
uv run -m tools.template_manager
```

UI region discovery (power/usage):

```bash
uv run -m tools.discover_ui_regions
```
