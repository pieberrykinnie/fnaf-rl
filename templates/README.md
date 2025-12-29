# Templates Directory

Reference screenshots used for template matching in game state detection.

## Directory Structure

```
templates/
├── office/                    # Full office starting frame
│   └── starting_frame.png    # Reference shot of office at night start
│
├── ui_elements/              # Individual UI control states
│   ├── power_bar*.png        # Power indicator at various levels
│   ├── door_left_closed.png  # Left door closed state
│   ├── door_left_open.png    # Left door open state
│   ├── door_right_*.png      # Right door states
│   ├── light_left_on.png     # Left hallway light on
│   ├── light_right_on.png    # Right hallway light on
│   ├── camera_up.png         # Camera monitor up
│   └── usage_bar*.png        # Power usage indicator
│
└── animatronics/             # Character detection templates
    ├── bonnie_stage.png      # Bonnie at show stage
    ├── bonnie_west_hall.png  # Bonnie in west hall
    ├── bonnie_door.png       # Bonnie at left door
    ├── chica_stage.png       # Chica at show stage
    ├── chica_east_hall.png   # Chica in east hall
    ├── chica_door.png        # Chica at right door
    ├── freddy_stage.png      # Freddy at show stage
    ├── foxy_stage*.png       # Foxy in pirate cove (stages 1-4)
    └── foxy_running.png      # Foxy running down west hall
```

## How to Collect Templates

### Quick Start

1. **Run the interactive capture tool:**

   ```bash
   python frame_capture.py
   ```

2. **Start a FNAF game** and position to key moments:
   - For office template: Start of night (left side view)
   - For UI elements: Position camera to see the control element clearly
   - For animatronics: Switch cameras when character is visible

3. **Press keys to capture:**
   - `S` - Save as office starting frame
   - `1` - Save as UI element (will prompt for name)
   - `2` - Save as animatronic (will prompt for character/location)
   - `Q` - Quit and show inventory

### Manual Collection

If you prefer manual file management:

1. Take a screenshot and save to appropriate subdirectory
2. Name it descriptively (lowercase, underscores for spaces)
3. Files will be auto-versioned if duplicates exist

## Template Guidelines

### Office Starting Frame (`office/starting_frame.png`)

**Mandatory** - Used for detecting active gameplay

Requirements:

- Full game window capture at night start
- Consistent lighting/color grading
- Left side view (before any panning)
- Should be 800x600 (or your game resolution)
- Filename must be exactly `starting_frame.png`

Capture at:

- Right after night starts
- Before any doors/lights are toggled
- Before any camera panning

### UI Elements

Optional for enhanced detection. Used for:

- Power bar level detection
- Door/light state verification
- Camera state confirmation

Tips:

- Crop close to the element
- Include context (surrounding pixels for distinctiveness)
- Multiple versions okay (different resolutions, themes)
- Name: `{element}_{state}.png` or `{element}*.png` for variants

Examples:

- `power_bar_100.png`, `power_bar_50.png`
- `door_left_closed.png`, `door_left_open.png`
- `light_left_on.png`, `light_left_off.png`

### Animatronics

Optional for advanced detection. Used for:

- Character location identification
- Jumpscare detection
- Camera feed analysis

Tips:

- Full character visible in frame
- Consistent camera angle
- Include surrounding environment
- Name: `{character}_{location}.png`

Examples:

- `bonnie_stage.png` (at show stage on pirate cove cam)
- `chica_west_hall.png` (west hall corner cam view)
- `foxy_stage1.png`, `foxy_stage2.png` (pirate cove progression)

## Troubleshooting

### Template file not found

- Check filename matches exactly (case-sensitive on Linux/Mac)
- Verify file is in correct subdirectory
- Run `python template_manager.py` to list all templates

### Template matching fails

- Ensure game resolution matches template capture resolution
- Try capturing from a different starting position
- Increase match threshold if needed (currently 0.7)

### Multiple template versions

- Old versions are preserved with timestamp: `starting_frame_20251228_143022.png`
- Original filename always points to latest version
- You can delete timestamped versions to clean up

## Version Control

⚠️ **Note:** `.png` files in `templates/` are NOT tracked by git by default (see `.gitignore`).

To track specific important templates:

```bash
# Force-add a specific template
git add -f templates/office/starting_frame.png
```

Or update `.gitignore` if you want to commit templates.
