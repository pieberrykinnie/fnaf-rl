"""
Game State Extractor

Extracts structured game state from FNAF screen captures.
Processes frame-by-frame to detect game state, animatronic positions, and player actions.
"""

from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path
import time
import cv2
import numpy as np
from enum import IntEnum


@dataclass
class UIRegion:
    """Represents a fixed screen region for UI element detection."""
    x: int
    y: int
    width: int
    height: int
    
    @property
    def bounds(self) -> tuple:
        """Return (x, y, x+width, y+height) for easy cropping."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)


class AnimatronicLocation(IntEnum):
    """Enum for animatronic locations to keep encoding consistent."""
    UNKNOWN = -1


@dataclass
class GameState:
    """Complete game state representation matching game-state-spec.md"""
    
    # Session State
    nightStarted: bool
    
    # Time & Resources
    timeElapsed: float
    power: int
    usage: int
    
    # Player Actions
    leftLight: bool
    rightLight: bool
    leftDoor: bool
    rightDoor: bool
    isOnCamera: bool
    currentCamera: int
    
    # Animatronic Tracking
    lastSeenBonnie: int
    lastSeenBonnieTime: float
    lastSeenChica: int
    lastSeenChicaTime: float
    lastSeenFreddy: int
    lastSeenFreddyTime: float
    lastSeenFoxy: int
    lastSeenFoxyTime: float
    
    # Special Events
    goldenFreddy: bool
    jumpscaredBy: int
    blackout: bool
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class GameStateExtractor:
    """
    Extracts game state from FNAF screen captures.

    Cursor-driven interaction model (no CV for doors/lights/camera):
    - Doors, lights, and camera flip are toggled via fixed `UIRegion` bounds.
    - Camera flip triggers only when entering `CAMERA_FLIP_REGION` from above
        and respects `cam_anim_time` cooldown between flips.
    - Door toggles respect `door_anim_time` cooldown to avoid animation spam.
    - Click handling uses rising-edge detection with `click_cooldown`, plus
        same-pixel double-click suppression (~300ms) to prevent accidental toggles.
    - Lights are mutually exclusive (left vs right); both forced off while camera is up.
    - Usage is computed deterministically: `1 + doors + lights + camera` when interaction is enabled.
    - Power and usage include smoothing and coherence to block impossible jumps.

    Regions are resolution-specific and should be discovered/updated via
    `tools/discover_ui_regions.py` rather than edited manually.
    """
    
    # Usage bar constants (discovered with tools/discover_ui_regions.py)
    USAGE_BAR_POS = (120, 657)  # (x, y) top-left
    USAGE_BAR_SIZE = (103, 32)  # (width, height)
    USAGE_MATCH_THRESHOLD = 0.6
    # Per-level minimum scores (cumulative sprites get lower raw scores); values can be tuned
    USAGE_LEVEL_MIN_SCORES = {
        1: 0.55,
        2: 0.50,
        3: 0.50,
        4: 0.50,
        5: 0.50,
    }
    # Slightly reward fuller bars to break template ties (area is normalized to [0,1])
    USAGE_AREA_WEIGHT = 0.01
    USAGE_MARGIN = 0.001  # required gap between best and second-best (kept tiny due to overlapping templates)

    # Known UI element regions (x, y, width, height from top-left)
    # These coordinates are discovered using tools/discover_ui_regions.py and fixed for this FNAF resolution
    # Do NOT adjust these manually - use the discovery tool instead for accuracy
    USAGE_BAR_REGION = UIRegion(
        x=USAGE_BAR_POS[0], y=USAGE_BAR_POS[1],
        width=USAGE_BAR_SIZE[0], height=USAGE_BAR_SIZE[1]
    )
    
    # Power percentage constants (discovered with tools/discover_ui_regions.py)
    # Format: [digit][digit][%] - e.g., "99%", "45%", "7%" (note: variable-width region)
    POWER_DIGIT_WIDTH = 19  # width of each digit sprite
    POWER_DIGIT_HEIGHT = 23  # height of each digit sprite
    POWER_PERCENT_WIDTH = 11  # width of percent sign sprite
    POWER_PERCENT_HEIGHT = 14  # height of percent sign sprite
    POWER_MATCH_THRESHOLD = 0.50  # Lowered from 0.7 - actual scores are ~0.58 for good matches

    POWER_PERCENTAGE_POS = (183, 623)  # (x, y)
    POWER_PERCENTAGE_SIZE = (52, 24)  # (width, height)
    POWER_PERCENTAGE_MATCH_THRESHOLD = 0.6  # tune as needed
    POWER_PERCENTAGE_REGION = UIRegion(
        x=183, y=623,
        width=52, height=24
    )

    # Interactive regions (doors/lights/camera flip) - fixed for this resolution
    LEFT_DOOR_REGION = UIRegion(
        x=22, y=273,
        width=62, height=92
    )

    LEFT_LIGHT_REGION = UIRegion(
        x=24, y=391,
        width=52, height=150
    )

    RIGHT_DOOR_REGION = UIRegion(
        x=1191, y=291,
        width=60, height=84
    )

    RIGHT_LIGHT_REGION = UIRegion(
        x=1197, y=397,
        width=56, height=116
    )

    CAMERA_FLIP_REGION = UIRegion(
        x=253, y=652,
        width=602, height=44
    )

    # Legacy CV thresholds (removed for cursor-driven mode)
    
    def __init__(self, template_dir: str | Path = "templates"):
        """
        Initialize the game state extractor.
        
        Args:
            template_dir: Path to directory containing pre-recorded template images
        """
        self.template_dir = Path(template_dir)
        
        # Load reference templates
        self.office_template = self._load_template("office/starting_frame.png")
        
        # Load usage bar sprites (1-5 levels, cumulative) with alpha masks
        # Store as {level: (bgr, mask, masked_bgr)} where mask may be None if no alpha
        self.usage_sprites: dict[int, tuple[np.ndarray, Optional[np.ndarray], np.ndarray]] = {}
        self.usage_sprite_area: dict[int, int] = {}
        for level in range(1, 6):
            sprite_path = self.template_dir / f"ui_elements/usage_{level}.png"
            if not sprite_path.exists():
                continue
            raw = cv2.imread(str(sprite_path), cv2.IMREAD_UNCHANGED)
            if raw is None:
                continue
            if raw.shape[2] == 4:
                bgr = raw[:, :, :3]
                alpha = raw[:, :, 3]
                mask = alpha
                masked_bgr = cv2.bitwise_and(bgr, bgr, mask=mask)
            else:
                bgr = raw
                mask = None
                masked_bgr = bgr
            self.usage_sprites[level] = (bgr, mask, masked_bgr)
            # Count lit pixels to use as a coverage-based tie breaker
            area = int(cv2.countNonZero(mask)) if mask is not None else bgr.shape[0] * bgr.shape[1]
            self.usage_sprite_area[level] = area
        self.usage_max_area: int = max(self.usage_sprite_area.values(), default=1)
        
        # Load power digit and percent sign templates (0-9 + %)
        # Store as {digit: bgr} for matching; will resize to fit detected region
        self.power_digit_templates: dict[int, np.ndarray] = {}
        for digit in range(10):
            sprite_path = self.template_dir / f"ui_elements/power_{digit}.png"
            if not sprite_path.exists():
                continue
            raw = cv2.imread(str(sprite_path), cv2.IMREAD_UNCHANGED)
            if raw is None:
                continue
            # Extract BGR, discard alpha for now (we'll use it for masking during matching)
            bgr = raw[:, :, :3]
            self.power_digit_templates[digit] = bgr
        
        # Load percent sign template
        self.power_percent_template: Optional[np.ndarray] = None
        percent_path = self.template_dir / "ui_elements/power_percent.png"
        if percent_path.exists():
            raw = cv2.imread(str(percent_path), cv2.IMREAD_UNCHANGED)
            if raw is not None:
                self.power_percent_template = raw[:, :, :3]
        
        # Placeholder for power percentage region (will be discovered with tool)
        # Format: {x, y, width, height} to encompass up to 3 digits + percent sign
        self.power_percentage_region: Optional[UIRegion] = self.POWER_PERCENTAGE_REGION
        
        # Power detection smoothing (to reduce jitter from flickering backgrounds)
        self.power_smoothing_buffer: list[int] = []
        self.power_smoothing_size: int = 5  # Use median of last 5 frames
        self.last_power: int = 99  # Track last valid power reading (game always starts at 99%)
        
        # Usage smoothing with temporal coherence
        self.last_usage: int = 0  # Track last valid usage level
        self.usage_smoothing_buffer: list[int] = []
        self.usage_smoothing_size: int = 11
        
        # Cursor-driven mode only: no visual matching for door/light/camera

        # Persistent player-action states
        self.last_leftDoor: bool = False
        self.last_rightDoor: bool = False
        self.last_leftLight: bool = False
        self.last_rightLight: bool = False

        # Simple interaction override (hack mode) using fixed regions + cursor events
        # Default to discovered interactive regions; can be overridden later
        self.left_door_region: Optional[UIRegion] = self.LEFT_DOOR_REGION
        self.left_light_region: Optional[UIRegion] = self.LEFT_LIGHT_REGION
        self.right_door_region: Optional[UIRegion] = self.RIGHT_DOOR_REGION
        self.right_light_region: Optional[UIRegion] = self.RIGHT_LIGHT_REGION
        self.camera_flip_region: Optional[UIRegion] = self.CAMERA_FLIP_REGION

        self.input_sample: Optional[tuple[int, int, bool, bool]] = None  # (x, y, in_bounds, left_down)
        self.last_left_down: bool = False
        self.last_in_cam_region: bool = False
        self.last_cursor_pos: Optional[tuple[int, int]] = None
        self.last_click_pos: Optional[tuple[int, int]] = None
        self.last_click_time: float = 0.0
        self.last_cam_toggle_time: float = 0.0
        self.camera_up: bool = False
        self.click_cooldown: float = 0.12
        self.cam_anim_time: float = 0.25
        self.door_anim_time: float = 0.5
        self.last_left_door_toggle_time: float = 0.0
        self.last_right_door_toggle_time: float = 0.0

        self.night_started: bool = False
        self.night_start_time: Optional[float] = None  # Wall-clock time when night began
    
    def extract(self, frame: np.ndarray, cursor_sample: Optional[tuple[int, int, bool, bool]] = None) -> GameState:
        """
        Extract complete game state from a frame.
        
        Args:
            frame: BGR frame from the game window
            cursor_sample: Optional latest cursor sample as `(x, y, in_bounds, left_down)`
            
        Returns:
            GameState object with all detected values
        """
        # Accept per-call cursor input to match other extractor inputs style
        if cursor_sample is not None:
            self.input_sample = cursor_sample
        # Detect night started state (only transitions from False->True, stays True until manual reset)
        if not self.night_started and self.office_template is not None:
            self.night_started = self._detect_night_started(frame)
            if self.night_started:
                # Record when night began
                self.night_start_time = time.perf_counter()
                # Clear smoothing buffer on night start
                self.power_smoothing_buffer = []
                self.usage_smoothing_buffer = []
                # Reset tracking to known starting values
                self.last_power = 99
                self.last_usage = 0
        
        # Calculate time elapsed since night started
        time_elapsed = 0.0
        if self.night_started and self.night_start_time is not None:
            time_elapsed = time.perf_counter() - self.night_start_time
        
        # Detect power and usage, storing confidence for coherence decisions
        # Detect power and usage (usage overridden in cursor-driven mode)
        detected_power = self._detect_power(frame) if self.night_started else 0
        detected_usage, usage_confidence = self._detect_usage(frame) if self.night_started else (0, 0.0)

        # Detect/Update camera state (cursor-driven)
        is_on_camera = False
        if self.night_started and self._interaction_enabled():
            self._apply_interaction(frame)
            is_on_camera = self.camera_up

        # Detect door/light buttons when visible; keep last-known when hidden
        left_door = self.last_leftDoor
        right_door = self.last_rightDoor
        left_light = False
        right_light = False

        if self.night_started and not is_on_camera:
            if self._interaction_enabled():
                left_door = self.last_leftDoor
                right_door = self.last_rightDoor
                left_light = self.last_leftLight
                right_light = self.last_rightLight
        else:
            # Lights always off when camera is up; doors persist
            left_light = False
            right_light = False

        # Smooth usage over recent frames to damp camera-flip glitches
        # Usage: cursor-driven invariant (usage - 1 = doors + lights + camera)
        if self.night_started and self._interaction_enabled():
            detected_usage = 1 + int(left_door) + int(right_door) + int(left_light) + int(right_light) + int(is_on_camera)
        elif self.night_started:
            # Smooth usage over recent frames to damp glitches when using CV
            self.usage_smoothing_buffer.append(detected_usage)
            if len(self.usage_smoothing_buffer) > self.usage_smoothing_size:
                self.usage_smoothing_buffer.pop(0)
            nonzero = [u for u in self.usage_smoothing_buffer if u > 0]
            detected_usage = int(np.median(nonzero)) if nonzero else 0
        
        # Start with minimum viable state (only nightStarted and timeElapsed filled in)
        state = GameState(
            nightStarted=self.night_started,
            timeElapsed=time_elapsed,
            power=detected_power,
            usage=detected_usage,
            leftLight=False,
            rightLight=False,
            leftDoor=False,
            rightDoor=False,
            isOnCamera=is_on_camera,
            currentCamera=0,
            lastSeenBonnie=-1,
            lastSeenBonnieTime=0.0,
            lastSeenChica=-1,
            lastSeenChicaTime=0.0,
            lastSeenFreddy=-1,
            lastSeenFreddyTime=0.0,
            lastSeenFoxy=-1,
            lastSeenFoxyTime=0.0,
            goldenFreddy=False,
            jumpscaredBy=0,
            blackout=False,
        )
        
        # Apply temporal coherence constraints - power always, usage with confidence-aware rules
        if self.night_started:
            state.power = self._apply_power_coherence(state.power)
            state.usage = self._apply_usage_coherence(state.usage, usage_confidence)
            self.last_usage = state.usage

            # Commit button states after coherence passes
            state.leftDoor = left_door
            state.rightDoor = right_door
            state.leftLight = left_light
            state.rightLight = right_light

            self.last_leftDoor = state.leftDoor
            self.last_rightDoor = state.rightDoor
            self.last_leftLight = state.leftLight
            self.last_rightLight = state.rightLight
        else:
            state.leftDoor = left_door
            state.rightDoor = right_door
            state.leftLight = left_light
            state.rightLight = right_light
        
        return state

    def _interaction_enabled(self) -> bool:
        """Return True if all interactive UI regions are configured."""
        return (
            self.left_door_region is not None
            and self.left_light_region is not None
            and self.right_door_region is not None
            and self.right_light_region is not None
            and self.camera_flip_region is not None
        )

    def _inside_region(self, x: int, y: int, region: Optional[UIRegion], w: int, h: int) -> bool:
        if region is None:
            return False
        x1, y1, x2, y2 = region.bounds
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            return False
        return (x1 <= x < x2) and (y1 <= y < y2)

    def _apply_interaction(self, frame: np.ndarray) -> None:
        """
        Update door/light/camera states based on the latest cursor sample.

        Input comes from `self.input_sample` as `(x, y, in_bounds, left_down)`.
        Rules:
        - Camera flips when the cursor enters the camera region from above,
            with `cam_anim_time` gating between flips.
        - Doors toggle on click rising edge when cursor is inside their region,
            with `door_anim_time` gating to mimic door animation cooldown.
        - Lights toggle on click rising edge; they are mutually exclusive.
        - While camera is up, lights are forced off and clicks do not toggle door/light.
        - Duplicate clicks on the exact same pixel within ~300ms are suppressed.
        """
        if self.input_sample is None:
            return
        cx, cy, in_bounds, left_down = self.input_sample
        h, w = frame.shape[:2]
        now = time.perf_counter()

        # Camera flip when entering region from outside, from above, with animation cooldown
        prev_y = self.last_cursor_pos[1] if self.last_cursor_pos is not None else None
        cam_top = self.camera_flip_region.y if self.camera_flip_region is not None else None
        enters_from_above = (
            prev_y is not None and cam_top is not None and prev_y < cam_top
        )
        in_cam = in_bounds and self._inside_region(cx, cy, self.camera_flip_region, w, h)
        if in_cam and not self.last_in_cam_region and enters_from_above and (now - self.last_cam_toggle_time) >= self.cam_anim_time:
            self.camera_up = not self.camera_up
            self.last_cam_toggle_time = now
            self.last_leftLight = False
            self.last_rightLight = False
            self.last_click_pos = None  # reset duplicate click memory after cam flip
        self.last_in_cam_region = in_cam

        # Door/light click handling on rising edge; toggles; lights mutually exclusive
        rising = (left_down and not self.last_left_down and in_bounds and not self.camera_up and (now - self.last_click_time) >= self.click_cooldown)
        same_pixel_double = (
            self.last_click_pos is not None and (cx, cy) == self.last_click_pos and (now - self.last_click_time) < 0.3
        )
        if rising and not same_pixel_double:
            if self._inside_region(cx, cy, self.left_door_region, w, h):
                if (now - self.last_left_door_toggle_time) >= self.door_anim_time:
                    self.last_leftDoor = not self.last_leftDoor
                    self.last_left_door_toggle_time = now
                    self.last_click_time = now
                    self.last_click_pos = (cx, cy)
            elif self._inside_region(cx, cy, self.left_light_region, w, h):
                self.last_leftLight = not self.last_leftLight
                if self.last_leftLight:
                    self.last_rightLight = False
                self.last_click_time = now
                self.last_click_pos = (cx, cy)
            elif self._inside_region(cx, cy, self.right_door_region, w, h):
                if (now - self.last_right_door_toggle_time) >= self.door_anim_time:
                    self.last_rightDoor = not self.last_rightDoor
                    self.last_right_door_toggle_time = now
                    self.last_click_time = now
                    self.last_click_pos = (cx, cy)
            elif self._inside_region(cx, cy, self.right_light_region, w, h):
                self.last_rightLight = not self.last_rightLight
                if self.last_rightLight:
                    self.last_leftLight = False
                self.last_click_time = now
                self.last_click_pos = (cx, cy)

        # Lights forced off while camera is up
        if self.camera_up:
            self.last_leftLight = False
            self.last_rightLight = False

        self.last_left_down = left_down
        self.last_cursor_pos = (cx, cy)

    
    def _detect_night_started(self, frame: np.ndarray) -> bool:
        """
        Detect whether an active night is being played (once per session).
        
        Uses template matching against pre-recorded office starting frame.
        Once nightStarted=True, it stays True until manually reset.
        
        Args:
            frame: BGR frame from game window
            
        Returns:
            True if frame matches office template above confidence threshold
        """
        if self.office_template is None:
            return False
        
        return self._match_template(frame, self.office_template, threshold=0.7)
    
    def _load_template(self, relative_path: str) -> Optional[np.ndarray]:
        """
        Load a template image from the templates directory.
        
        Args:
            relative_path: Path relative to template_dir (e.g., "office/starting_frame.png")
            
        Returns:
            Loaded image array or None if file not found
        """
        template_path = self.template_dir / relative_path
        
        if not template_path.exists():
            print(f"Warning: Template not found at {template_path}")
            return None
        
        template = cv2.imread(str(template_path))
        if template is None:
            print(f"Error: Failed to load template from {template_path}")
            return None
        
        return template
    
    def _match_template(self, frame: np.ndarray, template: np.ndarray, threshold: float = 0.7) -> bool:
        """
        Check whether a template matches the given frame above a threshold.

        Uses `cv2.matchTemplate` with TM_CCOEFF_NORMED. If the template and
        frame are the same size, performs a direct comparison. Otherwise, it
        scans the frame for the best match. Returns True if the maximum match
        score is greater than or equal to `threshold`.

        Args:
            frame: Current BGR frame to check
            template: Reference template image (BGR)
            threshold: Confidence threshold in [0.0, 1.0]

        Returns:
            True if a match is found above threshold, else False
        """
        if frame is None or template is None:
            return False

        try:
            fh, fw = frame.shape[:2]
            th, tw = template.shape[:2]

            # Convert both to grayscale for robust matching
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

            # If template is larger than frame in any dimension, resize template down
            if th > fh or tw > fw:
                scale_y = fh / float(th)
                scale_x = fw / float(tw)
                scale = min(scale_x, scale_y)
                new_w = max(1, int(tw * scale))
                new_h = max(1, int(th * scale))
                template_gray = cv2.resize(template_gray, (new_w, new_h))
                th, tw = template_gray.shape[:2]

            # Perform template matching
            result = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_CCOEFF_NORMED)
            if result.size == 0:
                return False

            max_val = float(np.max(result))
            return max_val >= threshold
        except cv2.error:
            return False
    
    def _extract_region(self, frame: np.ndarray, region: UIRegion) -> Optional[np.ndarray]:
        """
        Extract a fixed UI region from the frame for isolated detection.

        Args:
            frame: Full BGR frame from the game window
            region: UIRegion defining the area to extract (absolute pixels)

        Returns:
            Cropped region as a NumPy array, or None if out of bounds
        """
        if frame is None or region is None:
            return None

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = region.bounds

        # Bounds check
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h or x1 >= x2 or y1 >= y2:
            return None

        return frame[y1:y2, x1:x2]
    
    def _detect_usage(self, frame: np.ndarray) -> tuple[int, float]:
        """
        Detect current power usage level (1-5) from usage bar sprite.
        
        Extracts the fixed usage bar region and matches against loaded sprite templates.
        Returns the highest matching usage level and confidence score.
        
        Args:
            frame: Current game frame
            
        Returns:
            Tuple of (usage_level, confidence) where usage_level is 0-5 and confidence is 0.0-1.0
        """
        region = self._extract_region(frame, self.USAGE_BAR_REGION)
        if region is None:
            return 0, 0.0
        
        best_level = 0
        best_score = -1.0
        best_eff = -1.0  # score with area bias for tie-breaking
        second_level = 0
        second_eff = -1.0
        
        # Match against each usage sprite (1-5) and find best match
        for level in range(5, 0, -1):  # Check 5, 4, 3, 2, 1 (descending helps catch full bars first)
            if level not in self.usage_sprites:
                continue
            
            sprite, mask, masked_sprite = self.usage_sprites[level]
            
            # Skip if sprite doesn't match region dimensions
            if sprite.shape[0] != region.shape[0] or sprite.shape[1] != region.shape[1]:
                continue
            
            # Apply mask to region when available to suppress background noise
            try:
                if mask is not None:
                    masked_region = cv2.bitwise_and(region, region, mask=mask)
                    result = cv2.matchTemplate(masked_region, masked_sprite, cv2.TM_CCOEFF_NORMED)
                else:
                    result = cv2.matchTemplate(region, sprite, cv2.TM_CCOEFF_NORMED)
            except cv2.error:
                continue
            
            if result.size > 0:
                raw_score = float(np.max(result))
                area_norm = self.usage_sprite_area.get(level, 0) / float(self.usage_max_area or 1)
                eff_score = raw_score + (self.USAGE_AREA_WEIGHT * area_norm)

                # Prefer higher effective score; break exact ties by higher level number (fuller bar)
                if eff_score > best_eff or (abs(eff_score - best_eff) <= 1e-6 and level > best_level):
                    second_eff = best_eff
                    second_level = best_level
                    best_eff = eff_score
                    best_score = raw_score
                    best_level = level
                elif eff_score > second_eff:
                    second_eff = eff_score
                    second_level = level

        # Require both an absolute threshold (per level) and a margin over the next best
        level_min = self.USAGE_LEVEL_MIN_SCORES.get(best_level, self.USAGE_MATCH_THRESHOLD)
        margin_ok = (best_eff - second_eff) >= self.USAGE_MARGIN or (
            abs(best_eff - second_eff) <= 1e-6 and best_level > second_level
        )
        
        if best_score >= level_min and margin_ok:
            # Confidence is how much better the best match is than the second best
            confidence = max(0.0, min(1.0, (best_eff - second_eff) * 10.0))  # Scale margin to 0-1
            return best_level, confidence
        else:
            return 0, 0.0
    
    def _apply_power_coherence(self, detected_power: int) -> int:
        """
        Apply temporal coherence constraint to power readings.
        
        Power can only:
        - Stay the same
        - Decrease (power drain)
        
        Power NEVER increases. Detected increases are blocked (missing digits like "9" instead of "99").
        
        Args:
            detected_power: Power value detected from current frame
            
        Returns:
            Constrained power value (either detected or last valid value)
        """
        # Power increases are physically impossible - block them
        if detected_power > self.last_power:
            # Detected power went up - must be a recognition error
            return self.last_power
        
        # Allow decreases and same readings
        self.last_power = detected_power
        return detected_power
    
    def _apply_usage_coherence(self, detected_usage: int, confidence: float) -> int:
        """
        Apply temporal coherence constraint to usage readings when confidence is LOW.
        
        Used only for low-confidence detections (e.g., during camera flips when region is obscured).
        Blocks jumps of 2+ levels and transitions to 0 during gameplay.
        
        High-confidence detections bypass this check entirely.
        
        Args:
            detected_usage: Usage level detected from current frame
            
        Returns:
            Constrained usage value (either detected or last valid value)
        """
        # If no detection, keep last
        if detected_usage == 0:
            return self.last_usage

        # Allow small changes freely
        usage_diff = detected_usage - self.last_usage
        if abs(usage_diff) <= 1:
            return detected_usage

        # Larger jump: accept only if confident enough
        if confidence >= 0.6:
            return detected_usage

        # Otherwise, keep last (likely flip artifact)
        return self.last_usage
    
    def _detect_power(self, frame: np.ndarray) -> int:
        """
        Detect current power percentage (0-100) from power text display.
        
        Extracts the power percentage region (e.g., "99%") and matches individual digits
        against templates, then reconstructs the full number. Uses smoothing to reduce
        jitter from flickering backgrounds.
        
        Args:
            frame: Current game frame
            
        Returns:
            Power percentage (0-100) or 0 if no match found or region not yet discovered
        """
        # If region hasn't been discovered yet, return 0
        if self.power_percentage_region is None:
            return 0
        
        region = self._extract_region(frame, self.power_percentage_region)
        if region is None:
            return 0
        
        # Recognize digits from the region
        recognized = self._recognize_power_digits(region)
        
        if not recognized:
            # If recognition failed, use the smoothed previous value if available
            if self.power_smoothing_buffer:
                return int(np.median(self.power_smoothing_buffer))
            return 0
        
        # Check confidence: should have 2-3 characters (1-2 digits + percent sign)
        # If we're missing the percent sign, confidence is low (might be missing a digit too)
        has_percent = any(char == 10 for char in recognized)
        num_digits = sum(1 for char in recognized if char != 10)
        
        # Reconstruct the number from recognized digits
        # Find position of percent sign (10) if present
        percent_idx = -1
        for i, char in enumerate(recognized):
            if char == 10:  # 10 is percent sign
                percent_idx = i
                break
        
        # Get digits before the percent sign
        if percent_idx >= 0:
            digits = recognized[:percent_idx]
        else:
            # If no percent sign found, use all as digits (but low confidence)
            digits = recognized
        
        if not digits:
            # If recognition failed, use the smoothed previous value if available
            if self.power_smoothing_buffer:
                return int(np.median(self.power_smoothing_buffer))
            return 0
        
        try:
            power_value = int(''.join(map(str, digits)))
            power_value = min(power_value, 100)  # Clamp to 0-100
        except (ValueError, IndexError):
            # If parsing failed, use the smoothed previous value if available
            if self.power_smoothing_buffer:
                return int(np.median(self.power_smoothing_buffer))
            return 0
        
        # Apply coherence constraint: if missing percent sign or only 1 digit detected,
        # this is low confidence. Block large jumps.
        if not has_percent or num_digits < 2:
            # Low confidence detection - check against last known value
            if self.last_power is not None and power_value < self.last_power - 5:
                # Suspicious large drop - probably missing a digit
                # Return the smoothed value instead of this detection
                if self.power_smoothing_buffer:
                    return int(np.median(self.power_smoothing_buffer))
                return self.last_power
        
        # Apply smoothing: keep a buffer of recent readings and return the median
        self.power_smoothing_buffer.append(power_value)
        if len(self.power_smoothing_buffer) > self.power_smoothing_size:
            self.power_smoothing_buffer.pop(0)
        
        # Return median of buffer to smooth out jitter
        return int(np.median(self.power_smoothing_buffer))
    
    def _recognize_power_digits(self, region: np.ndarray) -> list[int]:
        """
        Recognize individual digits and percent sign from power percentage region.
        
        Segments the region into individual character bounding boxes, matches each
        against templates, and returns recognized characters (0-9 for digits, 10 for %).
        
        Args:
            region: Extracted power percentage region
            
        Returns:
            List of recognized digits (0-9) and percent sign (10) in order, or empty if failed
        """
        # Convert to grayscale for contour detection
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Threshold: bright white/light digits on dark background
        # Use a more lenient threshold to capture all character pixels
        _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        
        # Find contours (character blobs)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        
        # Get bounding boxes and sort left-to-right
        char_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter out very small noise but allow percent sign (11x14)
            # Digits are ~16x21, percent is ~11x14
            if w < 8 or h < 8:  # Minimum dimensions (filter tiny fragments)
                continue
            # Skip very tall/wide blobs (likely artifacts)
            if w > 25 or h > 25:
                continue
            char_boxes.append((x, y, w, h))
        
        if not char_boxes:
            return []
        
        # Sort by x position (left to right)
        char_boxes.sort(key=lambda b: b[0])
        
        # Recognize each character
        recognized_chars = []
        for x, y, w, h in char_boxes:
            char_roi = region[y:y+h, x:x+w]  # Use original region (BGR), not binary
            
            # Convert char ROI to grayscale for template matching
            char_roi_gray = cv2.cvtColor(char_roi, cv2.COLOR_BGR2GRAY)
            
            # Try to match against digit templates first
            best_digit = -1
            best_score = -1.0
            
            for digit in range(10):
                if digit not in self.power_digit_templates:
                    continue
                
                template = self.power_digit_templates[digit]
                
                # Resize template to match character ROI size
                template_resized = cv2.resize(template, (w, h))
                template_gray = cv2.cvtColor(template_resized, cv2.COLOR_BGR2GRAY)
                
                try:
                    # Match grayscale ROI against grayscale template
                    result = cv2.matchTemplate(char_roi_gray, template_gray, cv2.TM_CCOEFF_NORMED)
                    if result.size > 0:
                        score = float(np.max(result))
                        if score > best_score:
                            best_score = score
                            best_digit = digit
                except cv2.error:
                    continue
            
            # Determine if this is a digit or percent sign
            is_percent = False
            percent_score = -1.0
            
            # Check if it's a percent sign (try matching, separate from digit matching)
            if self.power_percent_template is not None:
                template = self.power_percent_template
                template_resized = cv2.resize(template, (w, h))
                template_gray = cv2.cvtColor(template_resized, cv2.COLOR_BGR2GRAY)
                
                try:
                    result = cv2.matchTemplate(char_roi_gray, template_gray, cv2.TM_CCOEFF_NORMED)
                    if result.size > 0:
                        percent_score = float(np.max(result))
                        # If percent sign matches well (>0.45), treat as percent
                        # OR if it matches better than the best digit
                        if percent_score > 0.45 or (percent_score > best_score and percent_score > 0.3):
                            recognized_chars.append(10)  # 10 represents '%'
                            is_percent = True
                except cv2.error:
                    pass
            
            # Add digit if it was recognized and not overridden as percent
            if not is_percent and best_digit >= 0 and best_score >= self.POWER_MATCH_THRESHOLD:
                recognized_chars.append(best_digit)
        
        return recognized_chars

