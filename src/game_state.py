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
    """Extracts game state from FNAF screen captures."""
    
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
        
        self.night_started: bool = False
        self.night_start_time: Optional[float] = None  # Wall-clock time when night began
    
    def extract(self, frame: np.ndarray) -> GameState:
        """
        Extract complete game state from a frame.
        
        Args:
            frame: BGR frame from the game window
            
        Returns:
            GameState object with all detected values
        """
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
        detected_power = self._detect_power(frame) if self.night_started else 0
        detected_usage, usage_confidence = self._detect_usage(frame) if self.night_started else (0, 0.0)

        # Smooth usage over recent frames to damp camera-flip glitches
        if self.night_started:
            self.usage_smoothing_buffer.append(detected_usage)
            if len(self.usage_smoothing_buffer) > self.usage_smoothing_size:
                self.usage_smoothing_buffer.pop(0)
            nonzero = [u for u in self.usage_smoothing_buffer if u > 0]
            if nonzero:
                detected_usage = int(np.median(nonzero))
            else:
                detected_usage = 0
        
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
            isOnCamera=False,
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
        
        return state
    
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
        Compare frame to template using normalized cross-correlation.
        
        Args:
            frame: Current frame to check
            template: Reference template to match against
            threshold: Confidence threshold (0.0-1.0) for match
            
        Returns:
            True if frame matches template above threshold
        """
        # Ensure frames are the same size
        if frame.shape != template.shape:
            return False
        
        # Use normalized cross-correlation for robust matching
        result = cv2.matchTemplate(
            frame, 
            template, 
            cv2.TM_CCOEFF_NORMED
        )
        
        # Get best match score
        if result.size > 0:
            max_val = float(np.max(result))
            return max_val > threshold
        
        return False
    
    def _extract_region(self, frame: np.ndarray, region: UIRegion) -> Optional[np.ndarray]:
        """
        Extract a fixed UI region from a frame for isolated detection.
        
        Args:
            frame: Full frame from game window
            region: UIRegion defining the area to extract
            
        Returns:
            Cropped region or None if region is out of bounds
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = region.bounds
        
        # Bounds check
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
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

