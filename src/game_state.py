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
        
        # Calculate time elapsed since night started
        time_elapsed = 0.0
        if self.night_started and self.night_start_time is not None:
            time_elapsed = time.perf_counter() - self.night_start_time
        
        # Start with minimum viable state (only nightStarted and timeElapsed filled in)
        state = GameState(
            nightStarted=self.night_started,
            timeElapsed=time_elapsed,
            power=0,
            usage=self._detect_usage(frame) if self.night_started else 0,
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
    
    def _detect_usage(self, frame: np.ndarray) -> int:
        """
        Detect current power usage level (1-5) from usage bar sprite.
        
        Extracts the fixed usage bar region and matches against loaded sprite templates.
        Returns the highest matching usage level.
        
        Args:
            frame: Current game frame
            
        Returns:
            Usage level (1-5) or 0 if no match found
        """
        region = self._extract_region(frame, self.USAGE_BAR_REGION)
        if region is None:
            return 0
        
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
        return best_level if (best_score >= level_min and margin_ok) else 0
