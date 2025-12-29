"""
Game State Extractor

Extracts structured game state from FNAF screen captures.
Processes frame-by-frame to detect game state, animatronic positions, and player actions.
"""

from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path
import cv2
import numpy as np
from enum import IntEnum


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
    
    def __init__(self, template_dir: str | Path = "templates"):
        """
        Initialize the game state extractor.
        
        Args:
            template_dir: Path to directory containing pre-recorded template images
        """
        self.template_dir = Path(template_dir)
        
        # Load reference templates
        self.office_template = self._load_template("office/starting_frame.png")
        self.night_started: bool = False
    
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
        
        # Start with minimum viable state (only nightStarted filled in)
        state = GameState(
            nightStarted=self.night_started,
            timeElapsed=0.0,
            power=0,
            usage=0,
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
