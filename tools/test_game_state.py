"""
Test Game State Extraction

Simple script to verify game state detection is working correctly.
Captures frames and prints detected state in real-time.

Usage:
    python -m tools.test_game_state
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import json

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.observer import FnafObserver, WINDOW_NAME
from src.game_state import GameStateExtractor


def test_state_extraction() -> None:
    """
    Real-time game state extraction test.
    
    Captures frames and displays:
    - nightStarted detection
    - Raw frame for manual inspection
    - Full state JSON
    """
    observer = FnafObserver(WINDOW_NAME)
    extractor = GameStateExtractor()
    
    if not observer.find_window() or observer.monitor is None:
        return
    
    print("\n" + "="*70)
    print("GAME STATE EXTRACTION TEST")
    print("="*70)
    print("\nCapturing frames and extracting state...")
    print("Press 'Q' to quit.\n")
    
    monitor = observer.monitor
    frame_count = 0
    
    try:
        while True:
            screenshot = np.array(observer.sct.grab(monitor))
            frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
            
            # Extract state
            state = extractor.extract(frame)
            frame_count += 1
            
            # Display frame with state info
            display_frame = frame.copy()
            h, w = display_frame.shape[:2]
            
            # Draw state indicator
            color = (0, 255, 0) if state.nightStarted else (0, 0, 255)
            status_text = "NIGHT ACTIVE" if state.nightStarted else "MENU/LOADING"
            
            cv2.putText(
                display_frame,
                f"Status: {status_text}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2
            )
            cv2.putText(
                display_frame,
                f"Frame: {frame_count}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                1
            )

            # Show current usage level
            cv2.putText(
                display_frame,
                f"Usage: {state.usage}",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 200, 255),
                2
            )
            
            # Display time elapsed if night is active
            if state.nightStarted:
                minutes = int(state.timeElapsed // 60)
                seconds = int(state.timeElapsed % 60)
                cv2.putText(
                    display_frame,
                    f"Time: {minutes}:{seconds:02d}",
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    1
                )
            cv2.putText(
                display_frame,
                "Press Q to quit",
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1
            )
            
            cv2.imshow("Game State Extraction Test", display_frame)
            
            # Print state every 24 frames (1 second at 24 FPS)
            if frame_count % 24 == 0:
                print(f"\nFrame {frame_count}:")
                print(f"  nightStarted: {state.nightStarted}")
                print(f"  usage: {state.usage}")
                print(f"  Full state:")
                state_dict = state.to_dict()
                print(json.dumps(state_dict, indent=4))
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        cv2.destroyAllWindows()
        print(f"\nTest completed. Processed {frame_count} frames.")


if __name__ == "__main__":
    test_state_extraction()
