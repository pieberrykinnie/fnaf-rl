"""
Template Collection Tool

Interactive utility for capturing and saving template screenshots
during gameplay. Run this to set up your template directory.

Usage:
    python -m tools.frame_capture
    
    Follow on-screen prompts to capture frames at key moments:
    - Press 'S' to save current frame as a template
    - Press 'Q' to quit
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.observer import FnafObserver, WINDOW_NAME
from tools.template_manager import TemplateManager


class FrameCapture:
    """Interactive frame capture tool for template collection."""
    
    def __init__(self, window_title: str = WINDOW_NAME, template_base: str = "."):
        """
        Initialize frame capture tool.
        
        Args:
            window_title: Game window title
            template_base: Base path for template storage
        """
        self.observer = FnafObserver(window_title)
        self.template_manager = TemplateManager(template_base)
        self.current_frame: np.ndarray | None = None
        self.frame_count: int = 0
    
    def capture_templates(self) -> None:
        """
        Start interactive frame capture for template collection.
        
        Key bindings:
            S - Save current frame as starting office template
            1 - Save as ui_element template (prompts for name)
            2 - Save as animatronic template (prompts for details)
            Q - Quit
        """
        if not self.observer.find_window() or self.observer.monitor is None:
            return
        
        print("\n" + "="*70)
        print("TEMPLATE CAPTURE TOOL")
        print("="*70)
        print("\nPosition the game window and start gameplay.")
        print("Use these keys to capture templates:")
        print("  S - Save current frame as OFFICE starting frame")
        print("  1 - Save as UI ELEMENT template")
        print("  2 - Save as ANIMATRONIC template")
        print("  Q - Quit\n")
        print("="*70 + "\n")
        
        monitor = self.observer.monitor
        
        try:
            while True:
                screenshot = np.array(self.observer.sct.grab(monitor))
                frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
                self.current_frame = frame
                self.frame_count += 1
                
                # Display frame with instructions
                display_frame = frame.copy()
                h, w = display_frame.shape[:2]
                
                # Add text overlay with current instructions
                cv2.putText(
                    display_frame,
                    f"Frame {self.frame_count} | Press S/1/2 to save, Q to quit",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                cv2.imshow("Template Capture - Press S/1/2 to save, Q to quit", display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    print("\nQuitting template capture...")
                    break
                elif key == ord('s') or key == ord('S'):
                    self._save_office_template()
                elif key == ord('1'):
                    self._save_ui_element()
                elif key == ord('2'):
                    self._save_animatronic()
        
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        finally:
            cv2.destroyAllWindows()
            self.template_manager.print_inventory()
    
    def _save_office_template(self) -> None:
        """Save current frame as office starting template."""
        if self.current_frame is None:
            print("No frame available!")
            return
        
        label = f"Captured at frame {self.frame_count}"
        self.template_manager.save_office_starting_frame(self.current_frame, label)
        print(f"\n✓ Saved office starting frame (#{self.frame_count})\n")
    
    def _save_ui_element(self) -> None:
        """Save current frame as UI element template."""
        if self.current_frame is None:
            print("No frame available!")
            return
        
        print("\nSaving as UI Element template...")
        element_name = input("Element name (e.g., 'power_bar', 'door_left', 'camera_up'): ").strip()
        
        if not element_name:
            print("Cancelled.")
            return
        
        label = f"Captured at frame {self.frame_count}"
        self.template_manager.save_ui_element(self.current_frame, element_name, label)
        print(f"✓ Saved UI element template\n")
    
    def _save_animatronic(self) -> None:
        """Save current frame as animatronic template."""
        if self.current_frame is None:
            print("No frame available!")
            return
        
        print("\nSaving as Animatronic template...")
        animatronic = input("Animatronic (bonnie/chica/freddy/foxy): ").strip().lower()
        location = input("Location/state (e.g., 'stage', 'west_hall', 'at_door'): ").strip()
        
        if not animatronic or not location:
            print("Cancelled.")
            return
        
        label = f"Captured at frame {self.frame_count}"
        self.template_manager.save_animatronic_template(
            self.current_frame,
            animatronic,
            location,
            label
        )
        print(f"✓ Saved animatronic template\n")


def main() -> None:
    """Run the template capture tool."""
    capturer = FrameCapture()
    capturer.capture_templates()


if __name__ == "__main__":
    main()
