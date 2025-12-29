"""
UI Region Discovery Tool

Interactively finds and records UI element locations in FNAF screenshots.
Generates Python constants for game_state.py.

Usage:
    python -m tools.discover_ui_regions
    
    Supports:
    - Auto-detection using sprite matching
    - Manual coordinate entry
"""

import sys
from pathlib import Path
import cv2
import numpy as np
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.observer import FnafObserver
from src.game_state import UIRegion


class UIRegionDiscoverer:
    """Interactive tool for discovering UI element regions in FNAF."""
    
    def __init__(self):
        self.regions = {}
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from FNAF game window.
        
        Returns:
            Frame array or None if capture failed
        """
        print("Setting up frame capture...")
        observer = FnafObserver()

        # Locate the window and monitor bounds
        if not observer.find_window() or observer.monitor is None:
            print("ERROR: Failed to locate game window")
            return None

        # Grab a single screenshot of the monitor region (BGRA -> BGR)
        try:
            screenshot = np.array(observer.sct.grab(observer.monitor))
            frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
            print(f"✓ Captured frame: {frame.shape} from monitor {observer.monitor}")
            return frame
        except Exception as e:
            print(f"ERROR: Failed to grab frame: {e}")
            return None
    
    def auto_detect_usage_bar(self, frame: np.ndarray) -> Optional[UIRegion]:
        """
        Automatically detect usage bar by matching sprite.
        
        Args:
            frame: Game frame
            
        Returns:
            UIRegion if detected, None otherwise
        """
        print("\n" + "="*60)
        print("Auto-Detecting: usage_bar")
        print("="*60)
        
        # Load template sprite with alpha (for masking)
        template_path = Path(__file__).parent.parent / "templates" / "ui_elements" / "usage_4.png"
        if not template_path.exists():
            print(f"ERROR: Sprite not found at {template_path}")
            return None

        raw = cv2.imread(str(template_path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            print("ERROR: Failed to load sprite")
            return None

        if raw.shape[2] == 4:
            bgr = raw[:, :, :3]
            alpha = raw[:, :, 3]
            mask = alpha
        else:
            bgr = raw
            mask = None

        template = bgr
        print(f"Template size: {template.shape}; Masked: {mask is not None}")
        print(f"Frame size: {frame.shape}")

        # Limit search area to bottom-left quadrant to reduce false positives
        h, w = frame.shape[:2]
        search_x0 = 0
        search_x1 = min(500, w)  # usage bar lives near bottom-left; 500px is generous
        search_y0 = max(h - 250, 0)
        search_y1 = h
        roi = frame[search_y0:search_y1, search_x0:search_x1]
        print(f"Searching ROI: x[{search_x0},{search_x1}) y[{search_y0},{search_y1}) -> {roi.shape}")

        # Try multiple matching methods; use mask where supported
        methods = [
            (cv2.TM_CCOEFF_NORMED, "Normalized Cross-Correlation", False),
            (cv2.TM_CCORR_NORMED, "Normalized Correlation", True),
            (cv2.TM_SQDIFF_NORMED, "Normalized Squared Difference", True),
        ]

        best_overall_score = -1.0
        best_overall_match = None
        best_method_name = ""

        for method, method_name, supports_mask in methods:
            try:
                if supports_mask and mask is not None:
                    result = cv2.matchTemplate(roi, template, method, mask=mask)
                else:
                    result = cv2.matchTemplate(roi, template, method)
            except cv2.error as e:
                print(f"  {method_name}: OpenCV error: {e}")
                continue

            if result.size == 0:
                continue

            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            if method == cv2.TM_SQDIFF_NORMED:
                score = 1.0 - min_val  # lower is better -> convert
                match_loc = min_loc
            else:
                score = max_val
                match_loc = max_loc

            print(f"  {method_name}: score={score:.2%} at {match_loc}")

            if score > best_overall_score:
                best_overall_score = score
                best_overall_match = match_loc
                best_method_name = method_name

        if best_overall_match is None:
            print("ERROR: No matches found with any method")
            return None

        # Adjust match location back to full-frame coordinates
        x_roi, y_roi = best_overall_match
        x = x_roi + search_x0
        y = y_roi + search_y0
        confidence = best_overall_score

        print(f"\nBest match: {best_method_name}")
        print(f"Position: x={x}, y={y}")
        print(f"Confidence: {confidence:.2%}")

        if confidence < 0.55:
            print(f"\n⚠ WARNING: Low confidence ({confidence:.2%}). Manual adjustment may be needed.")
            resp = input("Use anyway? (y/n): ").strip().lower()
            if resp != 'y':
                return None

        width, height = template.shape[1], template.shape[0]
        region = UIRegion(x=x, y=y, width=width, height=height)
        
        # Show visual confirmation
        show = input("\nShow visual confirmation? (y/n): ").strip().lower()
        if show == 'y':
            display = frame.copy()
            x1, y1, x2, y2 = x, y, x + width, y + height
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                display,
                f"usage_bar: ({x}, {y}) {width}x{height} conf={confidence:.2%}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            cv2.imshow("Auto-Detection Result", display)
            print("Press any key to close visualization...")
            cv2.waitKey(0)
            cv2.destroyWindow("Auto-Detection Result")
        
        print(f"✓ Region detected: {region}")
        return region
    
    def manual_entry(self, element_name: str) -> Optional[UIRegion]:
        """Allow manual entry of coordinates."""
        print(f"\n{'='*60}")
        print(f"Manual Entry: {element_name}")
        print(f"{'='*60}")
        print("Enter region coordinates:")
        
        try:
            x = int(input("  X coordinate: "))
            y = int(input("  Y coordinate: "))
            width = int(input("  Width: "))
            height = int(input("  Height: "))
            
            region = UIRegion(x=x, y=y, width=width, height=height)
            print(f"✓ Region created: {region}")
            return region
        except ValueError:
            print("ERROR: Invalid input")
            return None
    
    def generate_code(self) -> str:
        """Generate Python code for discovered regions."""
        lines = [
            "# Generated UI constants - paste into GameStateExtractor",
            "# Fixed for this FNAF resolution",
            ""
        ]

        for element_name, region in self.regions.items():
            prefix = element_name.upper()
            lines.append(f"{prefix}_POS = ({region.x}, {region.y})  # (x, y)")
            lines.append(f"{prefix}_SIZE = ({region.width}, {region.height})  # (width, height)")
            lines.append(f"{prefix}_MATCH_THRESHOLD = 0.6  # tune as needed")
            lines.append(
                f"{prefix}_REGION = UIRegion(\n"
                f"    x={region.x}, y={region.y},\n"
                f"    width={region.width}, height={region.height}\n"
                f")"
            )
            lines.append("")

        return "\n".join(lines)
    
    def run(self):
        """Main discovery workflow."""
        print("\n" + "="*60)
        print("FNAF UI Region Discovery Tool")
        print("="*60)
        print("\nThis tool finds exact pixel coordinates for UI elements.")
        print()
        
        # Capture frame once
        frame = self.capture_frame()
        if frame is None:
            print("\nFailed to capture frame. Make sure FNAF is running.")
            return
        
        # Discover usage_bar
        print("\n" + "-"*60)
        print("Element: usage_bar")
        print("-"*60)
        print("Options:")
        print("  1. Auto-detect (fastest & most accurate)")
        print("  2. Manual entry")
        print("  3. Skip")
        
        choice = input("Choice (1-3): ").strip()
        
        if choice == "1":
            region = self.auto_detect_usage_bar(frame)
            if region:
                self.regions['usage_bar'] = region
        elif choice == "2":
            region = self.manual_entry('usage_bar')
            if region:
                self.regions['usage_bar'] = region
        # else: skip
        
        # Generate output
        if self.regions:
            print("\n" + "="*60)
            print("Generated Code")
            print("="*60)
            code = self.generate_code()
            print("\n" + code + "\n")
            
            # Save to file
            save = input("Save to file? (y/n): ").strip().lower()
            if save == 'y':
                output_path = Path(__file__).parent / "discovered_regions.py"
                with open(output_path, 'w') as f:
                    f.write(code + "\n")
                print(f"✓ Saved to {output_path}")
                print("\nNext steps:")
                print("1. Review the generated code above")
                print("2. Copy POS/SIZE/MATCH_THRESHOLD/REGION constants")
                print("3. Paste them into GameStateExtractor in src/game_state.py")
        else:
            print("\nNo regions discovered.")


if __name__ == "__main__":
    try:
        discoverer = UIRegionDiscoverer()
        discoverer.run()
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
