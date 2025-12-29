import time
import ctypes
from ctypes import wintypes
import cv2
import mss
import numpy as np
import pygetwindow as gw
from colorama import Fore, Style, init


WINDOW_NAME: str = "Five Nights at Freddy's"
TARGET_FPS: int = 24  # Desired FPS for observation loop


# Initialize color output
init(autoreset=True)

class FnafObserver:
    def __init__(self, window_title: str = WINDOW_NAME) -> None:
        self.window_title: str = window_title
        self.sct = mss.mss()
        self.monitor: dict[str, int] | None = None
        
        self.target_fps: int = TARGET_FPS
        self.interval: float = 1.0 / self.target_fps
        
    def find_window(self) -> bool:
        """Locates the game window and defines the capture region."""
        window_found: bool = False

        try:
            # Get the window object
            win: gw.Win32Window = gw.getWindowsWithTitle(self.window_title)[0]
            
            if not win.isActive:
                print(f"{Fore.YELLOW}Warning: Game window is not active. Focusing now...{Style.RESET_ALL}")
                win.activate()
                time.sleep(0.5) # Wait for focus

            # Compute the client-area rectangle (excludes title bar and borders)
            hwnd = win._hWnd  # Win32 window handle
            rect = wintypes.RECT()
            if ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(rect)):
                # Convert client (0,0) to screen coords to locate the content area
                pt = wintypes.POINT(0, 0)
                ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(pt))
                client_width = rect.right - rect.left
                client_height = rect.bottom - rect.top
                self.monitor = {
                    "top": int(pt.y),
                    "left": int(pt.x),
                    "width": int(client_width),
                    "height": int(client_height),
                }
            else:
                # Fallback: capture full window bounds
                self.monitor = {
                    "top": win.top,
                    "left": win.left,
                    "width": win.width,
                    "height": win.height,
                }
            
            print(f"{Fore.GREEN}Window Found: {win.title} | Geometry: {self.monitor}{Style.RESET_ALL}")
            window_found = True
            
        except IndexError:
            print(f"{Fore.RED}ERROR: Window '{self.window_title}' not found.{Style.RESET_ALL}")
            print("Please launch FNAF and ensure it is in Windowed Mode (Alt+Enter).")
        
        return window_found

    def start_observing(self):
        if not self.find_window() or self.monitor is None:
            return

        print(f"{Fore.CYAN}Starting Observation Loop at {self.target_fps} FPS... (Press Ctrl+C to stop){Style.RESET_ALL}")
        
        frame_count: int = 0
        start_time: float = time.time()
        
        try:
            while True:
                loop_start = time.time()

                # 1. Capture Frame
                screenshot = np.array(self.sct.grab(self.monitor))
                
                # 2. Process (Simulate minimal preprocessing)
                # Drop Alpha channel (BGRA -> BGR) to save memory
                frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
                
                # 3. Log Stats
                frame_count += 1
                elapsed = time.time() - start_time
                actual_fps = frame_count / elapsed

                print(f"FPS: {actual_fps:.2f} | Size: {frame.shape}")
                
                # Show what the bot sees (updates every frame)
                cv2.imshow("Bot View", frame)
                # Non-blocking key check with 1ms timeout
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt

                # 4. Latency Management
                # Calculate how long processing took
                process_time = time.time() - loop_start
                sleep_time = self.interval - process_time
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    print(f"{Fore.RED}LAG WARNING: Frame took {process_time:.4f}s (Target: {self.interval}s){Style.RESET_ALL}")

        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            cv2.destroyAllWindows()
            print(f"Session ended. Avg FPS: {frame_count / (time.time() - start_time):.2f}")


def main() -> None:
    bot = FnafObserver(WINDOW_NAME)
    bot.start_observing()


if __name__ == "__main__":
    main()