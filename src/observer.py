import time
import ctypes
from ctypes import wintypes
from collections.abc import Callable
import cv2
import mss
import numpy as np
import pygetwindow as gw
from colorama import Fore, Style, init


WINDOW_NAME: str = "Five Nights at Freddy's"
TARGET_FPS: int = 24  # Desired FPS for observation loop
Monitor = dict[str, int]
CursorInfo = tuple[int, int, bool]
MouseSample = tuple[int, int, bool, bool]
user32 = ctypes.windll.user32


# Initialize color output
init(autoreset=True)

class FnafObserver:
    def __init__(self, window_title: str = WINDOW_NAME) -> None:
        self.window_title: str = window_title
        self.sct = mss.mss()
        self.monitor: Monitor | None = None
        self.hwnd: int | None = None
        
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
            self.hwnd = hwnd
            rect = wintypes.RECT()
            if user32.GetClientRect(hwnd, ctypes.byref(rect)):
                # Convert client (0,0) to screen coords to locate the content area
                pt = wintypes.POINT(0, 0)
                user32.ClientToScreen(hwnd, ctypes.byref(pt))
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

    def start_observing(self) -> None:
        """Start capturing frames and cursor position at target FPS."""
        if not self.find_window() or self.monitor is None:
            return

        monitor = self.monitor
        grab = self.sct.grab
        to_bgr = cv2.cvtColor
        imshow = cv2.imshow
        wait_key = cv2.waitKey
        sleep = time.sleep
        interval = self.interval
        cursor_fetch = self._make_cursor_fetcher(monitor)

        frame_count: int = 0
        start_time: float = time.perf_counter()

        print(f"{Fore.CYAN}Starting Observation Loop at {self.target_fps} FPS... (Press Ctrl+C to stop){Style.RESET_ALL}")

        try:
            while True:
                loop_start = time.perf_counter()

                screenshot = np.array(grab(monitor))
                frame = to_bgr(screenshot, cv2.COLOR_BGRA2BGR)

                cursor_pos = cursor_fetch()

                frame_count += 1
                elapsed = time.perf_counter() - start_time
                actual_fps = frame_count / elapsed
                if cursor_pos is not None:
                    cx, cy, in_bounds, left_down = cursor_pos
                    print(
                        f"FPS: {actual_fps:.2f} | Size: {frame.shape} | Cursor: ({cx}, {cy}) | "
                        f"InWindow:{in_bounds} L:{left_down}"
                    )
                else:
                    print(f"FPS: {actual_fps:.2f} | Size: {frame.shape} | Cursor: unavailable")

                imshow("Bot View", frame)
                if wait_key(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt

                process_time = time.perf_counter() - loop_start
                sleep_time = interval - process_time
                if sleep_time > 0:
                    sleep(sleep_time)
                else:
                    print(f"{Fore.RED}LAG WARNING: Frame took {process_time:.4f}s (Target: {interval}s){Style.RESET_ALL}")

        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            cv2.destroyAllWindows()
            print(f"Session ended. Avg FPS: {frame_count / (time.perf_counter() - start_time):.2f}")

    def _make_cursor_fetcher(self, monitor: Monitor) -> Callable[[], MouseSample | None]:
        """Builds a zero-allocation cursor + left-button sampler relative to the given monitor."""
        left = monitor["left"]
        top = monitor["top"]
        width = monitor["width"]
        height = monitor["height"]

        pt = wintypes.POINT()
        get_cursor_pos = user32.GetCursorPos
        get_key = user32.GetAsyncKeyState
        VK_LBUTTON = 0x01

        def fetch() -> MouseSample | None:
            if not get_cursor_pos(ctypes.byref(pt)):
                return None

            rel_x = pt.x - left
            rel_y = pt.y - top
            in_bounds = 0 <= rel_x < width and 0 <= rel_y < height
            left_down = bool(get_key(VK_LBUTTON) & 0x8000)
            return rel_x, rel_y, in_bounds, left_down

        return fetch


def main() -> None:
    bot = FnafObserver(WINDOW_NAME)
    bot.start_observing()


if __name__ == "__main__":
    main()
