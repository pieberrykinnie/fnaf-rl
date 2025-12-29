"""
Template Manager

Utility for collecting, organizing, and managing reference templates
for game state detection.
"""

from pathlib import Path
from datetime import datetime
import cv2
import numpy as np


class TemplateManager:
    """Manages template collection and organization."""
    
    TEMPLATE_DIRS = {
        "office": "templates/office",
        "ui_elements": "templates/ui_elements",
        "animatronics": "templates/animatronics",
    }
    
    def __init__(self, base_path: str | Path = "."):
        """
        Initialize template manager.
        
        Args:
            base_path: Base directory for template operations
        """
        self.base_path = Path(base_path)
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Ensure all template directories exist."""
        for dir_path in self.TEMPLATE_DIRS.values():
            full_path = self.base_path / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
    
    def save_office_starting_frame(self, frame: np.ndarray, label: str = "") -> Path:
        """
        Save a starting office frame as reference template.
        
        Args:
            frame: BGR image array to save
            label: Optional label/description for file naming
            
        Returns:
            Path to saved template
        """
        return self._save_template(
            frame,
            "office",
            "starting_frame.png",
            label
        )
    
    def save_ui_element(self, frame: np.ndarray, element_name: str, label: str = "") -> Path:
        """
        Save a UI element template.
        
        Args:
            frame: BGR image array to save
            element_name: Name of UI element (e.g., "power_bar", "door_closed")
            label: Optional label/description for file naming
            
        Returns:
            Path to saved template
        """
        return self._save_template(
            frame,
            "ui_elements",
            f"{element_name}.png",
            label
        )
    
    def save_animatronic_template(self, frame: np.ndarray, animatronic: str, 
                                  location: str, label: str = "") -> Path:
        """
        Save an animatronic detection template.
        
        Args:
            frame: BGR image array to save
            animatronic: Animatronic name (bonnie, chica, freddy, foxy)
            location: Location/state (e.g., "stage", "west_hall", "at_door")
            label: Optional label/description for file naming
            
        Returns:
            Path to saved template
        """
        filename = f"{animatronic}_{location}.png"
        return self._save_template(
            frame,
            "animatronics",
            filename,
            label
        )
    
    def _save_template(self, frame: np.ndarray, category: str, 
                      filename: str, label: str = "") -> Path:
        """
        Save a template image with optional versioning.
        
        Args:
            frame: BGR image array to save
            category: Template category (office, ui_elements, animatronics)
            filename: Base filename
            label: Optional descriptive label
            
        Returns:
            Path to saved file
        """
        if category not in self.TEMPLATE_DIRS:
            raise ValueError(f"Unknown category: {category}")
        
        dir_path = self.base_path / self.TEMPLATE_DIRS[category]
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Handle versioning if file exists
        file_path = dir_path / filename
        if file_path.exists():
            # Add timestamp/version
            stem = file_path.stem
            suffix = file_path.suffix
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            versioned_name = f"{stem}_{timestamp}{suffix}"
            file_path = dir_path / versioned_name
        
        # Save the frame
        success = cv2.imwrite(str(file_path), frame)
        if not success:
            raise IOError(f"Failed to save template to {file_path}")
        
        print(f"Saved template: {file_path}")
        if label:
            print(f"  Label: {label}")
        
        return file_path
    
    def list_templates(self, category: str | None = None) -> dict[str, list]:
        """
        List all available templates.
        
        Args:
            category: Optional specific category to list (None = all)
            
        Returns:
            Dictionary of category -> list of template files
        """
        templates = {}
        
        categories = [category] if category else self.TEMPLATE_DIRS.keys()
        
        for cat in categories:
            if cat not in self.TEMPLATE_DIRS:
                continue
            
            dir_path = self.base_path / self.TEMPLATE_DIRS[cat]
            if dir_path.exists():
                templates[cat] = [
                    f.name for f in dir_path.glob("*.png")
                ]
            else:
                templates[cat] = []
        
        return templates
    
    def print_inventory(self) -> None:
        """Print a formatted inventory of all templates."""
        templates = self.list_templates()
        
        print("\n" + "="*60)
        print("Template Inventory")
        print("="*60)
        
        for category, files in templates.items():
            print(f"\n{category.upper()} ({len(files)} templates)")
            print("-" * 60)
            if files:
                for filename in sorted(files):
                    print(f"  • {filename}")
            else:
                print("  (empty)")
        
        print("\n" + "="*60 + "\n")


def main() -> None:
    """Demo: show template inventory."""
    manager = TemplateManager()
    manager.print_inventory()


if __name__ == "__main__":
    main()
