import os
import shutil
import shlex
from pathlib import Path

# Set destination and clean it
dest = Path(__file__).resolve().parent / "../data/uploaded_images"
shutil.rmtree(dest, ignore_errors=True)
dest.mkdir(parents=True, exist_ok=True)

# Prompt for drag-and-drop
print("👉 Drag and drop image files or folders here, then press Enter:")
raw_input = input().strip()

# Use shlex to handle paths with spaces and quotes correctly
paths = shlex.split(raw_input)

# Supported image extensions
image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

copied = 0
for p in paths:
    path = Path(p)
    if path.is_dir():
        for file in path.rglob("*"):
            if file.suffix.lower() in image_exts and file.is_file():
                shutil.copy2(file, dest)
                copied += 1
    elif path.is_file() and path.suffix.lower() in image_exts:
        shutil.copy2(path, dest)
        copied += 1
    else:
        print(f"⚠️ Skipped (not image or not found): {path}")

print(f"✅ Done! {copied} image(s) copied to: {dest.resolve()}")
