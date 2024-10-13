import numpy as np
from PIL import Image
import cv2
import os

class VideoWriter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frames": ("IMAGE",),
                "fps": ("INT", {"default": 30, "min": 1, "max": 60}),
                "output_path": ("STRING", {"default": "output.mp4"}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    FUNCTION = "write_video"

    CATEGORY = "video"
    
    def write_video(self, frames, fps, output_path):
        if not frames:
            return "No frames to write", None

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Get dimensions from the first frame
        first_frame = frames[0]
        if isinstance(first_frame, Image.Image):
            width, height = first_frame.size
        else:  # Assume it's a numpy array
            height, width = first_frame.shape[:2]

        # Initialize the video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        last_frame = None
        for frame in frames:
            # Convert PIL Image to numpy array if necessary
            if isinstance(frame, Image.Image):
                frame = np.array(frame)
            
            # Ensure the frame is in BGR color space
            if frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            out.write(frame)
            last_frame = frame

        out.release()

        # Convert the last frame back to RGB for display
        if last_frame is not None:
            last_frame = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
            last_frame = Image.fromarray(last_frame)

        return (f"Video saved to {output_path}", last_frame)