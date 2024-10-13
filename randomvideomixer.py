import cv2
import numpy as np
import os
from comfy.model_management import get_torch_device

class RandomVideoMixer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_folder": ("STRING", {"default": "path/to/video/folder"}),
                "output_duration": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 300.0, "step": 0.1}),
                "transition_duration": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 5.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "mix_videos"
    CATEGORY = "video"

    def mix_videos(self, video_folder, output_duration, transition_duration):
        # Get list of video files in the folder
        video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
        
        if len(video_files) < 2:
            raise ValueError("At least two video files are required in the specified folder.")

        output_path = os.path.join(video_folder, "mixed_output.mp4")
        
        # Initialize variables
        mixed_frames = []
        current_duration = 0

        while current_duration < output_duration:
            # Randomly select two videos
            video1, video2 = np.random.choice(video_files, 2, replace=False)
            
            cap1 = cv2.VideoCapture(os.path.join(video_folder, video1))
            cap2 = cv2.VideoCapture(os.path.join(video_folder, video2))
            
            fps = int(cap1.get(cv2.CAP_PROP_FPS))
            frame_count = int(min(cap1.get(cv2.CAP_PROP_FRAME_COUNT), cap2.get(cv2.CAP_PROP_FRAME_COUNT)))
            
            for i in range(frame_count):
                ret1, frame1 = cap1.read()
                ret2, frame2 = cap2.read()
                
                if not ret1 or not ret2:
                    break
                
                # Calculate alpha for smooth transition
                alpha = i / frame_count if i < frame_count * transition_duration else 1.0
                
                # Blend frames
                blended_frame = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
                mixed_frames.append(blended_frame)
                
                current_duration += 1 / fps
                if current_duration >= output_duration:
                    break
            
            cap1.release()
            cap2.release()

        # Write output video
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (mixed_frames[0].shape[1], mixed_frames[0].shape[0]))
        for frame in mixed_frames:
            out.write(frame)
        out.release()

        return (output_path,)

NODE_CLASS_MAPPINGS = {
    "RandomVideoMixer": RandomVideoMixer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RandomVideoMixer": "Random Video Mixer"
}