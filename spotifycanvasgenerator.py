import torch
import numpy as np
from PIL import Image
import librosa
import cv2
import random

class SpotifyCanvasGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_file": ("STRING", {"default": "input.mp3"}),
                "cover_art": ("IMAGE",),
                "duration": ("FLOAT", {"default": 7.0, "min": 5.0, "max": 9.0}),
                "fps": ("INT", {"default": 30, "min": 24, "max": 60}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_canvas"

    CATEGORY = "artcpu/video"

    def generate_canvas(self, audio_file, cover_art, duration, fps, intensity):
        # Set canvas dimensions (9:16 ratio)
        canvas_width, canvas_height = 540, 960

        # Load and process audio
        y, sr = librosa.load(audio_file, duration=duration)
        
        # Calculate actual duration of loaded audio
        audio_duration = len(y) / sr
        
        # Extract audio features
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        
        # Prepare cover art
        cover_art = cover_art.squeeze(0)  # Remove batch dimension
        cover_art = cover_art.permute(2, 0, 1)  # Change to [3, H, W] for torchvision
        cover_art = torch.nn.functional.interpolate(cover_art.unsqueeze(0), size=(canvas_width, canvas_width), mode='bilinear', align_corners=False)
        cover_art = cover_art.squeeze(0).permute(1, 2, 0).numpy()  # Back to [H, W, 3] numpy array
        cover_art_np = (cover_art * 255).astype(np.uint8)  # Convert to 0-255 range
        
        # Generate video frames
        num_frames = int(audio_duration * fps)
        frames = []
        
        for i in range(num_frames):
            # Get audio features for this frame
            frame_time = i / fps
            mfcc_frame = mfcc[:, min(int(frame_time / audio_duration * mfcc.shape[1]), mfcc.shape[1] - 1)]
            chroma_frame = chroma[:, min(int(frame_time / audio_duration * chroma.shape[1]), chroma.shape[1] - 1)]
            onset_frame = onset_env[min(int(frame_time / audio_duration * len(onset_env)), len(onset_env) - 1)]
            
            # Create canvas
            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
            
            # Apply audio-reactive effects
            
            # 1. Dynamic cover art placement
            max_offset = (canvas_height - canvas_width) // 2
            y_offset = int(np.sin(frame_time * 2 * np.pi / audio_duration) * max_offset * intensity)
            y_offset = max_offset + y_offset  # Ensure y_offset is always positive
            canvas[y_offset:y_offset+canvas_width, :] = cover_art_np
            
            # 2. Color overlay based on MFCC
            color_overlay = (mfcc_frame[:3] * intensity * 25).astype(np.uint8)
            color_overlay = np.clip(color_overlay, 0, 255)
            color_mask = np.full((canvas_height, canvas_width, 3), color_overlay, dtype=np.uint8)
            canvas = cv2.addWeighted(canvas, 0.7, color_mask, 0.3, 0)
            
            # 3. Pulsating effect on beats
            if librosa.util.is_unique(beat_frames):
                beat_time = librosa.frames_to_time(beat_frames, sr=sr)
                nearest_beat = beat_time[np.argmin(np.abs(beat_time - frame_time))]
                if abs(frame_time - nearest_beat) < 0.1:
                    zoom_factor = 1 + (0.05 * intensity)
                    zoomed = cv2.resize(canvas, None, fx=zoom_factor, fy=zoom_factor)
                    start_y = (zoomed.shape[0] - canvas.shape[0]) // 2
                    start_x = (zoomed.shape[1] - canvas.shape[1]) // 2
                    canvas = zoomed[start_y:start_y+canvas.shape[0], start_x:start_x+canvas.shape[1]]
            
            # 4. Particle effect based on chroma
            for _ in range(int(50 * intensity)):
                x = random.randint(0, canvas_width-1)
                y = random.randint(0, canvas_height-1)
                color = (chroma_frame * 255).astype(np.uint8)
                size = int(max(1, onset_frame * intensity * 5))
                # Use only the first 3 values of color, and ensure it's a tuple
                cv2.circle(canvas, (x, y), size, tuple(color[:3].tolist()), -1)
            
            frames.append(Image.fromarray(canvas))
        
        return (frames,)