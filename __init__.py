from .randomvideomixer import RandomVideoMixer
from .spotifycanvasgenerator import SpotifyCanvasGenerator
from .videowriter import VideoWriter

NODE_CLASS_MAPPINGS = { 
    "RandomVideoMixer": RandomVideoMixer,
    "SpotifyCanvasGenerator": SpotifyCanvasGenerator,
    "VideoWriter": VideoWriter
    }

NODE_DISPLAY_NAME_MAPPINGS = {
    "RandomVideoMixer": "Random Video Mixer",
    "SpotifyCanvasGenerator": "Spotify Canvas Generator",
    "VideoWriter": "Video Writer"
}