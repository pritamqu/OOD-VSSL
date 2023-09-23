import av
import numpy as np
import math
from fractions import Fraction
from scipy.interpolate import interp1d

av.logging.set_level(0)

def av_open(inpt):
    try:
        container = av.open(inpt)
    except:
        container = av.open(inpt, metadata_errors="ignore")
    return container

# borrowed from from AVID
def av_load_video(container, video_fps=None, start_time=0, duration=None):
    video_stream = container.streams.video[0]
    _ss = video_stream.start_time * video_stream.time_base
    _dur = video_stream.duration * video_stream.time_base
    _ff = _ss + _dur
    _fps = video_stream.average_rate

    if video_fps is None:
        video_fps = _fps

    if duration is None:
        duration = _ff - start_time

    # Figure out which frames to decode
    outp_times = [t for t in np.arange(start_time, min(start_time + duration - 0.5/_fps, _ff), 1./video_fps)][:int(duration*video_fps)]
    outp_vframes = [int((t - _ss) * _fps) for t in outp_times]
    start_time = outp_vframes[0] / float(_fps)

    # Fast forward
    container.seek(int(start_time * av.time_base))

    # Decode snippet
    frames = []
    for frame in container.decode(video=0):
        if len(frames) == len(outp_vframes):
            break   # All frames have been decoded
        frame_no = frame.pts * frame.time_base * _fps
        if frame_no < outp_vframes[len(frames)]:
            continue    # Not the frame we want

        # Decode
        pil_img = frame.to_image()
        while frame_no >= outp_vframes[len(frames)]:
            frames += [pil_img]
            if len(frames) == len(outp_vframes):
                break   # All frames have been decoded

    return frames, video_fps, start_time


