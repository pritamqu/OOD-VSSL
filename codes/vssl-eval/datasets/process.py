"""
this script resize the video files to the shorter side at 256 px, 
and convert mp4 files to avi.
"""

import os
import sys
import subprocess, shlex
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count
import time
import datetime
import glob


def process2(file):
    '''Start up a new ffmpeg subprocess transcode the given video file
    and save the newly transcoded file to a new directory with the same
    directory tree '''
    
    # works fine with linux
    
    old_filename = os.path.join(source, subset, file)
    new_filename = os.path.join(destination, subset, file.replace('.mp4', '.avi', 1))

    if os.path.exists(new_filename):
        print(f'{old_filename} Skipped')
        return True

    result = os.popen(
        f'ffprobe -hide_banner -loglevel error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 {old_filename}'  # noqa:E501
    )
    try:
        w, h = [int(d) for d in result.readline().rstrip().split(',')]
    except:
        # some videos are damaged, just skip
        print(f'{old_filename} failed')
        return True
    
    if w > h:
        cmd = (f'ffmpeg -hide_banner -loglevel error -i {old_filename} '
               f'-c:v mpeg4 -vtag xvid '
               f'-filter_complex scale=-2:256 '
               f'-c:a copy -r 30 -ar 44100 '
               f'{new_filename} -y')
    else:
        cmd = (f'ffmpeg -hide_banner -loglevel error -i {old_filename} '
               f'-c:v mpeg4 -vtag xvid '
               f'-filter_complex scale=256:-2 '
               f'-c:a copy -r 30 -ar 44100 '
               f'{new_filename} -y')


    # os.popen(cmd) # debug with this
    completed = subprocess.run(cmd,
                                stderr=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL,
                                stdin=subprocess.PIPE,
                                shell=True)
    
    # print(f"'old: {old_filename} - new: {new_filename}' - return code: {completed.returncode}")
    if completed.returncode != 0:
        print(f'{old_filename} failed')
    else:
        print(f'{old_filename} success')

    return True


if __name__ == "__main__":
    
    global source, destination, subset
    print(f'total cpus: {cpu_count()}')
    
    source = sys.argv[1] # source dir of kinetics
    subset = sys.argv[2] # train or val
    destination = sys.argv[3] # root of destination directory
           
    filenames = ['/'.join(fn.split('/')[-1:]) for fn in glob.glob(f"{source}/{subset}/*.mp4")]
    os.makedirs(os.path.join(destination, subset), exist_ok=True)
    STARTTIME = time.time()

    with ThreadPool(cpu_count())as p:
           
            
        print(f"Transcoding {len(filenames)} Video files")
        JOBS = p.map(process2, filenames)

    print("Done")
    sys.exit(0)
             
# example of usage:            
# python process_kinetics.py '/scratch/ssd004/datasets/kinetics400' 'train' '/scratch/ssd004/datasets/kinetics400/kinetics400_processed'
# python process_kinetics.py '/scratch/ssd004/datasets/kinetics400' 'val' '/scratch/ssd004/datasets/kinetics400/kinetics400_processed'
