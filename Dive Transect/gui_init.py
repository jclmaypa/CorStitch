# GNU GENERAL PUBLIC LICENSE
# CorStitch Copyright (C) 2025  Julian Christopher L. Maypa, Johnenn R. Manalang, and Maricor N. Soriano 
# This program comes with ABSOLUTELY NO WARRANTY;
# This is free software, and you are welcome to redistribute it under the conditions specified in the GNU General Public License.; 
# Please properly cite our paper when using this software: https://arxiv.org/abs/2505.00462

import cv2
import numpy as np
import os
from PIL import Image, ImageFile
import pandas as pd
import time
import scipy as sp
import datetime
from pathlib import Path
from gpxcsv import gpxtolist
import gc
from alive_progress import alive_bar
from pyfftw.interfaces.numpy_fft import fft2, ifft2, fftshift
import pyfftw
pyfftw.interfaces.cache.enable()
from pathlib import Path
from scipy.spatial import cKDTree
import av

NUM_THREADS = max(1, os.cpu_count() - 1)


os.environ['OPENCV_LOG_LEVEL'] = 'OFF'
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = "-8"
ImageFile.LOAD_TRUNCATED_IMAGES = True
import sys
import shutil

resolutions = {
    '1080p': (1920,1080),
    '720p': (1280,720),
    '480p': (854,480),
    '360p': (640,360),
}
skip_rows = 0
deg2rad = np.pi/180
rad2deg = 180/np.pi

r_e = 6378.137*1000
sl_ratio = 0.2
valid_video_types = np.array([".mp4", ".mov", ".MP4", ".MOV"])
valid_image_types = np.array([".jpg", ".jpeg", ".png"])


def scan_frames(vid_dir, mosaics_dir, frame_interval):

    all_fps = []

    # ----------------------------
    # Get FPS from all videos
    # ----------------------------
    for file_name in os.listdir(vid_dir):
        if os.path.splitext(file_name)[1].lower() in valid_video_types:

            container = av.open(os.path.join(vid_dir, file_name))
            stream = container.streams.video[0]
            if stream.average_rate is not None:
                all_fps.append(float(stream.average_rate))

            container.close()

    fps = np.mean(all_fps)

    # Initialize CSV
    columns = ["frame_number", "frame_location", "video_file", "frame_timestamp"]
    pd.DataFrame(columns=columns).to_csv(os.path.join(mosaics_dir, "frame_scan_data.csv"),index=False)

    rows = []
    last_usable_frame = 0
    currentframe = 0

    # ----------------------------
    # Scan frames
    # ----------------------------
    for file_name in np.sort(os.listdir(vid_dir)):
        frame_location = 0
        
        if os.path.splitext(file_name)[1].lower() not in valid_video_types:
            continue

        video_path = os.path.join(vid_dir, file_name)
        container = av.open(video_path)
        stream = container.streams.video[0]

        # total frames estimate
        length = stream.frames

        with alive_bar(length if length > 0 else None, title=f"Scanning frames in {file_name}") as bar:
            for frame in container.decode(stream):
                # PyAV gives timestamp in stream timebase
                if frame.pts is not None:
                    timestamp = float(frame.pts * stream.time_base) * 1000
                else:
                    timestamp = currentframe / fps * 1000

                # valid frame
                if currentframe % frame_interval == 0:
                    rows.append([currentframe, frame_location, file_name, timestamp])
                    last_usable_frame = currentframe
                currentframe += 1
                frame_location += 1
                bar()
        container.close()

    # ----------------------------
    # Save dataframe
    # ----------------------------
    frame_data = pd.DataFrame(rows, columns=columns)
    frame_data = frame_data.sort_values(by=["frame_number"])
    frame_data.to_csv(os.path.join(mosaics_dir,"frame_scan_data.csv"),index=False)


    # metadata
    frame_meta_data = {
        "time": str(datetime.datetime.now()),
        "fps": fps,
        "last_frame": last_usable_frame,
        "frame_interval": frame_interval,
    }


    with open( os.path.join(mosaics_dir, "frame_data.txt"), "w") as file:
        file.write(str(frame_meta_data))


    return last_usable_frame
    

def matching(im1, im2, yc, xc, threshold):
    Fim1= fft2(im1, threads=NUM_THREADS)
    Fim2= fft2(im2, threads=NUM_THREADS)
    cc = np.conj(Fim1)*Fim2
    pc = cc/(abs(cc) + 1e-20)

    recon_cc = np.abs(fftshift(ifft2(cc, threads=NUM_THREADS)))
    recon_pc = np.abs(fftshift(ifft2(pc, threads=NUM_THREADS)))

    py_cc, px_cc = np.unravel_index(recon_cc.argmax(), recon_cc.shape)
    py_pc, px_pc = np.unravel_index(recon_pc.argmax(), recon_pc.shape)

    if np.sqrt((py_pc - yc)**2 + (px_pc-xc)**2) <= threshold:
        py, px = py_pc, px_pc
        chosen = "pc"
    else:
        py, px = py_cc, px_cc
        chosen = "cc"

    return py, px, [py_pc, px_pc, py_cc, px_cc, chosen]


def get_imgdim(path):
    image = np.array(Image.open(path))
    return image.shape[0], image.shape[1]

def remove_bad_substrings(s):
    badSubstring = ".jpg"
    s = s.replace(badSubstring, "")
    return s


def mosaic_creation(mosaic_t, sync_vid_time, vid_dir, mosaics_dir, video_res):

    try:
        with open(os.path.join(mosaics_dir, f"frame_data.txt"), 'r') as file:
            frame_data = file.readline()
        fps = eval(frame_data)["fps"]
        max_frame = eval(frame_data)["last_frame"]
        interval = int(eval(frame_data)["frame_interval"])


        strip_width = int(int(video_res[:-1])*sl_ratio/2-1)
        sl = resolutions[video_res][0]
        closing_kernel = np.ones((5,5),np.uint8)
        yc, xc = resolutions[video_res][1]//2, resolutions[video_res][0]//2
        
    except:
        print("ERROR: Frame data could not be found. Process will abort in 60 seconds. You can close this window now.")
        time.sleep(60)
        sys.exit()
        

    starting_image = int(sync_vid_time*fps)
    
    upper_threshold = 20*sl
    stitching_threshold = 20*sl
    mosaic_counter = 0
    current_filename = None
    frame_count = 0
    frame_data = pd.read_csv(os.path.join(mosaics_dir, "frame_scan_data.csv"))
    video_files = frame_data["video_file"].unique()


    for current_filename in video_files:
        mosaic_name = Path(current_filename).stem

        video_data = frame_data[frame_data["video_file"] == current_filename]
        video_data = video_data.sort_values(by=["frame_number"])
        idset = video_data[video_data["frame_number"] >= starting_image]["frame_number"].values
        if len(idset) == 0:
            print(f"No frames found in {current_filename} after the specified sync time. Skipping video...")
            continue

        container = av.open(os.path.join(vid_dir, current_filename))
        stream = container.streams.video[0]
        frame_count = 0
        frame_iterator = container.decode(stream)
        length = stream.frames

        with alive_bar(int(length), title = f"Frames processed from {current_filename}:") as bar:
            left_border = 0
            right_border = sl

            for img_counter in range(len(idset)):
                current_frame = frame_data.loc[frame_data.frame_number == idset[img_counter], "frame_location"].values[0]

                 # Move forward until desired frame
                while frame_count < current_frame-1:
                    try:
                        av_frame = next(frame_iterator)
                        frame_count += 1
                        bar()
                    except StopIteration:
                        break
                
                # Decode target frame
                try:
                    av_frame = next(frame_iterator)
                    frame_count += 1
                    bar()
                    frame = av_frame.to_ndarray(format="rgb24")
                    frame = cv2.resize(frame, dsize=resolutions[video_res], interpolation=cv2.INTER_AREA)
                except StopIteration:
                    continue
            
                if img_counter == 0:
                    img = frame
                    strip1 = img[yc - strip_width : yc + strip_width + 1]
                    mosaic = strip1

                    # Padding
                    current_x = int(sl*1.5)
                    current_y = int(0.25*strip_width*len(idset))*interval
                    mosaic = cv2.copyMakeBorder(mosaic, current_y, 0, current_x, current_x, cv2.BORDER_CONSTANT)
                    left_border = 0
                    right_border = mosaic.shape[1]
                    
                
                elif img_counter < len(idset)-1:
                    strip2 = frame[yc - strip_width : yc + strip_width + 1]
                    try:
                        py, px, reg_stats  = matching(np.ascontiguousarray(strip2[...,1]), np.ascontiguousarray(strip1[...,1]), strip_width, xc, upper_threshold)
                    except:
                        print("Matching failed. Skipping frame...")
                        continue
                    if np.sqrt((py - strip_width)**2 + (px-xc)**2) > stitching_threshold:
                        continue

                    y_offset = py - strip_width
                    if y_offset <= 0:
                        x_offset = px - xc
                        current_x += x_offset 
                        current_y += y_offset
                        mstrip = strip2

                        if idset[img_counter + 1] is None:
                            continue

                        if current_y < 0:
                            mosaic = cv2.copyMakeBorder(mosaic, abs(current_y), 0, 0, 0, cv2.BORDER_CONSTANT)
                            current_y = 0

                        if current_x < left_border:
                            mosaic = cv2.copyMakeBorder(mosaic,0, 0, -(current_x - left_border), 0, cv2.BORDER_CONSTANT) 
                            left_border = current_x
                        elif current_x + sl > right_border:
                            mosaic = cv2.copyMakeBorder(mosaic,0, 0, 0, current_x + sl - right_border, cv2.BORDER_CONSTANT) 
                            right_border = current_x + sl 
                        
                        mosaic[current_y:current_y + strip_width*2+1,current_x - left_border : sl + current_x - left_border] = mstrip

                        strip1 = strip2
            try:
                non_black_rows = np.any(mosaic != [0, 0, 0], axis=(1, 2))
                non_black_columns = np.any(mosaic != [0, 0, 0], axis=(0, 2))
                mosaic = mosaic[non_black_rows, :]
                mosaic = mosaic[:, non_black_columns]
                mask = np.ones((mosaic.shape[0:2]))*255
                black_region = mosaic == [0,0,0]
                mask[black_region[...,0]] = 0
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_kernel)
                mask = mask.astype(np.uint8)
                mosaic = Image.fromarray(mosaic.astype(np.uint8)).convert('RGB')
                mask = Image.fromarray(mask.astype(np.uint8)).convert('L')
                mosaic.putalpha(mask)
                mosaic.save(os.path.join(mosaics_dir, f"{mosaic_name}.png"), "PNG")
            except:
                print(f"ERROR: Mosaic {mosaic_name} could not be created. Skipping...")
                continue
            del(mosaic)
            del(mask)
            gc.collect()

            mosaic_counter+=1
            
        mosaic_meta_data = {
            "mosaic_time" : mosaic_t,
            "time": str(datetime.datetime.now()),
            "sync_vid_time": sync_vid_time,
            "num_mosaics": mosaic_counter,
            "vid_len": resolutions[video_res][0],
        }
        with open(os.path.join(mosaics_dir,f"mosaic_data_{mosaic_name}.txt"), 'w') as file:
            file.write(str(mosaic_meta_data))
def random_points_nonblack(image, N, radius, max_attempts=1000000):

    if image.ndim == 3:
        mask = np.any(image != 0, axis=2)
    else:
        mask = image != 0

    candidates = np.argwhere(mask)  

    if len(candidates) < N:
        raise ValueError("Not enough valid pixels")

    selected = []

    tree = None

    attempts = 0

    while len(selected) < N and attempts < max_attempts:

        idx = np.random.randint(len(candidates))
        point = candidates[idx]

        if len(selected) == 0:
            selected.append(point)
            tree = cKDTree(np.array(selected))
            continue

        dist, _ = tree.query(point, k=1)

        if dist >= radius:
            selected.append(point)
            tree = cKDTree(np.array(selected))

        attempts += 1

    if len(selected) < N:
        print(
            f"Only found {len(selected)} points. "
            "Try reducing radius."
        )

    return np.array(selected)

def mark_mosaics(num_markings, mark_dir, mosaics_dir):
    frame_data = pd.read_csv(os.path.join(mosaics_dir, "frame_scan_data.csv"))
    video_files = frame_data["video_file"].unique()
    mosaics = [f for f in os.listdir(mosaics_dir) if f.endswith('.png') and os.path.isfile(os.path.join(mosaics_dir, f))]

    with alive_bar(len(mosaics), title=f"Mosaics marked") as bar:
        for mosaic in mosaics:

            if len(os.listdir(mosaics_dir)) == 0:
                print("No mosaics found. Aborting process...")
                return
            
            with open(os.path.join(mosaics_dir, f"mosaic_data_{Path(mosaic).stem}.txt"), 'r') as file:
                frame_data = file.readline()
            vid_len = int(eval(frame_data)["vid_len"])

            radius = 0.05*vid_len
            chosen_marks = []
            file = os.path.join(mosaics_dir, mosaic)
            img = np.array(cv2.imread(file, cv2.IMREAD_UNCHANGED))
            vert = img.shape[0]
            marking_box = int(0.0002*vert)
            thickness = 1 

            chosen_marks = random_points_nonblack(img, num_markings, radius)
            chosen_marks = chosen_marks[np.lexsort((chosen_marks[:,1], chosen_marks[:,0]))]

            if thickness == 0:
                thickness = 1
            if marking_box < 5:
                marking_box = 5
            

            offset = marking_box + 5

            img_height, img_width = img.shape[:2]
            center_y, center_x = img_height // 2, img_width // 2
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            text_color = (0, 255, 255)
            
            for mark_num, mark in enumerate(chosen_marks):
                centroid_y = mark[0]
                centroid_x = mark[1]

                img = cv2.line(img, (centroid_x - marking_box, centroid_y- marking_box), (centroid_x + marking_box, centroid_y+ marking_box), (0, 0, 255), thickness)
                img = cv2.line(img, (centroid_x - marking_box, centroid_y+ marking_box), (centroid_x + marking_box, centroid_y- marking_box), (0, 0, 255), thickness)
                
                # Determine quadrant and text position
                is_left = centroid_x < center_x
                is_top = centroid_y < center_y
                
                if is_top and is_left:  # Upper left quadrant: number on lower right
                    text_x = centroid_x + offset
                    text_y = centroid_y + offset
                elif is_top and not is_left:  # Upper right quadrant: number on lower left
                    text_x = centroid_x - offset + 2
                    text_y = centroid_y + offset
                elif not is_top and is_left:  # Lower left quadrant: number on upper right
                    text_x = centroid_x + offset
                    text_y = centroid_y - offset + 2
                else:  # Lower right quadrant: number on upper left
                    text_x = centroid_x - offset + 2 
                    text_y = centroid_y - offset + 2
                
                cv2.putText(img, str(mark_num + 1), (text_x, text_y), font, font_scale, text_color, font_thickness)
        
            cv2.imwrite(os.path.join(mark_dir, f"marked_{Path(mosaic).stem}.jpg"), img)
            bar()