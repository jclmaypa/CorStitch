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
# from tqdm import tqdm
import scipy as sp
import datetime
from gpxcsv import gpxtolist
import gc
from alive_progress import alive_bar
from pyfftw.interfaces.numpy_fft import fft2, ifft2, fftshift
import pyfftw
import simplekml
import imutils
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
pyfftw.interfaces.cache.enable()
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
sl_ratio = 0.20
valid_video_types = np.array([".mp4", ".mov", ".MP4", ".MOV"])

kmz_limit = 100
img_counter = 0
kmz_counter = 0

angles_inc = 0.01
angles = (np.pi/180)*np.arange(-180, 180 + angles_inc, angles_inc)
SW = np.arange(270, 180, -angles_inc)
SE = np.arange(180, 90, -angles_inc)
NE = np.arange(90, 0, -angles_inc)
NW = np.arange(360, 270-angles_inc, -angles_inc)
NW[0] = 0 

bearing_values = np.concatenate((SW,SE,NE,NW), axis = 0)

bearing_conv = sp.interpolate.interp1d(angles, bearing_values,kind = 'linear', fill_value = 'extrapolate', bounds_error = False)

conventional_column_names = ["date_time", "lat", "lon", "dep_m", "instr_heading"]

def HMS2Conv(x):
    h,m,s = map(float,x.split(':'))
    return ((h)*60+m)*60+s

def extract_raw_data(file_path, skiprows=0):
    file_type = file_path[-3:]

    if file_type == "csv":
        for i in range(1000):
            try:
                data_file = pd.read_csv(file_path, skiprows = i)
                data_file = data_file.loc[:, ~data_file.columns.str.contains('^Unnamed')]
                
                for j in range(100):
                    row = data_file.iloc[j].isna()
                    empty = all(element == True for element in row)
                    if empty == True:
                        break

                if empty == False and len(data_file.columns) >=3:
                    break
            except:
                continue


    elif file_type == "gpx":
        data_file = pd.DataFrame(gpxtolist(file_path))
        
    else:
        print("CorStitch does not support that file type.")
    return data_file


class GPSdata():
    def __init__(self, file_name):
        self.raw_data = extract_raw_data(file_name)
        self.raw_data.dropna(how='all', axis=1, inplace=True)
        self.columns = self.raw_data.columns
        self.gps_data = None
        self.heading_status = 1
        self.depth_status = 1

    def read_gps_data(self, usable_columns):
        usable_columns = np.array(usable_columns)
        if "NA" in usable_columns[0:3]:
            print("The time, longitude, and lattitude data are essential for georeferencing. Without these data, CorStitch cannot georeference your data.\n Cancelling georeferencing process...")
            time.sleep(15)
            sys.exit()
        if usable_columns[3] == "NA":
            self.depth_status = 0
        if usable_columns[4] == "NA":
            self.heading_status = 0

        self.gps_data = self.raw_data.loc[:, usable_columns[usable_columns != "NA"]]
        renames =  dict(zip(usable_columns ,conventional_column_names))
        self.gps_data.rename(columns = renames, inplace = True)
        if self.depth_status == 1:
            self.gps_data.loc[0:self.gps_data.dep_m.first_valid_index(), "dep_m"] = self.gps_data.dep_m[self.gps_data.dep_m.first_valid_index()]
            self.gps_data.loc[self.gps_data.dep_m.last_valid_index(): self.gps_data.index[-1], "dep_m"] = self.gps_data.dep_m[self.gps_data.dep_m.last_valid_index()]
            self.gps_data["dep_m"]= self.gps_data["dep_m"].interpolate(method = "linear")
        self.gps_data.dropna(how = "any", inplace = True, ignore_index = True)

        if self.heading_status == 1:
            self.gps_data["instr_heading"]= self.gps_data["instr_heading"].interpolate(method = "linear")
        if self.heading_status == 0:
            headings = []

            for i in range(len(self.gps_data) - 1):
                lon1 = self.gps_data.lon[i] * deg2rad
                lon2 = self.gps_data.lon[i+1] * deg2rad
                lat1 = self.gps_data.lat[i] * deg2rad
                lat2 = self.gps_data.lat[i+1] * deg2rad

                dlon = lon2 - lon1

                y = np.sin(dlon) * np.cos(lat2)
                x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

                brng = np.arctan2(y, x)  # radians, from North, clockwise
                heading = (brng * rad2deg + 360) % 360

                headings.append(heading)

            headings = np.array(headings)

            # pad first/last to keep same length
            headings = np.insert(headings, 0, headings[0])
            self.gps_data["instr_heading"] = headings

        return self.depth_status
    
    def date_time_split(self, chosen_date = 0, splitting = True):
        if splitting == True:
            self.gps_data.loc[:, 'date'] = self.gps_data.date_time.replace(r"[T].+$", "", regex=True)
            self.gps_data.loc[:, 'time'] = self.gps_data.date_time.str.extract(r'.*T(.*)Z')
        else:
            self.gps_data.rename(columns = {"date_time":"time"}, inplace=True)

    def convert_time(self):
        self.gps_data["conv_time"] = self.gps_data.time.apply(HMS2Conv)
        # add synch time
    def export(self):
        return self.gps_data
        # add synch time


def scan_frames(vid_dir, mosaics_dir, frame_interval):
    currentframe = 0 

    all_fps = []
    for file_name in os.listdir(vid_dir):
        if np.any([filetype in file_name.lower() for filetype in valid_video_types]):
            data = cv2.VideoCapture(os.path.join(vid_dir, f"{file_name}"))
            fps = data.get(cv2.CAP_PROP_FPS) 
            all_fps.append(fps)
    fps = np.mean(all_fps)
    
    if currentframe == 0:
       frame_data = pd.DataFrame(columns = ["frame_number", "frame_location", "video_file", "frame_timestamp"])
       frame_data.to_csv(os.path.join(mosaics_dir, "frame_scan_data.csv"))
    
    if currentframe != 0:
        try:
            frame_data = pd.read_csv(os.path.join(mosaics_dir, "frame_scan_data.csv"))
        except:
            print("ERROR: Could not locate previous frame scan data. Aborting process...")

     
    
    rows = []
    for file_name in np.sort(os.listdir(vid_dir)):
        if os.path.splitext(file_name)[1].lower() in valid_video_types:
            frame_location = 0
            cam = cv2.VideoCapture(os.path.join(vid_dir, file_name))
            property_id = int(cv2.CAP_PROP_FRAME_COUNT)  
            length = int(cv2.VideoCapture.get(cam, property_id)) 
            with alive_bar(length, title=f"Scanning frames in {file_name}") as bar:
                for i in range(length):
                    ret, frame = cam.read()

                    # Retry until frame is valid (with safety break)
                    retry_count = 0
                    while not ret and retry_count < 10:
                        ret, frame = cam.read()
                        frame_location += 1
                        retry_count += 1

                    if not ret:
                        # Skip this frame if still invalid after retries
                        currentframe += 1
                        frame_location += 1
                        bar()
                        continue

                    timestamp_ms = cam.get(cv2.CAP_PROP_POS_MSEC)

                    if currentframe % frame_interval == 0:
                        rows.append([
                            currentframe,
                            frame_location,
                            file_name,
                            float(timestamp_ms)
                        ])

                    currentframe += 1
                    frame_location += 1
                    bar()

    frame_data = pd.DataFrame(rows, columns=frame_data.columns)
    frame_data = frame_data.sort_values(by=["frame_number"])
    frame_data.to_csv(os.path.join(mosaics_dir, "frame_scan_data.csv"), index = False)

    cam.release() 
    cv2.destroyAllWindows()
    frame_meta_data = {
        "time": str(datetime.datetime.now()),
        "fps": fps,
        "last_frame": currentframe,
        "frame_interval": frame_interval,
    }
    with open(os.path.join(mosaics_dir, "frame_data.txt"), 'w') as file:
        file.write(str(frame_meta_data))
    return currentframe
    

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
        closing_kernel = np.ones((15,15),np.uint8)
        yc, xc = resolutions[video_res][1]//2, resolutions[video_res][0]//2
        
    except:
        print("ERROR: Frame data could not be found. Process will abort in 60 seconds. You can close this window now.")
        time.sleep(60)
        sys.exit()
        

    starting_image = int(sync_vid_time*fps)

    frame_data = pd.read_csv(os.path.join(mosaics_dir, "frame_scan_data.csv"))
    img_ids = frame_data["frame_number"].to_list()
    mosaic_boundaries = np.arange(starting_image, max_frame, int(round(fps*mosaic_t)))
    if mosaic_boundaries[-1] < max_frame:
        mosaic_boundaries = np.append(mosaic_boundaries, max_frame)

    upper_threshold = 0.2*sl
    stitching_threshold = 0.5*sl
    mosaic_counter = 0
    current_filename = None
    frame_count = 0
    end_time = 0
    accumulated_time = 0
    mosaic_time_boundaries = pd.DataFrame(columns = ["mosaic_number", "start_time_s", "end_time_s"])


    with alive_bar(len(mosaic_boundaries)-1, title = f"Creating mosaics...") as bar:
        for i in range(len(mosaic_boundaries)-1):
            idset = [x for x in img_ids if x >= mosaic_boundaries[i] and x < mosaic_boundaries[i+1]]
            left_border = 0
            right_border = sl

            for img_counter in range(len(idset)):
                file_name = frame_data.loc[frame_data.frame_number == idset[img_counter], "video_file"].values[0]
                current_frame = frame_data.loc[frame_data.frame_number == idset[img_counter], "frame_location"].values[0]

                if file_name != current_filename:
                    current_filename = file_name
                    try:
                        cam.release()
                        cv2.destroyAllWindows()
                        gc.collect()
                    except:
                        pass
                    print(f"Processing mosaics from {current_filename}...")
                    cam = cv2.VideoCapture(os.path.join(vid_dir, current_filename))
                    property_id = int(cv2.CAP_PROP_FRAME_COUNT)
                    length = int(cam.get(property_id))
                    frame_count = 0
                    accumulated_time = end_time

                if frame_count >= length:
                    break

                # Seek to the target frame by skipping preceding frames
                while frame_count < current_frame and frame_count < length:
                    ret, frame = cam.read()
                    frame_count += 1
                    if not ret:
                        if frame_count >= length:
                            break

                # Read the actual target frame
                if frame_count == current_frame and frame_count < length:
                    ret, frame = cam.read()
                    frame_count += 1
                    if ret:
                        frame = cv2.resize(frame, dsize = resolutions[video_res], interpolation = cv2.INTER_AREA)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
                        if img_counter == 0:
                            start_time = float(frame_data.loc[frame_data.frame_number == idset[img_counter], "frame_timestamp"].values[0]) + accumulated_time
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
                end_time = float(frame_data.loc[frame_data.frame_number == idset[img_counter], "frame_timestamp"].values[0]) + accumulated_time
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
                mosaic.save(os.path.join(mosaics_dir, f"{mosaic_counter}.png"), "PNG")
            except:
                print(f"ERROR: Mosaic {mosaic_counter} could not be created. Skipping...")
                continue
            del(mosaic)
            del(mask)
            gc.collect()
            mosaic_time_boundaries.loc[len(mosaic_time_boundaries)] = [mosaic_counter, start_time/1000, end_time/1000]

            mosaic_counter+=1
            bar()
    mosaic_time_boundaries.to_csv(os.path.join(mosaics_dir, "mosaic_time_boundaries.csv"), index = False)
    mosaic_meta_data = {
        "mosaic_time" : mosaic_t,
        "time": str(datetime.datetime.now()),
        "sync_vid_time": sync_vid_time,
        "num_mosaics": mosaic_counter
    }
    with open(os.path.join(mosaics_dir,f"mosaics_data.txt"), 'w') as file:
        file.write(str(mosaic_meta_data))



def georeference(gps_data, data, vid_dir, mosaics_dir, kmz_dir, rect_mosaics_dir):
    utc_offset = int(data["utc_offset"])
    sync_UTC_time = str(data["sync_time"])
    sync_UTC_time = datetime.datetime.strptime(sync_UTC_time,'%H:%M:%S')
    sync_UTC_time = sync_UTC_time + datetime.timedelta(hours = - utc_offset)
    sync_UTC_time = HMS2Conv(sync_UTC_time.strftime("%H:%M:%S"))
    date = data["date_picker"]

    unique_dates = data["unique_dates"]
    depth_status = int(data["depth_status"])
    if len(unique_dates) > 1:
        gps_data = gps_data[gps_data.date == date].reset_index(drop=True)

    gps_data["conv_time"] = gps_data.time.apply(HMS2Conv)
    print(gps_data.conv_time.min(), gps_data.conv_time.max(), sync_UTC_time)
    print(gps_data.conv_time)

    if len(gps_data[gps_data.conv_time <= sync_UTC_time].index) <= 0:
        print("WARNING: No data points were removed during GPS and Video synchronization. This could mean that GPS data collection started after the synchronization time.")

    lon_interp = sp.interpolate.interp1d(gps_data.conv_time, gps_data.lon, kind='linear', fill_value='extrapolate', bounds_error=False)
    lat_interp = sp.interpolate.interp1d(gps_data.conv_time, gps_data.lat, kind='linear', fill_value='extrapolate', bounds_error=False)
    heading_interp = sp.interpolate.interp1d(gps_data.conv_time, gps_data.instr_heading, kind='linear', fill_value='extrapolate', bounds_error=False)

    if depth_status == 1:
        depth_interp = sp.interpolate.interp1d(gps_data.conv_time, gps_data.dep_m, kind='linear', fill_value='extrapolate', bounds_error=False)
        ave_depth = np.mean(gps_data.dep_m)

    print("Georeferencing images...")
    with open(os.path.join(mosaics_dir, "mosaics_data.txt"), 'r') as file:
        mosaic_data = file.readline()
    mosaic_data = eval(mosaic_data)
    num_mosaics = mosaic_data["num_mosaics"]
    if num_mosaics == 0:
        print("No mosaics detected for this project. No images can be georeferenced")
    else:
        
        kmz_limit = 100
        img_counter = 0
        kmz_counter = 0
        kml = simplekml.Kml()
        depth = 0
        if depth_status == 0:
            width_m = 5
        mosaic_boundaries = pd.read_csv(os.path.join(mosaics_dir, "mosaic_time_boundaries.csv"))
        start_times = mosaic_boundaries['start_time_s'].tolist()
        end_times = mosaic_boundaries['end_time_s'].tolist()
        time_floor = start_times[0]
        time_sync = sync_UTC_time - time_floor
        with alive_bar(len(mosaic_boundaries), title = f"Georeferencing mosaics...") as bar:
            for i in range(len(mosaic_boundaries)):
                    
                start = start_times[i] + time_sync
                end = end_times[i] + time_sync
                mid = start + (end-start)//2

                mosaic_name = int(i)
                icon_path = os.path.join(mosaics_dir, f"{mosaic_name}.png")
                lpx, wpx = get_imgdim(icon_path)
                headings = heading_interp(start)
                try:
                    headinge = heading_interp(end)
                except:
                    print("Could not georeference all mosaics due to the lack of GPS data.")
                    break
                heading = heading_interp(mid)

                lat_s = lat_interp(start)
                lon_s = lon_interp(start)
                lat_e = lat_interp(end)
                lon_e = lon_interp(end)
                lat_m = lat_interp(mid)
                lon_m = lon_interp(mid)

                if depth_status == 1:
                    depth = np.mean(depth_interp(np.linspace(start, end, 1000)))
                    width_m = 1.55948 * ave_depth
                    px2m = width_m / wpx
                    scalebar = ScaleBar(dx=px2m,
                                        units='m',
                                        fixed_value=1,
                                        fixed_units='m',
                                        location="lower left",
                                        font_properties={'family': 'monospace',
                                                        'weight': 'semibold',
                                                        'size': 20})
                    
                img = np.array(Image.open(os.path.join(mosaics_dir, f"{mosaic_name}.png")))[:, :, 0:3]
                img = imutils.rotate_bound(img, angle=heading)
                non_black_rows = np.any(img != [0, 0, 0], axis=(1, 2))
                non_black_columns = np.any(img != [0, 0, 0], axis=(0, 2))
                img = img[non_black_rows, :]
                img = img[:, non_black_columns]
                rlpx, rwpx = img.shape[0], img.shape[1]
                fig, ax = plt.subplots(figsize=(20, 20))
                img_desc = '\n'.join((
                    r'mosaic no. %.0f' % (mosaic_name),
                    r'date: %s' % (date),
                    r'lat.: %.8f°' % (lat_m),
                    r'lon.: %.8f°' % (lon_m),
                    r'bearing: %.2f°' % (heading),
                    r'ave. depth: %.3f' % (depth),
                ))

                ax.imshow(img)
                ax.set_title(img_desc, loc="left", fontsize=30)
                plt.gca().set_aspect('equal', adjustable='box')
                if depth_status == 1:
                    plt.gca().add_artist(scalebar)
                ax.set_axis_off()

                fig.savefig(os.path.join(rect_mosaics_dir, f"{mosaic_name}.jpg"), bbox_inches='tight')
                plt.close('all')
                gc.collect()

                point = kml.newpoint(name=f"{mosaic_name}", coords=[(lon_m, lat_m)])
                picpath = kml.addfile(os.path.join(rect_mosaics_dir, f"{mosaic_name}.jpg"))
                img_desc = f'<img src="{picpath}" alt="picture" width="{rwpx}" height="{rlpx}" align="left" />'
                point.style.balloonstyle.text = img_desc

                ground = kml.newgroundoverlay(name=f"{mosaic_name}.png")
                ground.icon.href = icon_path
                ground.description = f"{mosaic_name}.png\nheading: {heading}"

                # if (heading >= 270 and heading <= 360) or (heading >= 0 and heading <= 90):
                # radians
                perp_s = (headings + 90.0) * deg2rad
                perp_e = (headinge + 90.0) * deg2rad

                dxs = 0.5 * width_m * np.sin(perp_s)   
                dys = 0.5 * width_m * np.cos(perp_s)   
                dx2s = -dxs
                dy2s = -dys

                dxe = 0.5 * width_m * np.sin(perp_e)
                dye = 0.5 * width_m * np.cos(perp_e)
                dx2e = -dxe
                dy2e = -dye

                # meters → lat/lon
                tr_lon = lon_s + (dxs / (r_e * np.cos(lat_s * deg2rad))) * rad2deg
                tr_lat = lat_s + (dys / r_e) * rad2deg

                tl_lon = lon_s + (dx2s / (r_e * np.cos(lat_s * deg2rad))) * rad2deg
                tl_lat = lat_s + (dy2s / r_e) * rad2deg

                br_lon = lon_e + (dxe / (r_e * np.cos(lat_e * deg2rad))) * rad2deg
                br_lat = lat_e + (dye / r_e) * rad2deg

                bl_lon = lon_e + (dx2e / (r_e * np.cos(lat_e * deg2rad))) * rad2deg
                bl_lat = lat_e + (dy2e / r_e) * rad2deg

                ground.gxlatlonquad.coords = [
                    (tl_lon, tl_lat),
                    (tr_lon, tr_lat),
                    (br_lon, br_lat),
                    (bl_lon, bl_lat)
                ]


                img_counter += 1
                bar()
                if img_counter % kmz_limit == 0:
                    kml.savekmz(os.path.join(kmz_dir, f"{kmz_counter}.kmz"))
                    kml = simplekml.Kml()
                    kmz_counter += 1
            if img_counter % kmz_limit != 0:
                kml.savekmz(os.path.join(kmz_dir, f"{kmz_counter}.kmz"))
