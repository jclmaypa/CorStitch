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



os.environ['OPENCV_LOG_LEVEL'] = 'OFF'
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = "-8"
ImageFile.LOAD_TRUNCATED_IMAGES = True
import sys
import shutil

resolutions = {
    '1080p': (1920, 1080),
    '720p': (1280, 720),
    '480p': (854,480),
    '360p': (640, 360),
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
            for i in range(1,len(self.gps_data.loc[:"date_time"])-1,1):
                lon1 = deg2rad*self.gps_data.lon[i]
                lon2 = deg2rad*self.gps_data.lon[i+1]

                lat1 = deg2rad*self.gps_data.lat[i]
                lat2 = deg2rad*self.gps_data.lat[i+1]

                brng = np.arctan2(lat2 - lat1, lon2-lon1 + 1e-20)
                heading = bearing_conv(brng)

                    
                headings.append(heading)
            headings = np.array(headings)
            headings = np.insert(headings, [0, len(headings)-1], [headings[0], headings[len(headings)-1]] )
            self.gps_data["instr_heading"] = headings
        return self.depth_status
    
    def date_time_split(self, chosen_date = 0, splitting = True):
        if splitting == True:
            self.gps_data.loc[:, 'date'] = self.gps_data.date_time.replace(r"[T].+$", "", regex=True)
            self.gps_data.loc[:, 'time'] = self.gps_data.date_time.str.extract(r'.*T(.*)Z')
        

            # if len(unique_dates) > 1:
            #     print("Multiple dates are detected, which one will you use?")
            #     for i in range(len(unique_dates)):
            #         print(f"[{i+1}] {unique_dates[i]}")
            #     chosen_date = int(input("Choose the corresponding number of the correct date: "))
            #     if chosen_date <=0 or chosen_date > len(unique_dates):
            #         print("ERROR: Invalid option")
            #         time.sleep(15)
            #         sys.exit()
            #     self.gps_data = self.gps_data.drop(self.gps_data[self.gps_data.date != unique_dates[chosen_date-1]].index)
        else:
            self.gps_data.rename(columns = {"date_time":"time"}, inplace=True)

    def convert_time(self):
        self.gps_data["conv_time"] = self.gps_data.time.apply(HMS2Conv)
        # add synch time
    def export(self):
        return self.gps_data
        # add synch time

def vid2frames(filename, vid_dir, frames_dir, frame_interval, reduce = False, res = "480p", image_counter = 0):  

    cam = cv2.VideoCapture(os.path.join(vid_dir, filename))
    currentframe = 0 + image_counter

    all_fps = []
    for file_name in os.listdir(vid_dir):
        if np.any([filetype in file_name.lower() for filetype in valid_video_types]):
            data = cv2.VideoCapture(os.path.join(vid_dir, f"{filename}"))
            fps = data.get(cv2.CAP_PROP_FPS) 
            all_fps.append(fps)
    fps = np.mean(all_fps)
    
    if currentframe == 0:
        print("Clearing previous frames...")
        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir)
        os.makedirs(frames_dir, exist_ok=True)

    property_id = int(cv2.CAP_PROP_FRAME_COUNT)  
    length = int(cv2.VideoCapture.get(cam, property_id))  
    with alive_bar(length, title = f"Extracting frames from {filename}") as bar:
        for i in range(length):
            ret,frame = cam.read() 

        
            while ret == False:
                ret,frame = cam.read() 
                
            name = os.path.join(frames_dir, str(currentframe) + '.jpg')
            if currentframe % frame_interval == 0: 
                if reduce == True:
                    frame = cv2.resize(frame, dsize = resolutions[res], interpolation = cv2.INTER_AREA)
                cv2.imwrite(name, frame) 
            currentframe += 1
            bar()
    cam.release() 
    cv2.destroyAllWindows()
    frame_meta_data = {
        "res" : res,
        "time": str(datetime.datetime.now()),
        "fps": fps,
        "last_frame": currentframe,
    }
    with open(os.path.join(frames_dir, "frame_data.txt"), 'w') as file:
        file.write(str(frame_meta_data))
    return currentframe


def matching(im1, im2, yc, xc, threshold):
    Fim1= np.fft.fft2(im1)
    Fim2= np.fft.fft2(im2)
    cc = np.conj(Fim1)*Fim2
    pc = cc/abs(cc)

    recon_cc = abs(np.fft.fftshift(np.fft.ifft2(cc)))
    recon_pc = abs(np.fft.fftshift(np.fft.ifft2(pc)))

    py_cc, px_cc = np.unravel_index(recon_cc.argmax(), recon_cc.shape)
    py_pc, px_pc = np.unravel_index(recon_pc.argmax(), recon_pc.shape)

    if np.sqrt((py_pc - yc)**2 + (px_pc-xc)**2) <= threshold:
        py, px = py_pc, px_pc
        chosen = "pc"
    else:
        py, px = py_cc, px_cc
        chosen = "cc"

    return py, px, [py_pc, px_pc, py_cc, px_cc, chosen]

def read_images(ids, frames_dir, channel = 1):
    images = []
    frames = 1

    for id in ids:
        if id == None:
            continue
        with Image.open(os.path.join(frames_dir, f"{id}.jpg")) as Img:
            img = np.array(Img)
        if img is not None:
            images.append(img[...,channel])

    return images

def get_imgdim(path):
    image = np.array(Image.open(path))
    return image.shape[0], image.shape[1]

def remove_bad_substrings(s):
    badSubstring = ".jpg"
    s = s.replace(badSubstring, "")
    return s


def mosaic_creation(mosaic_t, sync_vid_time, frames_dir, mosaics_dir, interval = 1):

    try:
        with open(os.path.join(frames_dir, f"frame_data.txt"), 'r') as file:
            frame_data = file.readline()
        video_res = eval(frame_data)["res"]
        fps = eval(frame_data)["fps"]
        max_frame = eval(frame_data)["last_frame"]
        strip_width = int(int(video_res[:-1])*sl_ratio/2-1)
        sl = resolutions[video_res][0]
        closing_kernel = np.ones((15,15),np.uint8)
        
    except:
        print("ERROR: Frame data could not be found. Process will abort in 60 seconds. You can close this window now.")
        time.sleep(60)
        sys.exit()
        

    starting_image = int(sync_vid_time*fps)
    # num_images = len(os.listdir(frames_dir))-1

    # img_ids = np.arange(0,num_images, interval, dtype = np.int32)
    img_ids = os.listdir(frames_dir)
    img_ids.remove("frame_data.txt")
    img_ids = [int(remove_bad_substrings(s)) for s in img_ids]
    img_ids.sort()
    mosaic_boundaries = np.arange(starting_image, max_frame, int(round(fps*mosaic_t)))
    if mosaic_boundaries[-1] < max_frame:
        mosaic_boundaries = np.append(mosaic_boundaries, max_frame)

    upper_threshold = 0.2*sl
    stitching_threshold = 0.5*sl
    mosaic_counter = 0
    stats = {"pcy":[], "pcx":[],"ccy":[], "ccx":[], "mosaic_number":[], "chosen" : []}
    
    with alive_bar(len(mosaic_boundaries)-1, title = f"Creating mosaics...") as bar:
        for i in range(len(mosaic_boundaries)-1):
            idset = [x for x in img_ids if x >= mosaic_boundaries[i] and x < mosaic_boundaries[i+1]]
            # idset = img_ids[mosaic_boundaries[i]:mosaic_boundaries[i+1]]
            img_counter = 0
            current_x = 0
            left_border = 0
            right_border = sl
            images = []
            for id in idset:
                if id is None:
                    continue
                img_path = os.path.join(frames_dir, f"{id}.jpg")
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)  #
                if img is not None:
                    images.append(img)
            for img in images:
                if img_counter == 0:
                    if idset[img_counter] is None:
                        continue
                    yc, xc = img.shape[0]//2, img.shape[1]//2
                    strip1 = img[yc - strip_width : yc + strip_width + 1] 
                    mstrip = np.array(Image.open(os.path.join(frames_dir, f"{idset[img_counter]}.jpg" )))
                    mstrip = mstrip[yc - strip_width : yc + strip_width + 1] 
                    mosaic = mstrip.copy()
                    # mosaic = cv2.copyMakeBorder(mosaic, 0, strip_width*2, 0, 0, cv2.BORDER_CONSTANT) 
                elif img_counter < len(images)-1:
                    strip2 = images[img_counter+1][yc - strip_width : yc + strip_width + 1] 
                    py, px, reg_stats  = matching(strip2[...,1], strip1[...,1], strip_width, xc, upper_threshold)
                    # stats["pcy"].append(reg_stats[0] - strip_width)
                    # stats["pcx"].append(reg_stats[1] - xc)
                    # stats["ccy"].append(reg_stats[2] - strip_width)
                    # stats["ccx"].append(reg_stats[3] - xc)
                    # stats["mosaic_number"].append(mosaic_counter)
                    # stats["chosen"].append(reg_stats[4] )


                    if np.sqrt((py - strip_width)**2 + (px-xc)**2) > stitching_threshold:
                        continue

                    y_offset = py - strip_width
                    if y_offset <= 0:
                        x_offset = px - xc
                        current_x += x_offset 

                        if idset[img_counter + 1] is None:
                            continue
                        
                        mosaic = cv2.copyMakeBorder(mosaic, abs(y_offset), 0, 0, 0, cv2.BORDER_CONSTANT)
                        mstrip = np.array(Image.open(os.path.join(frames_dir,f"{idset[img_counter + 1]}.jpg")))
                        mstrip = mstrip[yc - strip_width : yc + strip_width + 1] 
                        if current_x < left_border:
                            mosaic = cv2.copyMakeBorder(mosaic,0, 0, -(current_x - left_border), 0, cv2.BORDER_CONSTANT) 
                            left_border = current_x
                        elif current_x + sl > right_border:
                            mosaic = cv2.copyMakeBorder(mosaic,0, 0, 0, current_x + sl - right_border, cv2.BORDER_CONSTANT) 
                            right_border = current_x + sl 
                        
                        mosaic[0:strip_width*2+1,current_x - left_border : sl + current_x - left_border] = mstrip

                        strip1 = strip2
                    
                
                img_counter +=1
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
                # mask = mask[..., np.newaxis] 
                # mosaic = np.concatenate((mosaic, mask), axis=2, dtype= np.uint8)
                # mosaic = cv2.cvtColor(mosaic, cv2.COLOR_BGR2BGRA)
                # cv2.imwrite(os.path.join(mosaics_dir, f"{mosaic_counter}.png"), mosaic)

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
            mosaic_counter+=1
            bar()
    mosaic_meta_data = {
        "mosaic_time" : mosaic_t,
        "time": str(datetime.datetime.now()),
        "sync_vid_time": sync_vid_time,
        "num_mosaics": len(os.listdir(mosaics_dir)) - 1
    }
    with open(os.path.join(mosaics_dir,f"mosaics_data.txt"), 'w') as file:
        file.write(str(mosaic_meta_data))
    # reg_statistics = pd.DataFrame(stats)
    # reg_statistics.to_csv(f"./Outputs/{project_name}/registration_statistics.csv")

def trim_and_mark(num_div, num_markings, trim_dir, mosaics_dir):
    if len(os.listdir(mosaics_dir)) == 0:
        print("No mosaics found. Aborting process...")
        return
    num_mosaics = len([f for f in os.listdir(mosaics_dir) 
     if f.endswith('.png') and os.path.isfile(os.path.join(mosaics_dir, f))])

    for file_number in range(num_mosaics):
        file = os.path.join(mosaics_dir, f"{file_number}.png")
        img = np.array(cv2.imread(file))
        vert = img.shape[0]
        marking_box = int(0.02*vert/num_div)
        thickness = int(0.002*vert/num_div)

        if thickness == 0:
            thickness = 1
        if marking_box == 0:
            marking_box = 10
        
        boundaries = np.arange(0, vert, int(vert/num_div), dtype = np.int32)
        boundaries[-1] = vert
        boundaries = np.flip(boundaries)

        for i in range(len(boundaries)-1):
            upper_bound = boundaries[i]
            lower_bound = boundaries[i+1]

            trimmed = img[lower_bound:upper_bound,:,:]

            non_black_rows = np.any(trimmed != [0, 0, 0], axis=(1, 2))
            non_black_columns = np.any(trimmed != [0, 0, 0], axis=(0, 2))
            trimmed = trimmed[non_black_rows, :]
            trimmed = trimmed[:, non_black_columns]

            trimmed = np.ascontiguousarray(trimmed)

            mask = np.ones((trimmed.shape[0:2]))
            black_region = trimmed == [0,0,0]
            mask[black_region[...,0]] = 0

            rows, columns = np.where(mask == 1)
            valid_points = np.stack([rows, columns], axis = 1)

            random_markings = np.random.randint(0, len(valid_points), size = num_markings)

            chosen_marks = valid_points[random_markings]
            # print(chosen_marks)

            for mark in chosen_marks:
                centroid_y = mark[0]
                centroid_x = mark[1]


                trimmed = cv2.line(trimmed, (centroid_x - marking_box, centroid_y- marking_box), (centroid_x + marking_box, centroid_y+ marking_box), (0, 255, 255), thickness)
                trimmed = cv2.line(trimmed, (centroid_x - marking_box, centroid_y+ marking_box), (centroid_x + marking_box, centroid_y- marking_box), (0, 255, 255), thickness)
            cv2.imwrite(os.path.join(trim_dir, f"{file_number}_{i}.jpg"), trimmed)
