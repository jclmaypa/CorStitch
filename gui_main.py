# GNU GENERAL PUBLIC LICENSE
# CorStitch Copyright (C) 2025  Julian Christopher L. Maypa, Johnenn R. Manalang, and Maricor N. Soriano 
# This program comes with ABSOLUTELY NO WARRANTY;
# This is free software, and you are welcome to redistribute it under the conditions specified in the GNU General Public License.; 



from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QCheckBox, QComboBox, QFormLayout, QDateEdit,
    QTimeEdit, QSizePolicy, QGridLayout, QFrame, QToolButton, QToolTip, QDialog
)
import datetime
from PyQt5.QtCore import QDate, Qt, QTime
from PyQt5.QtGui import QIntValidator
import sys
import numpy as np
import gc
import simplekml
import scipy as sp
import os
import time
import copy
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import pandas as pd
from PIL import Image
import imutils
import matplotlib
matplotlib.use('Agg', force = True)
from gui_init import HMS2Conv, mosaic_creation, vid2frames, get_imgdim, GPSdata
valid_video_types = ['.mp4', '.avi', '.mov', '.mkv']
r_e = 6378.137*1000
deg2rad = np.pi/180
rad2deg = 180/np.pi

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CorStitch")
        # Remove fixed size, allow dynamic resizing
        # self.setFixedSize(480, 750)
        # self.setMinimumSize(400, 400)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._ignore_toggle = False
        self.init_ui()
        self.georef_part1 = False
        self.georef_part2 = False
        self.georef_part3 = False
        self.gps_data = 0
        self.raw_data = 0


    def init_ui(self):
        layout = QVBoxLayout()

        # Common info icon HTML string
        self.info_icon_html = (
            '<span style="display:inline-block; width:30px; height:18px; '
            'line-height:18px; text-align:center; font-family:Arial; font-weight:bold; '
            'color:#1976d2; border:2px solid #1976d2; border-radius:9px; '
            'font-size:13px;">  ?  </span>'
        )

        form_layout = QFormLayout()
        form_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        # Project name
        project_name_widget = QWidget()
        project_name_layout = QHBoxLayout()
        project_name_layout.setContentsMargins(0, 0, 0, 0)
        self.project_name = QLineEdit()
        self.project_name.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        project_name_info = QLabel(self.info_icon_html)
        project_name_info.setToolTip(
            '<div style="white-space:pre-line; width:240px;">Create a name for your project. This name will be used to create a folder in the Outputs directory. This folder is known as the project folder </div>'
        )
        project_name_layout.addWidget(self.project_name)
        project_name_layout.addWidget(project_name_info)
        project_name_widget.setLayout(project_name_layout)
        form_layout.addRow("Project name:", project_name_widget)

        # Video Folder
        video_folder_widget = QWidget()
        video_folder_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        video_folder_layout = QHBoxLayout()
        video_folder_layout.setContentsMargins(0, 0, 0, 0)
        video_folder_layout.setAlignment(Qt.AlignVCenter)
        self.projects_dir = QLineEdit()
        self.projects_dir.setReadOnly(True)
        self.projects_dir.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        projects_button = QPushButton("Browse")
        projects_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        projects_button.clicked.connect(self.browse_projects)
        video_folder_info = QLabel(self.info_icon_html)
        video_folder_info.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        # video_folder_info.setToolTip(
        #     '<div style="white-space:pre-line; width:240px;">Select the folder containing the videos you want to process.</div>'
        # )
        video_folder_layout.addWidget(self.projects_dir, stretch=1)
        video_folder_layout.addWidget(projects_button)
        video_folder_layout.addWidget(video_folder_info)
        video_folder_widget.setLayout(video_folder_layout)
        form_layout.addRow("Video Folder:", video_folder_widget)

        # Output directory
        output_dir_widget = QWidget()
        output_dir_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        output_dir_layout = QHBoxLayout()
        output_dir_layout.setContentsMargins(0, 0, 0, 0)
        output_dir_layout.setAlignment(Qt.AlignVCenter)
        self.output_dir = QLineEdit()
        self.output_dir.setReadOnly(True)
        self.output_dir.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        browse_button = QPushButton("Browse")
        browse_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        browse_button.clicked.connect(self.browse_output)
        output_dir_info = QLabel(self.info_icon_html)
        output_dir_info.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        output_dir_info.setToolTip(
            '<div style="white-space:pre-line; width:240px;">Select the output directory for your processed files. Inside this directory, a project folder will be created. This folder will contain all the processed files.</div>'
        )
        output_dir_layout.addWidget(self.output_dir, stretch=1)
        output_dir_layout.addWidget(browse_button)
        output_dir_layout.addWidget(output_dir_info)
        output_dir_widget.setLayout(output_dir_layout)
        form_layout.addRow("Output directory:", output_dir_widget)

        layout.addLayout(form_layout)

        self.project_name.textChanged.connect(self.update_checkboxes_enabled)
        self.projects_dir.textChanged.connect(self.update_checkboxes_enabled)
        self.output_dir.textChanged.connect(self.update_checkboxes_enabled)

        # --- Add horizontal line ---
        line1 = QFrame()
        line1.setFrameShape(QFrame.Shape.HLine)
        line1.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line1)

        # Frame Extraction
        self.frame_extraction_checkbox = QCheckBox("Frame extraction – Extracts the frames from your project videos")
        self.frame_extraction_checkbox.stateChanged.connect(self.toggle_frame_extraction)
        layout.addWidget(self.frame_extraction_checkbox)

        # Change frame_layout to QFormLayout
        frame_form = QFormLayout()
        frame_label = QLabel("Frame resolution:")
        frame_info = QLabel(self.info_icon_html)
        frame_info.setToolTip(
            '<div style="white-space:pre-line; width:240px;">This specifies the resolution of the frames that will be extracted from your videos. If you proceed to mosaic creation, the mosaics will have the same quality as your frames.</div>'
        )
        self.frame_resolution = QComboBox()
        self.frame_resolution.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.frame_resolution.addItems(["360p","480p", "720p", "1080p"])

        # Widget for input + info (resolution)
        frame_widget = QWidget()
        frame_widget_layout = QHBoxLayout()
        frame_widget_layout.setContentsMargins(0, 0, 0, 0)
        frame_widget_layout.addWidget(self.frame_resolution)
        frame_widget_layout.addWidget(frame_info)
        frame_widget.setLayout(frame_widget_layout)
        frame_form.addRow(frame_label, frame_widget)

        # Frame interval (natural numbers only)
        frame_interval_label = QLabel("Frame interval:")
        self.frame_interval = QLineEdit()
        self.frame_interval.setText("1")  # Set default value to 1
        self.frame_interval.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.frame_interval.setValidator(QIntValidator(1, 99999, self))  # Only allow natural numbers
        frame_interval_info = QLabel(self.info_icon_html)
        frame_interval_info.setToolTip(
            '<div style="white-space:pre-line; width:240px;">This specifies the interval at which frames are extracted from your videos. For example, an input of 5 will extract every 5th frame.</div>'
        )
        frame_interval_widget = QWidget()
        frame_interval_layout = QHBoxLayout()
        frame_interval_layout.setContentsMargins(0, 0, 0, 0)
        frame_interval_layout.addWidget(self.frame_interval)
        frame_interval_layout.addWidget(frame_interval_info)
        frame_interval_widget.setLayout(frame_interval_layout)
        frame_form.addRow(frame_interval_label, frame_interval_widget)
        layout.addLayout(frame_form)

        # --- Add horizontal line ---
        line2 = QFrame()
        line2.setFrameShape(QFrame.Shape.HLine)
        line2.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line2)

        self.frame_widgets = [self.frame_resolution, self.frame_interval]
        self.set_enabled(self.frame_widgets, False)
        self.frame_extraction_checkbox.setEnabled(False)

        # Create Mosaics
        self.create_mosaics_checkbox = QCheckBox("Create mosaics – Combines the frames into mosaics")
        self.create_mosaics_checkbox.stateChanged.connect(self.toggle_create_mosaics)
        layout.addWidget(self.create_mosaics_checkbox)

        mosaic_form = QFormLayout()
        mosaic_form.setVerticalSpacing(0)

        # Mosaic time (positive integers only)
        self.mosaic_time = QLineEdit()
        self.mosaic_time.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.mosaic_time.setValidator(QIntValidator(1, 99999, self))  # Only allow positive integers

        # Starting time (positive integers only, or 0 if you want to allow zero)
        self.starting_time = QLineEdit()
        self.starting_time.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.starting_time.setValidator(QIntValidator(0, 99999, self))  # Allow zero and positive integers

        mosaic_time_info = QLabel(self.info_icon_html)
        mosaic_time_info.setToolTip(
            '<div style="white-space:pre-line; width:240px;">This sets the length of your mosaics. For example, an input of \'5\' would mean each mosaic uses at most 5 seconds worth of video frames.</div>'
        )
        mosaic_time_widget = QWidget()
        mosaic_time_layout = QHBoxLayout()
        mosaic_time_layout.setContentsMargins(0, 0, 0, 0)
        mosaic_time_layout.addWidget(self.mosaic_time)
        mosaic_time_layout.addWidget(mosaic_time_info)
        mosaic_time_widget.setLayout(mosaic_time_layout)
        mosaic_form.addRow("Mosaic time (seconds):", mosaic_time_widget)

        starting_time_info = QLabel(self.info_icon_html)
        starting_time_info.setToolTip(
            '<div style="white-space:pre-line; width:240px;">This specifies the video time at which mosaic creation starts. If you want to start mosaic creation at the beginning of the video (i.e. you will use all the video frames for your mosaics), then set this to 0.</div>'
        )
        starting_time_widget = QWidget()
        starting_time_layout = QHBoxLayout()
        starting_time_layout.setContentsMargins(0, 0, 0, 0)
        starting_time_layout.addWidget(self.starting_time)
        starting_time_layout.addWidget(starting_time_info)
        starting_time_widget.setLayout(starting_time_layout)
        mosaic_form.addRow("Starting video time:", starting_time_widget)

        mosaic_widget = QWidget()
        mosaic_widget.setLayout(mosaic_form)
        mosaic_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout.addWidget(mosaic_widget)

        self.mosaic_widgets = [self.mosaic_time, self.starting_time]
        self.create_mosaics_checkbox.setEnabled(False)
        self.set_enabled(self.mosaic_widgets, False)

         # --- Add horizontal line ---
        line3 = QFrame()
        line3.setFrameShape(QFrame.Shape.HLine)
        line3.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line3)

        # Georeference
        self.georef_checkbox = QCheckBox("Georeference – Compiles and exports the mosaics as KMZs")
        self.georef_checkbox.stateChanged.connect(self.toggle_georeference)
        layout.addWidget(self.georef_checkbox)

        gnss_layout = QHBoxLayout()
        self.gnss_file = QLineEdit()
        self.gnss_file.setReadOnly(True)  
        gnss_browse = QPushButton("Browse")
        gnss_browse.clicked.connect(self.browse_gnss)

        gnss_info = QLabel(self.info_icon_html)
        gnss_info.setToolTip(
            '<div style="white-space:pre-line; width:240px;">The supported formats are: CSV and GPX.</div>'
        )
        gnss_layout.addWidget(self.gnss_file)
        gnss_layout.addWidget(gnss_browse)
        gnss_layout.addWidget(gnss_info)
        layout.addLayout(gnss_layout)

        # GNSS columns (two-column layout)
        self.time_col = QComboBox()
        self.lat_col = QComboBox()
        self.lon_col = QComboBox()
        self.depth_col = QComboBox()
        self.bearing_col = QComboBox()
        for cb in [self.time_col, self.lat_col, self.lon_col, self.depth_col, self.bearing_col]:
            cb.addItem("NA")

        column_grid = QGridLayout()
        column_grid.addWidget(QLabel("Time:"),      0, 0)
        column_grid.addWidget(self.time_col,        0, 1)
        column_grid.addWidget(QLabel("Depth:"),     0, 2)
        column_grid.addWidget(self.depth_col,       0, 3)

        column_grid.addWidget(QLabel("Latitude:"),  1, 0)
        column_grid.addWidget(self.lat_col,         1, 1)
        column_grid.addWidget(QLabel("Bearing:"),   1, 2)
        column_grid.addWidget(self.bearing_col,     1, 3)

        column_grid.addWidget(QLabel("Longitude:"), 2, 0)
        column_grid.addWidget(self.lon_col,         2, 1)

        layout.addLayout(column_grid)

        self.check_columns_button = QPushButton("Check columns")
        self.check_columns_button.clicked.connect(self.check_columns)
        layout.addWidget(self.check_columns_button)

        # Time and offset settings (aligned and equal width)
        time_form = QFormLayout()
        # input_width = 200  # Remove fixed width

        self.date_picker_container = QWidget()
        self.date_picker_layout = QHBoxLayout()
        self.date_picker_layout.setContentsMargins(0, 0, 0, 0)

        self.date_picker = QComboBox()
        self.date_picker.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        date_picker_info = QLabel(self.info_icon_html)
        date_picker_info.setToolTip(
            '<div style="white-space:pre-line; width:240px;">Select the date when the data was collected.</div>'
        )
        self.date_picker_layout.addWidget(self.date_picker)
        self.date_picker_layout.addWidget(date_picker_info)
        self.date_picker_container.setLayout(self.date_picker_layout)
        time_form.addRow("The date of data collection:", self.date_picker_container)

        self.sync_time = QTimeEdit()
        self.sync_time.setDisplayFormat("HH:mm:ss")
        self.sync_time.setTime(QTime(0, 0, 0))
        self.sync_time.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sync_time_info = QLabel(self.info_icon_html)
        sync_time_info.setToolTip(
            '<div style="white-space:pre-line; width:240px;">This specifies the time at which your GNSS data synchronizes with your video data. This is in a 24-hour format.</div>'
        )
        sync_time_widget = QWidget()
        sync_time_layout = QHBoxLayout()
        sync_time_layout.setContentsMargins(0, 0, 0, 0)
        sync_time_layout.addWidget(self.sync_time)
        sync_time_layout.addWidget(sync_time_info)
        sync_time_widget.setLayout(sync_time_layout)
        time_form.addRow("GNSS Synchronization time:", sync_time_widget)

        self.utc_offset = QComboBox()
        self.utc_offset.addItems([str(i) for i in range(-12, 13)])
        self.utc_offset.setCurrentText("0")  # Set the placeholder/default to 0
        self.utc_offset.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        utc_offset_info = QLabel(self.info_icon_html)
        utc_offset_info.setToolTip(
            '<div style="white-space:pre-line; width:240px;">This specifies the UTC format of the GNSS data in your GNSS file. For example, if your GNSS data is in UTC+2, select 2. If it is in UTC-5, select -5.</div>'
        )
        utc_offset_widget = QWidget()
        utc_offset_layout = QHBoxLayout()
        utc_offset_layout.setContentsMargins(0, 0, 0, 0)
        utc_offset_layout.addWidget(self.utc_offset)
        utc_offset_layout.addWidget(utc_offset_info)
        utc_offset_widget.setLayout(utc_offset_layout)
        time_form.addRow("UTC offset in your GNSS data:", utc_offset_widget)

        layout.addLayout(time_form)

        self.georef_widgets = [
            self.gnss_file, gnss_browse, 
            self.time_col, 
            self.lat_col,
            self.lon_col, 
            self.depth_col, 
            self.bearing_col,
            self.check_columns_button, 
            self.date_picker,
            self.sync_time, 
            self.utc_offset
        ]
        self.georef_widgets_part1 = [self.gnss_file, gnss_browse]
        self.georef_widgets_part2 = [ self.time_col, self.lat_col, self.lon_col, self.depth_col, self.bearing_col, self.check_columns_button, ]
        self.georef_widgets_part3 = [self.date_picker, self.sync_time, self.utc_offset]
        self.set_enabled(self.georef_widgets, False)
        self.georef_checkbox.setEnabled(False)

        # Run button
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.run_data)
        layout.addWidget(self.run_button)

        self.setLayout(layout)

    def update_checkboxes_enabled(self):
        all_filled = np.any([
            not self.project_name.text().strip(),
            not self.projects_dir.text().strip(),
            not self.output_dir.text().strip()]
        )
        partially_filled = np.any([
            not self.project_name.text().strip(),
            not self.output_dir.text().strip()]
        )
        self.frame_extraction_checkbox.setEnabled(not all_filled)
        self.create_mosaics_checkbox.setEnabled(not partially_filled)
        self.georef_checkbox.setEnabled(not partially_filled)
        

        if self.frame_extraction_checkbox.checkState() == Qt.Checked:
            self.set_enabled(self.frame_widgets, not partially_filled)
        if self.create_mosaics_checkbox.checkState() == Qt.Checked:
            self.set_enabled(self.mosaic_widgets, not partially_filled)

        if self.georef_checkbox.checkState() == Qt.Checked:
            if self.georef_part1 == False:
                self.set_enabled(self.georef_widgets_part1, not partially_filled)
            if self.georef_part1 == True:
                self.set_enabled(self.georef_widgets_part1, not partially_filled)
                self.set_enabled(self.georef_widgets_part2, not partially_filled)
            if self.georef_part1 == True and self.georef_part2 == True:
                self.set_enabled(self.georef_widgets_part1, not partially_filled)
                self.set_enabled(self.georef_widgets_part2, not partially_filled)
                self.set_enabled(self.georef_widgets_part3, not partially_filled)
           


    def set_enabled(self, widgets, enabled):
        for w in widgets:
            w.setEnabled(enabled)

    def toggle_frame_extraction(self, state):
        if self._ignore_toggle:
            return

        self.set_enabled(self.frame_widgets, state == Qt.Checked)

    def toggle_create_mosaics(self, state): 
        if self._ignore_toggle:
            return

        self.set_enabled(self.mosaic_widgets, state == Qt.Checked)


    def toggle_georeference(self, state):
        if self._ignore_toggle:
            return
        
        if self.georef_part1 == False:
            self.set_enabled(self.georef_widgets_part1, state == Qt.Checked)
        if self.georef_part1 == True:
                self.set_enabled(self.georef_widgets_part1, state == Qt.Checked)
                self.set_enabled(self.georef_widgets_part2, state == Qt.Checked)
        if self.georef_part1 == True and self.georef_part2 == True:
                self.set_enabled(self.georef_widgets_part1, state == Qt.Checked)
                self.set_enabled(self.georef_widgets_part2, state == Qt.Checked)
                self.set_enabled(self.georef_widgets_part3, state == Qt.Checked)

    def browse_output(self):
        dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir:
            self.output_dir.setText(dir)

    def browse_projects(self):
        dir = QFileDialog.getExistingDirectory(self, "Select videos folder")
        if dir:
            self.projects_dir.setText(dir)

    def browse_gnss(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select GNSS File", "", "Comma Separated Values (*.csv);;GPS eXchange (*.gpx)")
        if file:
            self.gnss_file.setText(file)
            self.set_enabled(self.georef_widgets_part2, True)
            self.set_enabled(self.georef_widgets_part3, False)

            self.raw_data = GPSdata(file)
            self.raw_columns = self.raw_data.columns
            
            for cb in [self.time_col, self.lat_col, self.lon_col, self.depth_col, self.bearing_col]:
                cb.clear()
                cb.addItem("NA")
                cb.addItems([str(col) for col in self.raw_columns])
        self.georef_part1 = True
        self.georef_part2 = False

    def check_columns(self):
        if np.all([self.time_col.currentText() == "NA", self.lat_col.currentText() == "NA", self.lon_col.currentText() == "NA"]):
            self.show_custom_popup("Please select at least one column for Time, Latitude, and Longitude. These are essential for georeferencing, and we cannont proceed without them.")
            return
        self.chosen_columns = [
            self.time_col.currentText(),
            self.lat_col.currentText(),
            self.lon_col.currentText(),
            self.depth_col.currentText(),
            self.bearing_col.currentText()
        ]
        self.raw_data.read_gps_data(self.chosen_columns)
        self.gps_data = self.raw_data.export()
        if np.all(self.gps_data.date_time.str.contains('Z')) and np.all(self.gps_data.date_time.str.contains('T')):
            self.raw_data.date_time_split()
            self.gps_data = self.raw_data.export()
            self.unique_dates = self.gps_data.date.unique()

            # Remove all widgets from date_picker_layout
            while self.date_picker_layout.count():
                item = self.date_picker_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

            if len(self.unique_dates) >= 1:
                # Use dropdown
                self.date_picker = QComboBox()
                self.date_picker.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
                self.date_picker.addItems([str(d) for d in self.unique_dates])
            else:
                # Use string input
                self.date_picker = QLineEdit()
                self.date_picker.setPlaceholderText("Enter date (YYYY-MM-DD)")
                self.date_picker.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

            # Always re-add the info icon
            date_picker_info = QLabel(self.info_icon_html)
            date_picker_info.setToolTip(
                '<div style="white-space:pre-line; width:240px;">Select the date when the data was collected.</div>'
            )

            self.date_picker_layout.addWidget(self.date_picker)
            self.date_picker_layout.addWidget(date_picker_info)
        else:
            self.raw_data.date_time_split(splitting=False)
            self.unique_dates = 0

            # Remove all widgets from date_picker_layout
            while self.date_picker_layout.count():
                item = self.date_picker_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

            self.date_picker = QLineEdit()
            self.date_picker.setPlaceholderText("")
            self.date_picker.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

            date_picker_info = QLabel(self.info_icon_html)
            date_picker_info.setToolTip(
                '<div style="white-space:pre-line; width:240px;">Select the date when the data was collected. You may choose any date format convenient to you.</div>'
            )

            self.date_picker_layout.addWidget(self.date_picker)
            self.date_picker_layout.addWidget(date_picker_info)

        self.gps_data = self.raw_data.export()
        self.georef_part2 = True
        self.set_enabled(self.georef_widgets_part3, True)
        # Update georef_widgets_part3 to avoid referencing deleted widgets
        self.georef_widgets_part3 = [self.date_picker, self.sync_time, self.utc_offset]


    def show_custom_popup(self, message, title = "Message"):
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        layout = QVBoxLayout()
        label = QLabel(message)
        layout.addWidget(label)
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(dialog.accept)
        layout.addWidget(ok_button, alignment=Qt.AlignCenter)
        dialog.setLayout(layout)
        dialog.exec_()

    def run_data(self):
        self.chosen_processes = []
        if self.frame_extraction_checkbox.isChecked():
            self.chosen_processes.append("frame_extraction")
        if self.create_mosaics_checkbox.isChecked():
            self.chosen_processes.append("create_mosaics") 
        if self.georef_checkbox.isChecked():
            self.chosen_processes.append("georeference")
        if isinstance(self.date_picker, QComboBox):
            self.date_value = self.date_picker.currentText()
        else:
            self.date_value = self.date_picker.text()

        data = {
            "frame_extraction": self.frame_extraction_checkbox.isChecked(),
            "create_mosaics": self.create_mosaics_checkbox.isChecked(),
            "georeference": self.georef_checkbox.isChecked(),
            "project_name": self.project_name.text(),
            "video_folder": self.projects_dir.text(),
            "output_directory": self.output_dir.text(),
            "frame_resolution": self.frame_resolution.currentText(),
            "frame_interval": self.frame_interval.text(),
            "mosaic_time": self.mosaic_time.text(),
            "starting_time": self.starting_time.text(),
            "gnss_file": self.gnss_file.text(),
            "time_col": self.time_col.currentText(),
            "lat_col": self.lat_col.currentText(),
            "lon_col": self.lon_col.currentText(),
            "depth_col": self.depth_col.currentText(),
            "bearing_col": self.bearing_col.currentText(),
            "utc_offset": self.utc_offset.currentText(),
        }

        if "georeference" in self.chosen_processes:
            if self.georef_part2 == True:
                data["sync_time"] = self.sync_time.time().toString("HH:mm:ss"),
                data["date_picker"] = self.date_value
                data["unique_dates"] = self.unique_dates,
                data["depth_status"] = self.raw_data.depth_status,
                data["bearing_status"] = self.raw_data.heading_status
            else:
                data["sync_time"] = " "
                data["date_picker"] = " "
                data["unique_dates"] = " "
                data["depth_status"] = " "
                data["bearing_status"] = " "
        else:
            data["sync_time"] = " "
            data["date_picker"] = " "
            data["unique_dates"] = " "
            data["depth_status"] = " "
            data["bearing_status"] = " "

        if np.all([not data["project_name"].strip(), not data["video_folder"].strip(), not data["output_directory"].strip()]):
            self.show_custom_popup("Please fill in the Project Name, Video Folder, and Output Directory.", title="Error")
            return

        if len(self.chosen_processes) == 0:
            self.show_custom_popup("Please select at least one process to run.", title="Error")
            return

        if "create_mosaics" in self.chosen_processes:
            if np.any([not data["mosaic_time"].strip(), not data["starting_time"].strip()]):
                self.show_custom_popup("Please fill in the Mosaic Time and Starting Video Time.", title="Error")
                return
        if "georeference" in self.chosen_processes:
            if not data["gnss_file"].strip():
                self.show_custom_popup("Please choose a GNSS file.", title="Error")
                return
            if self.georef_part2 == False:
                self.show_custom_popup("Please click 'Check Columns' first", title="Error")
                return
            if np.any([data["time_col"] == "NA", data["lat_col"] == "NA", data["lon_col"] == "NA"]):
                self.show_custom_popup("Please select at least one column for Time, Latitude, and Longitude. These are essential for georeferencing, and we cannot proceed without them.", title="Error")
                return

        self.exported_data = data  # Store as an attribute
        self.show_custom_popup("Your data will now be processed. Please click 'OK' to proceed.", title="Data Preparation Complete")
        self.run_button.setEnabled(False)

        # Call the processing function
        self.process_data()
        # Exit the application
        QApplication.quit()

    def process_data(self):
        self.hide()
        """Process the data after the GUI is closed."""
        start_time = time.time()
        data = copy.deepcopy(self.exported_data)
        chosen_processes = copy.deepcopy(self.chosen_processes)
        gps_data = copy.deepcopy(self.gps_data)
        raw_data = copy.deepcopy(self.raw_data)

        if "create_mosaics" in chosen_processes:
            mosaic_t = int(data["mosaic_time"])
            sync_vid_time = int(data["starting_time"])
        if "georeference" in chosen_processes:
            utc_offset = int(data["utc_offset"])
            sync_UTC_time = str(data["sync_time"][0])
            sync_UTC_time = datetime.datetime.strptime(sync_UTC_time,'%H:%M:%S')
            sync_UTC_time = sync_UTC_time + datetime.timedelta(hours = - 8 + utc_offset)
            sync_UTC_time = sync_UTC_time.strftime("%H:%M:%S")

        project_name = data["project_name"]
        vid_dir = data["video_folder"]
        output_dir = data["output_directory"]
        frame_res = data["frame_resolution"]
        frame_interval = data["frame_interval"]
        date = data["date_picker"]

        video_res = data["frame_resolution"]
        unique_dates = data["unique_dates"]
        depth_status = data["depth_status"][0]
        bearing_status = data["bearing_status"]

        project_dir = os.path.join(output_dir, project_name)
        frames_dir = os.path.join(project_dir, "Frames")
        mosaics_dir = os.path.join(project_dir, "Mosaics")
        georef_dir = os.path.join(project_dir, "Georeferenced")
        rect_mosaics_dir = os.path.join(georef_dir, "Rectified Mosaics")
        kmz_dir = os.path.join(georef_dir, "KMZ files")
        
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(mosaics_dir, exist_ok=True)
        os.makedirs(rect_mosaics_dir, exist_ok=True)
        os.makedirs(kmz_dir, exist_ok=True)

        if "frame_extraction" in chosen_processes:
            print("Extracting frames...")
            last_frame = 0
            for file_name in np.sort(os.listdir(vid_dir)):
                if os.path.splitext(file_name)[1].lower() in valid_video_types:
                    last_frame = vid2frames(file_name, vid_dir, frames_dir, int(frame_interval), reduce=True, res=video_res, image_counter=last_frame)

        if "create_mosaics" in chosen_processes:
            print("Creating Mosaics...")
            mosaic_creation(mosaic_t, sync_vid_time, frames_dir, mosaics_dir)

        if "georeference" in chosen_processes:
            if len(unique_dates) > 1:
                gps_data = gps_data[gps_data.date != date.index]

            gps_data["conv_time"] = gps_data.time.apply(HMS2Conv)

            if len(gps_data[gps_data.time <= sync_UTC_time].index) <= 0:
                print("WARNING: No data points were removed during GPS and Video synchronization. This could mean that GPS data collection started after the synchronization time.")
            gps_data = gps_data[gps_data.time >= sync_UTC_time].reset_index(drop=True)
            interp_inc = 1
            interpolation_time = np.arange(np.min(gps_data.conv_time), np.max(gps_data.conv_time)+interp_inc, interp_inc)
            lon_interp = sp.interpolate.interp1d(gps_data.conv_time, gps_data.lon, kind='linear', fill_value='extrapolate', bounds_error=False)
            lat_interp = sp.interpolate.interp1d(gps_data.conv_time, gps_data.lat, kind='linear', fill_value='extrapolate', bounds_error=False)
            heading_interp = sp.interpolate.interp1d(gps_data.conv_time, gps_data.instr_heading, kind='linear', fill_value='extrapolate', bounds_error=False)

            interp_gps_data = pd.DataFrame({"conv_time": interpolation_time})
            interp_gps_data["lon"] = lon_interp(interpolation_time)
            interp_gps_data["lat"] = lat_interp(interpolation_time)
            interp_gps_data["instr_heading"] = heading_interp(interpolation_time)

            if depth_status == 1:
                depth_interp = sp.interpolate.interp1d(gps_data.conv_time, gps_data.dep_m, kind='linear', fill_value='extrapolate', bounds_error=False)
                interp_gps_data["dep_m"] = depth_interp(interpolation_time)
            interp_gps_data['sync_time'] = interp_gps_data.conv_time - interp_gps_data.conv_time.min()

            print("Georeferencing images...")
            with open(os.path.join(mosaics_dir, "mosaics_data.txt"), 'r') as file:
                mosaic_data = file.readline()
            mosaic_data = eval(mosaic_data)
            mosaic_t = mosaic_data["mosaic_time"]
            num_mosaics = mosaic_data["num_mosaics"]
            if num_mosaics == 0:
                print("No mosaics detected for this project. No images can be georeferenced")
            else:
                mosaic_end = int(np.floor(mosaic_t/interp_inc))
                mosaic_boundaries = np.arange(0, mosaic_end*num_mosaics+mosaic_end, mosaic_end)
                kmz_limit = 100
                img_counter = 0
                kmz_counter = 0
                mosaics = np.arange(0, num_mosaics, 1)
                kml = simplekml.Kml()
                depth = 0
                if depth_status == 1:
                    width_m = 1.55948 * (np.mean(interp_gps_data.dep_m[mosaic_boundaries[0]: mosaic_boundaries[-1]]))
                else:
                    width_m = 5
                for i in range(0, len(mosaic_boundaries)-1):
                    start = mosaic_boundaries[i]
                    end = mosaic_boundaries[i+1]
                    mid = start + (end-start)//2
                    icon_path = os.path.join(mosaics_dir, f"{mosaics[i]}.png")
                    lpx, wpx = get_imgdim(icon_path)

                    headings = interp_gps_data.instr_heading.iloc[start]
                    try:
                        headinge = interp_gps_data.instr_heading.iloc[end]
                    except:
                        print("Could not georeference all mosaics due to the lack of GPS data.")
                        break
                    heading = interp_gps_data.instr_heading.iloc[mid]

                    lat_s = interp_gps_data.lat.iloc[start]
                    lon_s = interp_gps_data.lon.iloc[start]
                    lat_e = interp_gps_data.lat.iloc[end]
                    lon_e = interp_gps_data.lon.iloc[end]
                    lat_m = interp_gps_data.lat.iloc[mid]
                    lon_m = interp_gps_data.lon.iloc[mid]

                    if depth_status == 1:
                        depth = np.mean(interp_gps_data.dep_m[start:end+1])
                        wm = 1.55948 * depth
                        px2m = wm / wpx
                        scalebar = ScaleBar(dx=px2m,
                                            units='m',
                                            fixed_value=1,
                                            fixed_units='m',
                                            location="lower left",
                                            font_properties={'family': 'monospace',
                                                            'weight': 'semibold',
                                                            'size': 20})
                        
                    img = np.array(Image.open(os.path.join(mosaics_dir, f"{mosaics[i]}.png")))[:, :, 0:3]
                    img = imutils.rotate_bound(img, angle=heading)
                    non_black_rows = np.any(img != [0, 0, 0], axis=(1, 2))
                    non_black_columns = np.any(img != [0, 0, 0], axis=(0, 2))
                    img = img[non_black_rows, :]
                    img = img[:, non_black_columns]
                    rlpx, rwpx = img.shape[0], img.shape[1]
                    fig, ax = plt.subplots(figsize=(20, 20))
                    img_desc = '\n'.join((
                        r'mosaic no. %.0f' % (i),
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

                    fig.savefig(os.path.join(rect_mosaics_dir, f"{mosaics[i]}.jpg"), bbox_inches='tight')
                    plt.close('all')
                    gc.collect()

                    point = kml.newpoint(name=f"{mosaics[i]}", coords=[(lon_m, lat_m)])
                    picpath = kml.addfile(os.path.join(rect_mosaics_dir, f"{i}.jpg"))
                    img_desc = f'<img src="{picpath}" alt="picture" width="{rwpx}" height="{rlpx}" align="left" />'
                    point.style.balloonstyle.text = img_desc

                    ground = kml.newgroundoverlay(name=f"{mosaics[i]}.png")
                    ground.icon.href = icon_path
                    ground.description = f"{mosaics[i]}.png\nheading: {heading}"

                    if (heading >= 270 and heading <= 360) or (heading >= 0 and heading <= 90):
                        dxs = -0.5 * width_m * np.cos(headings * deg2rad)
                        dys = -0.5 * width_m * np.sin(headings * deg2rad)
                        dx2s = -dxs
                        dy2s = -dys
                        dxe = -0.5 * width_m * np.cos(headinge * deg2rad)
                        dye = -0.5 * width_m * np.sin(headinge * deg2rad)
                        dx2e = -dxe
                        dy2e = -dye

                        tr_lon = lon_s + (dxs / r_e) * (rad2deg) / np.cos(lat_s * deg2rad)
                        tr_lat = lat_s + (dys / r_e) * (rad2deg)
                        tl_lon = lon_s + (dx2s / r_e) * (rad2deg) / np.cos(lat_s * deg2rad)
                        tl_lat = lat_s + (dy2s / r_e) * (rad2deg)
                        bl_lon = lon_e + (dx2e / r_e) * (rad2deg) / np.cos(lat_e * deg2rad)
                        bl_lat = lat_e + (dy2e / r_e) * (rad2deg)
                        br_lon = lon_e + (dxe / r_e) * (rad2deg) / np.cos(lat_e * deg2rad)
                        br_lat = lat_e + (dye / r_e) * (rad2deg)

                        if br_lon < bl_lon:
                            bl_lon, br_lon = br_lon, bl_lon
                        if tr_lon < tl_lon:
                            tl_lon, tr_lon = tr_lon, tl_lon

                        ground.gxlatlonquad.coords = [(tl_lon, tl_lat), (tr_lon, tr_lat), (br_lon, br_lat), (bl_lon, bl_lat)]
                    else:
                        dxs = 0.5 * width_m * np.cos(headings * deg2rad)
                        dys = 0.5 * width_m * np.sin(headings * deg2rad)
                        dx2s = -dxs
                        dy2s = -dys
                        dxe = 0.5 * width_m * np.cos(headinge * deg2rad)
                        dye = 0.5 * width_m * np.sin(headinge * deg2rad)
                        dx2e = -dxe
                        dy2e = -dye

                        tr_lon = lon_s + (dxs / r_e) * (rad2deg) / np.cos(lat_s * deg2rad)
                        tr_lat = lat_s + (dys / r_e) * (rad2deg)
                        tl_lon = lon_s + (dx2s / r_e) * (rad2deg) / np.cos(lat_s * deg2rad)
                        tl_lat = lat_s + (dy2s / r_e) * (rad2deg)
                        bl_lon = lon_e + (dx2e / r_e) * (rad2deg) / np.cos(lat_e * deg2rad)
                        bl_lat = lat_e + (dy2e / r_e) * (rad2deg)
                        br_lon = lon_e + (dxe / r_e) * (rad2deg) / np.cos(lat_e * deg2rad)
                        br_lat = lat_e + (dye / r_e) * (rad2deg)

                        if br_lon < bl_lon:
                            bl_lon, br_lon = br_lon, bl_lon
                        if tr_lon < tl_lon:
                            tl_lon, tr_lon = tr_lon, tl_lon

                        ground.gxlatlonquad.coords = [(tr_lon, tr_lat), (tl_lon, tl_lat), (bl_lon, bl_lat), (br_lon, br_lat)]

                    img_counter += 1
                    if img_counter % kmz_limit == 0:
                        kml.savekmz(os.path.join(kmz_dir, f"{kmz_counter}.kmz"))
                        print(f"Created: {kmz_counter}.kmz")
                        kml = simplekml.Kml()
                        kmz_counter += 1
                if img_counter % kmz_limit != 0:
                    kml.savekmz(os.path.join(kmz_dir, f"{kmz_counter}.kmz"))
                    print(f"Created: {kmz_counter}.kmz")

        print("All selected processes are complete!")
        print(f"You may access the processed data in the Outputs -> {project_dir}")
        print("Total runtime: ", np.round(time.time() - start_time, 2), "s")
        print("You can now safely exit the application.")
    

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet("""
        QToolTip {
            background-color: #fdfdfd;
            color: #222;
            padding: 6px;
            font-size: 12px;
            max-width: 250px;
        }
    """)
    window = MainWindow()
    window.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    screen = app.primaryScreen().availableGeometry()
    default_width = screen.width()  # or any reasonable width
    # Let Qt compute the optimal size
    window.adjustSize()  # Let Qt compute the optimal size
    window.move((screen.width() - default_width) // 2, 0)
    window.show()
    sys.exit(app.exec_())