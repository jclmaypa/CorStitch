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
from PyQt5.QtGui import QRegExpValidator
from PyQt5.QtCore import QRegExp
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
from gui_init import mosaic_creation, scan_frames, mark_mosaics
import warnings
warnings.filterwarnings('ignore')
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
        video_folder_info.setToolTip(
            '<div style="white-space:pre-line; width:240px;">Select the folder containing the videos that you want to process.</div>'
        )

        video_folder_layout.addWidget(self.projects_dir, stretch=1)
        video_folder_layout.addWidget(projects_button)
        video_folder_layout.addWidget(video_folder_info)
        video_folder_widget.setLayout(video_folder_layout)
        form_layout.addRow("Data Folder:", video_folder_widget)

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


        self.frame_widgets = [self.frame_interval]
        self.set_enabled(self.frame_widgets, False)
        self.frame_extraction_checkbox.setEnabled(False)

        # --- Add horizontal line ---
        line2 = QFrame()
        line2.setFrameShape(QFrame.Shape.HLine)
        line2.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line2)

        # Create Mosaics
        self.create_mosaics_checkbox = QCheckBox("Create mosaics – Combines the frames into mosaics")
        self.create_mosaics_checkbox.stateChanged.connect(self.toggle_create_mosaics)
        layout.addWidget(self.create_mosaics_checkbox)

        mosaic_form = QFormLayout()
        mosaic_form.setVerticalSpacing(0)

        # Starting time (positive integers only, or 0 if you want to allow zero)
        self.starting_time = QLineEdit()
        self.starting_time.setText("0")
        self.starting_time.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        regex = QRegExp("[0-9]*")
        validator = QRegExpValidator(regex, self.starting_time)
        self.starting_time.setValidator(validator)

        starting_time_info = QLabel(self.info_icon_html)
        starting_time_info.setToolTip(
            '<div style="white-space:pre-line; width:240px;">This specifies the video time at which mosaic creation starts. If you want to start mosaic creation at the beginning of the video (i.e. you will use all the video frames for your mosaics), then set this to 0. For multiple videos, separate starting video times with a comma (e.g. 0, 10, 20).</div>'
        )
        starting_time_widget = QWidget()
        starting_time_layout = QHBoxLayout()
        starting_time_layout.setContentsMargins(0, 0, 0, 0)
        starting_time_layout.addWidget(self.starting_time)
        starting_time_layout.addWidget(starting_time_info)
        starting_time_widget.setLayout(starting_time_layout)
        mosaic_form.addRow("Starting video time:", starting_time_widget)

        frame_label = QLabel("Frame resolution:")
        frame_info = QLabel(self.info_icon_html)
        frame_info.setToolTip(
            '<div style="white-space:pre-line; width:240px;">This specifies the resolution of the frames that will be extracted from your videos. This will determine the quality of the mosaics.</div>'
        )

        self.frame_resolution = QComboBox()
        self.frame_resolution.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.frame_resolution.addItems(["360p","480p", "720p", "1080p"])

        frame_widget = QWidget()
        frame_widget_layout = QHBoxLayout()
        frame_widget_layout.setContentsMargins(0, 0, 0, 0)
        frame_widget_layout.addWidget(self.frame_resolution)
        frame_widget_layout.addWidget(frame_info)
        frame_widget.setLayout(frame_widget_layout)
        mosaic_form.addRow(frame_label, frame_widget)
        

        mosaic_widget = QWidget()
        mosaic_widget.setLayout(mosaic_form)
        mosaic_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout.addWidget(mosaic_widget)

        self.mosaic_widgets = [self.starting_time, self.frame_resolution]
        self.create_mosaics_checkbox.setEnabled(False)
        self.set_enabled(self.mosaic_widgets, False)


         # --- Add horizontal line ---
        line3 = QFrame()
        line3.setFrameShape(QFrame.Shape.HLine)
        line3.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line3)

        # Marking
        self.mark_checkbox = QCheckBox("Mark – Randomly distributes marks across the mosaic")
        self.mark_checkbox.stateChanged.connect(self.toggle_mark)
        layout.addWidget(self.mark_checkbox)

        mark_form = QFormLayout()
        mark_form.setVerticalSpacing(0)

        self.num_marks = QLineEdit()
        self.num_marks.setText("0") 
        self.num_marks.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        validator = QRegExpValidator(regex, self.num_marks)
        self.num_marks.setValidator(validator)

        mark_info = QLabel(self.info_icon_html)
        mark_info.setToolTip(
            '<div style="white-space:pre-line; width:240px;"> The number of marks to distribute per mosaic</div>'
        )
        num_marks_widget = QWidget()
        num_marks_layout = QHBoxLayout()
        num_marks_layout.setContentsMargins(0, 0, 0, 0)
        num_marks_layout.addWidget(self.num_marks)
        num_marks_layout.addWidget(mark_info)
        num_marks_widget.setLayout(num_marks_layout)
        mark_form.addRow("Number of marks per mosaic:", num_marks_widget)

        self.mark_widgets = [self.num_marks]
        self.set_enabled(self.mark_widgets, False)
        self.mark_checkbox.setEnabled(False)

        mark_widget = QWidget()
        mark_widget.setLayout(mark_form)
        mark_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout.addWidget(mark_widget)


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
        self.create_mosaics_checkbox.setEnabled(not all_filled)
        self.mark_checkbox.setEnabled(not partially_filled)
        

        if self.frame_extraction_checkbox.checkState() == Qt.Checked:
            self.set_enabled(self.frame_widgets, not partially_filled)
        
        if self.create_mosaics_checkbox.checkState() == Qt.Checked:
            self.set_enabled(self.mosaic_widgets, not partially_filled)

        if self.mark_checkbox.checkState() == Qt.Checked:
            self.set_enabled(self.mark_widgets, not partially_filled)
           

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
    def toggle_mark(self, state): 
        if self._ignore_toggle:
            return

        self.set_enabled(self.mark_widgets, state == Qt.Checked)


    def browse_output(self):
        dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir:
            self.output_dir.setText(dir)

    def browse_projects(self):
        dir = QFileDialog.getExistingDirectory(self, "Select videos folder")
        if dir:
            self.projects_dir.setText(dir)


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
        if self.mark_checkbox.isChecked():
            self.chosen_processes.append("mark_mosaics")


        data = {
            "frame_extraction": self.frame_extraction_checkbox.isChecked(),
            "create_mosaics": self.create_mosaics_checkbox.isChecked(),
            "project_name": self.project_name.text(),
            "video_folder": self.projects_dir.text(),
            "output_directory": self.output_dir.text(),
            "frame_resolution": self.frame_resolution.currentText(),
            "frame_interval": self.frame_interval.text(),
            "mosaic_time": int(99999999999999),
            "num_marks": int(self.num_marks.text()),
            "starting_time": self.starting_time.text(),
        }


        if np.all([not data["project_name"].strip(), not data["video_folder"].strip(), not data["output_directory"].strip()]):
            self.show_custom_popup("Please fill in the Project Name, Video Folder, and Output Directory.", title="Error")
            return

        if len(self.chosen_processes) == 0:
            self.show_custom_popup("Please select at least one process to run.", title="Error")
            return

        if "create_mosaics" in self.chosen_processes:
            if not data["starting_time"].strip():
                self.show_custom_popup("Please fill in the Starting Video Time.", title="Error")
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

        if "create_mosaics" in chosen_processes:
            mosaic_t = int(data["mosaic_time"])
            sync_vid_time = int(data["starting_time"])

        project_name = data["project_name"]
        vid_dir = data["video_folder"]
        output_dir = data["output_directory"]
        frame_interval = data["frame_interval"]

        video_res = data["frame_resolution"]
        num_marks  = data["num_marks"]

        project_dir = os.path.join(output_dir, project_name)
        mosaics_dir = os.path.join(project_dir, "Mosaics")
        mark_dir = os.path.join(project_dir, "Marked Mosaics")
        
        os.makedirs(mosaics_dir, exist_ok=True)
        os.makedirs(mark_dir, exist_ok=True)


        if "frame_extraction" in chosen_processes:
            print("Scanning Frames")
            scan_frames(vid_dir, mosaics_dir, int(frame_interval))

        if "create_mosaics" in chosen_processes:
            print("Creating Mosaics")
            mosaic_creation(mosaic_t, sync_vid_time, vid_dir, mosaics_dir, video_res)

        if "mark_mosaics" in chosen_processes and num_marks > 0:
            print("Marking mosaics")
            mark_mosaics(num_marks, mark_dir, mosaics_dir)


        print("All selected processes are complete!")
        print(f"You may access the processed data in the Outputs -> {project_dir}")
        print("Total runtime: ", np.round(time.time() - start_time, 2), "s")
        print("You can now safely exit the application.")
        time.sleep(3600)
    

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
