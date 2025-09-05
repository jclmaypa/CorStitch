# Running CorStitch
The .zip files contain the executables. To run the Python script, run gui_main.py with gui_init.py in the same directory.

User's Manual: https://tinyurl.com/CorStitchManual

# Overview
An Automated Rapid Reef Assessment System (ARRAS) has two parts. The first part is composed of a banca-towable platform equipped with a down-looking camera<sup>[1](#ref1)</sup>. By using the first part to conduct an ARRAS survey, a belt transect video is obtained. A sample of this video can be seen below.
<p align="center">
  <img src="https://github.com/jclmaypa/CorStitch/blob/47b69ecaae19efd6889afb5eb8edb0f2497467f3/Sample_Images/Sample_clip.gif?raw=true" alt="Description" width="100%"/>
</p>

The second part of ARRAS is a software for automatic video stitching to create georeferenced visual records from the data collected from the first part. Enter CorStitch, a free, open-source software aimed at converting down-looking belt transect videos from ARRAS surveys into georeferenced mosaics. 

# CorStitch

CorStitch can create panorama-like images called mosaics by using Fourier-based image registration<sup>[2](#ref2)</sup> on the central strips<sup>[3](#ref3)</sup> of adjacent frames to stitch them together. By repeating this process for an $n$ number of central strips of succeeding frames, CorStitch can create an $n$-second mosaic. The mosaics below are samples of $5$-second mosaics.

<p align="center">
  <img src="https://github.com/jclmaypa/CorStitch/blob/main/Sample_Images/Sample_mosaics.png?raw=true" alt="Description" width="100%"/>
</p>

Once the mosaics are created, they can then be georeferenced. CorStitch uses GNSS data to georeference the mosaics, and its main outputs are KMZ files and rectified mosaics. An example is shown below when a KMZ file is viewed using a Geospatial Software.

<p align="center">
  <img src="https://github.com/jclmaypa/CorStitch/blob/main/Sample_Images/Sample_georef.png?raw=true" alt="Description" width="100%"/>
</p>

# References:
<a name="ref1">[1]</a> Soriano, M. N. (n.d.). Automated Rapid Reef Assessment System (Arras). DOST Technology Transfer. <a href="https://tapitechtransfer.dost.gov.ph/technologies/it-development/automated-rapid-reef-assessment-system-arras">https://tapitechtransfer.dost.gov.ph/technologies/it-development/automated-rapid-reef-assessment-system-arras</a><br>

<a name="ref2">[2]</a> X. Tong et al., "Image Registration With Fourier-Based Image Correlation: A Comprehensive Review of Developments and Applications," in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 12, no. 10, pp. 4062-4081, Oct. 2019, doi: <a href="https://doi.org/10.1109/JSTARS.2019.2937690">10.1109/JSTARS.2019.2937690</a>.<br>

<a name="ref3">[3]</a> Aguinaldo, R.A, and Soriano, M., "Telecentric approximation in underwater image mosaics for minimizing parallax-induced errors", Proceedings of the Samahang Pisika ng Pilipinas 34, SPP-2016-4C-02 (2016). URL: <a href="https://proceedings.spp-online.org/article/view/SPP-2016-4C-02">https://proceedings.spp-online.org/article/view/SPP-2016-4C-02</a>.


