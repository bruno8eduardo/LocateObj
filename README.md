# LocateObj: Augmented Reality for Drone-Based Geo-Objects

A **Python** system that overlays **georeferenced 3D virtual objects** on drone video frames.  
It consumes the drone’s video and synchronized **SRT telemetry** to compute camera pose and project objects.  
If a **GeoTIFF** (DEM) is provided, object projection follows real terrain; otherwise, a **flat-terrain assumption** is used.
The system currently works only with offline videos and their associated `.SRT` telemetry files.
Support for real-time video streams is planned for future development.

---

## Features

- Render **3D virtual objects** aligned with the drone video.  
- **Drone telemetry overlay** displayed on the top of the screen.  
- **Mouse interaction**: place new virtual objects at the correct georeferenced location by clicking on the video.  
- **Optional terrain correction** using GeoTIFF elevation data.  
- Orientation correction using **ground control points**.  
- Interactive playback controls for quick video inspection.  

---

## ⚙️ Requirements

This project requires Python **3.9+**.

### Python Libraries & Links

- **[OpenCV](https://opencv.org/)** – video and image processing 
  - Install: `pip install opencv-python`  
  - ⚡ **Highly recommended:** use an **OpenCV build with CUDA support** for accelerated video decoding and processing.  
    - There are several step-by-step guides online. Example: [Building OpenCV with CUDA Support](https://www.blog.neudeep.com/python/building-opencv-with-cuda-support-a-step-by-step-guide/2292/)   

- **[NumPy](https://numpy.org/)** – numerical computing   
  - Install: `pip install numpy`  

- **[PyOpenGL](https://pypi.org/project/PyOpenGL/)** – OpenGL bindings for Python  
  - Install: `pip install PyOpenGL PyOpenGL_accelerate`  

- **[glfw](https://pypi.org/project/glfw/)** – window/context creation and input handling    
  - Install: `pip install glfw`  

- **[rasterio](https://pypi.org/project/rasterio/)** – reading GeoTIFF / raster data  
  - Requires: Python ≥ 3.9, GDAL ≥ 3.5, NumPy ≥ 1.24  
  - Install: `pip install rasterio` (or use `conda install -c conda-forge rasterio` for easier GDAL management)  

- **[pymap3d](https://pypi.org/project/pymap3d/)** – geodetic/ENU coordinate conversions    
  - Install: `pip install pymap3d`  

- **[utm](https://pypi.org/project/utm/)** – UTM coordinate conversions  
  - Install: `pip install utm`  

---

### Installation (pip)

```bash
pip install opencv-python numpy PyOpenGL PyOpenGL_accelerate glfw rasterio pymap3d utm
```

## 📂 Parameters File

The system requires a configuration file `parameters.json` with video paths, telemetry, and calibration data:

```json
{
  "K_path": "tests/QuintaBoaVista/K-mavic-HD.json",
  "tif_path": "tests/QuintaBoaVista/MDE_27454so_v1.tif",
  "video_path": "tests/QuintaBoaVista/DJI_20241209160542_0002_S.MP4",
  "video_data_path": "tests/QuintaBoaVista/DJI_20241209160542_0002_S.SRT",
  "video_width": 1920,
  "video_height": 1080
}
```
- **K_path**: Path to camera intrinsics (JSON).  
- **tif_path**: Path to GeoTIFF DEM. **If omitted, terrain is assumed flat.**  
- **video_path**: Path to the drone video file.  
- **video_data_path**: Path to the `.SRT` telemetry file (must align with the video frames).  
- **video_width / video_height**: Resolution of the video being processed.  

---

## ▶️ Running

1. Prepare a `parameters.json` file as shown above.  
2. Launch the program with:  
```bash
python locate-obj.py
```
3. Use the keyboard and mouse controls during playback to interact with the system.

## 🎮 Controls

### Keyboard
| Key        | Action                                                                                       |
|------------|-----------------------------------------------------------------------------------------------|
| **Space**  | Pause / resume execution                                                                     |
| **a**      | Rewind **10** frames                                                                          |
| **d**      | Advance **1** frame                                                                           |
| **f**      | Advance **10** frames                                                                         |
| **g**      | Toggle rendering of **3D virtual objects**                                                    |
| **s**      | Activate **rotation correction** based on known ground control points                           |
| **q**      | Quit program                                                                                  |

### Mouse
| Action       | Effect                                                                                      |
|--------------|---------------------------------------------------------------------------------------------|
| **Left Click** | Instantiate a new **virtual object** at the clicked georeferenced location                 |
| **Right Click** | Delete the **nearest virtual object** at the clicked georeferenced location                 |