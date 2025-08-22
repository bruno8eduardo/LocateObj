import cv2
import numpy as np
from collections import deque
import json
import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import pymap3d.enu as enu
import rasterio
import utm
from utils.graphics import Graphics
from utils.geodetic import Geodetic
from utils.geometry import Geometry
from utils.parse_dji import parse_srt
from utils.ui import *
import time

lat0 = -22.905812 
lon0 = -43.221329
h0 = 12.456
utm0_x, utm0_y, utm_zn, utm_zl = utm.from_latlon(lat0, lon0)

roi_minimum_confidence = 0.65

with open("parameters.json", "r") as json_file:
    parameters = json.load(json_file)

K_path = parameters["K_path"]
with open(K_path, "r") as json_file:
    K = np.array(json.load(json_file), dtype=np.float64)

original_width = parameters["video_width"]
original_height = parameters["video_height"]

try:
    tif_path = parameters["tif_path"]
    with rasterio.open(tif_path) as dem_dataset:
        dem_elevation_data = dem_dataset.read(1)
        dem_transform = dem_dataset.transform
        dem_crs = dem_dataset.crs
        geodetic = Geodetic(dem_elevation_data, dem_transform, dem_crs)
except Exception as e:
    geodetic = Geodetic(None, None, None)
    print(f"Error: {e}\nConsidering flat terrain...")

if dem_elevation_data is not None:
    h0_dem = geodetic.get_DEM_alt(utm0_x, utm0_y)
    if h0_dem is not None:
        h_dem_offset = h0 - h0_dem
    else:
        raise Exception("Origem do sistema de coordenadas fora do mapa de elevação carregado!")
    
geometry = Geometry()
graphics = Graphics(geometry, original_width, original_height)

# Inicializar GLFW
if not glfw.init():
    raise Exception("GLFW não pôde ser inicializado!")

cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
gsrc = cv2.cuda.GpuMat()
gtemplate = cv2.cuda.GpuMat()
gresult = cv2.cuda.GpuMat()
if cuda_count != 0:
    print("CUDA enabled")
    cuda_matcher = cv2.cuda.createTemplateMatching(cv2.CV_8UC1, cv2.TM_CCOEFF_NORMED)

# Criar janela OpenGL
window = glfw.create_window(original_width, original_height, "Render 3D", None, None)
glfw.make_context_current(window)

glEnable(GL_DEPTH_TEST)

# Ativar iluminação
glEnable(GL_LIGHTING)

# Criar e ativar uma luz
glEnable(GL_LIGHT0)

# Definir a posição da luz (x, y, z, w)
light_position = [0, 3, 3, 1]  # (x=0, y=3, z=3, w=1 para luz pontual)
glLightfv(GL_LIGHT0, GL_POSITION, light_position)

# Definir intensidade da luz ambiente, difusa e especular
light_ambient = [0.2, 0.2, 0.2, 1.0]  # Luz fraca no ambiente
light_diffuse = [0.8, 0.8, 0.8, 1.0]  # Luz principal
light_specular = [1.0, 1.0, 1.0, 1.0]  # Reflexo especular forte

glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)
glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular)

# Ativar normalização de vetores normais (evita distorções)
glEnable(GL_NORMALIZE)

# Configurar matriz de projeção
proj_matrix = graphics.build_projection_matrix(K, original_width, original_height)
glMatrixMode(GL_PROJECTION)
glLoadMatrixf(np.transpose(proj_matrix))

K_inv = geometry.inv_K(K)

project_id = "car-models-rr7w5"
model_version = 1

source = parameters["video_path"]
cap = cv2.VideoCapture(source)

frame_info = parse_srt(parameters["video_data_path"])
frame_index = 0

original_width = parameters["video_width"]
original_height = parameters["video_height"]
window_name = "Locate"

frame_time_list = []

get_roi = False
image_roi_gray_list = []
roi_data_list = []
roi_pixel_list = []
roi_confidence_list = []
good_roi_list = []
good_roi_data_list = []

scale_reduct_inference = 6

clicks = deque(maxlen=10)
clicks_ENU = deque(maxlen=10)

# Localizacao carro: [latitude: -22.905551] [longitude: -43.221218] [rel_alt: 2.847 abs_alt: 15.331] 15.331 - 2.847 = 12.484
car_x, car_y, car_z = enu.geodetic2enu(-22.905551, -43.221218, 12.484, lat0, lon0, h0)
t_car_mundo = np.array([[car_x],[car_y],[car_z]])

play = True
images = []
while not glfw.window_should_close(window):
    start_frame = time.perf_counter_ns()
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    ret, image = cap.read()
    if ret:
        images.append(image)
    if play:
        frame_index += 1
    if frame_index >= len(frame_info):
        frame_index = 1

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('d'):
        if frame_index + 1 < len(images):
            frame_index += 1
        continue
    elif key & 0xFF == ord('f'):
        if frame_index + 10 < len(images):
            frame_index += 10
        continue
    elif key & 0xFF == ord('a'):
        frame_index -= 10
        if frame_index < 1:
            frame_index = 1
        continue
    elif key & 0xFF == ord('g'):
        graphics.glMode = not graphics.glMode
        continue
    elif key & 0xFF == ord('s'):
        get_roi = True
        continue
    elif key & 0xFF == ord(' '):
        play = not play
    
    image = images[frame_index - 1 if frame_index > 0 else 0].copy()
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    roi_pixel_list.clear()
    roi_confidence_list.clear()
    good_roi_list.clear()
    good_roi_data_list.clear()
    R_roi = None

    yaw = float(frame_info[frame_index]['gb_yaw'])
    pitch = float(frame_info[frame_index]['gb_pitch'])
    roll = float(frame_info[frame_index]['gb_roll'])
    R_drone = geometry.yaw_pitch_roll_to_rotation_matrix(yaw, pitch, roll)
    R_drone_T = np.transpose(R_drone)

    h_rel = float(frame_info[frame_index]['rel_alt'])
    h_abs = float(frame_info[frame_index]['abs_alt'])
    lat = float(frame_info[frame_index]['latitude'])
    long = float(frame_info[frame_index]['longitude'])

    easting, northing, h_enu = enu.geodetic2enu(lat, long, h_abs, lat0, lon0, h0)

    t_drone_mundo = np.array([[easting], [northing], [h_enu]])
    graphics.print_on_pixel(image, f"index:{frame_index}, N:{int(northing)}, E:{int(easting)}, h_rel:{h_rel}, yaw:{yaw}, pitch:{pitch}, roll:{roll}", 10, 10, (0,0,0))

    R = geometry.droneToCameraR @ R_drone_T @ geometry.mundoToDroneR

    if get_roi:
        rois = cv2.selectROIs("Select ROIs", image)
        cv2.destroyWindow("Select ROIs")
        for i,roi in enumerate(rois):
            x, y, w, h = roi
            image_roi = image[y:y+h, x:x+w]
            image_roi_gray_list.append(cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY))
            roi_data = get_roi_data(i)
            roi_data_list.append(roi_data)
        get_roi = False
    
    for i,image_roi_gray in enumerate(image_roi_gray_list):
        if cuda_count == 0:
            templ_match = cv2.matchTemplate(image_gray, image_roi_gray, cv2.TM_CCOEFF_NORMED)
        else:
            gsrc.upload(image_gray)
            gtemplate.upload(image_roi_gray)
            gresult = cuda_matcher.match(gsrc, gtemplate)
            templ_match = gresult.download()
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(templ_match)
        w, h = image_roi_gray.shape[::-1]
        roi_x = max_loc[0] + w/2
        roi_y = max_loc[1] + h/2
        roi_pixel = np.array([[roi_x], [roi_y], [1]])
        roi_pixel_list.append(roi_pixel)
        roi_confidence_list.append(max_val)
        graphics.desenhar_centro(image, int(roi_x), int(roi_y), (100, 0, 100), roi_flag=True)
        graphics.print_on_pixel(image, f"ROI similarity: {max_val:.3f}", int(roi_x), int(roi_y), (100, 0, 100))
    
    for i,roi_confidence in enumerate(roi_confidence_list):
        if roi_confidence > roi_minimum_confidence:
            good_roi_list.append(roi_pixel_list[i])
            lat_roi, long_roi, h_abs_roi = roi_data_list[i]
            easting_roi, northing_roi, h_enu_roi = enu.geodetic2enu(lat_roi, long_roi, h_abs_roi, lat0, lon0, h0)
            roi_enu = np.array([[easting_roi],[northing_roi],[h_enu_roi]])
            good_roi_data_list.append(roi_enu)
    
    if len(good_roi_list) == 1:
        R_roi = geometry.get_R_one_roi(good_roi_data_list[0], good_roi_list[0], R, K_inv, t_drone_mundo)
    elif len(good_roi_list) >= 2:
        R_roi = geometry.get_R_roi(good_roi_data_list, good_roi_list, K_inv, t_drone_mundo)
    
    t =  - R @ t_drone_mundo

    # Carro
    pixel_car = K @ np.concatenate((R, t), axis=1) @ np.vstack((t_car_mundo, [1]))
    pixel_car = pixel_car.flatten()
    pixel_car = pixel_car / pixel_car[2]
    graphics.instantiate(image, K, R, t, t_car_mundo, "red", t_drone_mundo, pitch)

    # Origem coordenada ENU
    graphics.instantiate(image, K, R, t, np.array([[0],[0],[0]]), "black", t_drone_mundo, pitch)

    for click in clicks:
        reta = geometry.reta3D(K_inv, geometry.droneToMundoR @ R_drone @ geometry.cameraToDroneR, t_drone_mundo, (click[0], click[1]))
        # click_ENU = find_ground_intersection_ENU(northing, easting, h_enu, reta[1])
        vec_DEM = geometry.norm_vec(reta[1].flatten())
        if vec_DEM[2] < 0:
            vec_DEM = (-1) * vec_DEM
        if dem_elevation_data is not None:
            click_ENU = geodetic.find_DEM_intersection(easting + utm0_x, northing + utm0_y, h_abs - h_dem_offset, vec_DEM)
        else:
            click_ENU = geodetic.find_ground_intersection_ENU(northing, easting, h_enu, vec_DEM)
        if click_ENU is not None:
            if dem_elevation_data is not None:
                click_ENU[0,0] -= utm0_x
                click_ENU[1,0] -= utm0_y
                click_ENU[2,0] += h_dem_offset - h0
            erro_car = np.linalg.norm(click_ENU - t_car_mundo)
            dist_drone = np.linalg.norm(t_drone_mundo - t_car_mundo)
            # Frame; Erro; Altura do Drone; Distância do Drone; Click ENU; Click Pixel; Car Pixel; Drone ENU
            # print(f"{frame_index}; {erro_car}; {h_rel}; {dist_drone}; {click_ENU.copy().flatten()}; {(click[0], click[1])}; {(pixel_car[0], pixel_car[1])}; {t_drone_mundo.copy().flatten()}")
            clicks_ENU.append(click_ENU)

    clicks.clear()
    clicks_ENU_copy = clicks_ENU.copy()

    for enu_click in clicks_ENU_copy:
        graphics.instantiate(image, K, R, t, enu_click, "blue", t_drone_mundo, pitch)
        if R_roi is not None:
            graphics.instantiate(image, K, R_roi, - R_roi @ t_drone_mundo, enu_click, "green", t_drone_mundo, pitch)
    
    glfw.poll_events()
    glfw.swap_buffers(window)
    
    pixels = glReadPixels(0, 0, original_width, original_height, GL_RGB, GL_UNSIGNED_BYTE)
    image = graphics.draw_opengl(pixels, image)
    
    cv2.imshow(window_name, image)
    cv2.setMouseCallback(window_name, mouse_click, (clicks, clicks_ENU))
    
    end_frame = time.perf_counter_ns()
    frame_time = end_frame - start_frame
    frame_time_list.append(frame_time)
    print(f"Frame: {frame_index}; Time ns: {frame_time}")

time_mean = np.mean(frame_time_list)
time_max = np.max(frame_time_list)
time_std = np.std(frame_time_list)

print(f"Time Mean: {time_mean * 1e-6} ms; Time Max: {time_max * 1e-6} ms; Time STD: {time_std * 1e-6} ms")