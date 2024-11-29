import cv2
from inference_sdk import InferenceHTTPClient
import numpy as np
import re
from collections import deque
import json

def parse_srt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        srt_content = file.read()

    # Dividir o conteúdo em blocos por frame
    frames = srt_content.strip().split('\n\n')
    frame_data = []

    for frame in frames:
        lines = frame.split('\n')
        
        # Extraindo o índice do frame
        frame_index = int(lines[0])
        
        # Extraindo o intervalo de tempo
        time_range = lines[1].strip()
        start_time, end_time = time_range.split(" --> ")

        # Extraindo o DiffTime
        match_difftime = re.search(r'DiffTime: (\d+)ms', lines[2])
        diff_time_ms = int(match_difftime.group(1))

        # Extraindo data e hora
        data_time = lines[3]

        # Extraindo dados
        matches = re.findall(r'\[(.*?)\]', lines[4])
        data = {}
        for match in matches:
            pairs = match.split()
            for i in range(0, len(pairs) - 1):
                if ':' in pairs[i]:
                    key = pairs[i].replace(":", "")
                    value = pairs[i+1]
                    data[key] = value
        
        frame_data.append({
                'frame_index': frame_index,
                'start_time': start_time,
                'end_time': end_time,
                'diff_time_ms': diff_time_ms,
                'data_time': data_time,
                **data  # Mesclar informações extraídas dos colchetes
            })

    return frame_data

def yaw_pitch_roll_to_rotation_matrix(yaw, pitch, roll):
    # Converter ângulos de graus para radianos
    yaw = np.radians(yaw)
    pitch = np.radians(pitch)
    roll = np.radians(roll)

    # Matrizes de rotação básicas
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0,            0,           1]
    ])

    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0,             1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    Rx = np.array([
        [1, 0,           0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])

    # Matriz de rotação composta: R = Rz * Ry * Rx
    R = Rz @ Ry @ Rx
    return R

def find_ground_intersection(lat, lon, alt, vec):

    # Descompactar vetor
    x, y, z = vec

    # Evitar divisão por zero no vetor
    if z == 0:
        raise ValueError("O vetor é paralelo ao solo e nunca tocará o chão.")

    # Calcular t (tempo escalar para atingir o solo)
    t = -alt / z

    # Coordenadas deslocadas no plano cartesiano
    x_t = t * x
    y_t = t * y

    # Conversão de deslocamento para latitude e longitude
    new_lat = lat + (y_t / 111320)
    new_lon = lon + (x_t / (111320 * np.cos(np.radians(lat))))

    return new_lat, new_lon

def find_ground_intersection_ECEF(lat, lon, alt, vec, earth_radius=6371000):
    """
    Encontra a latitude e longitude onde o vetor atinge o solo, considerando a curvatura da Terra.
    
    :param lat: Latitude inicial em graus
    :param lon: Longitude inicial em graus
    :param alt: Altitude inicial em metros
    :param vec: Vetor (x, y, z) representando a direção
    :param earth_radius: Raio da Terra em metros
    :return: Nova latitude e longitude em graus
    """
    # Converter latitude, longitude e altitude para coordenadas ECEF (Earth-Centered, Earth-Fixed)
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x0 = (earth_radius + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y0 = (earth_radius + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z0 = (earth_radius + alt) * np.sin(lat_rad)
    
    # Direção do vetor
    dx, dy, dz = vec
    
    # Resolver interseção do vetor com a superfície esférica da Terra
    # |P + t * D|^2 = R^2
    # P = (x0, y0, z0), D = (dx, dy, dz), R = earth_radius
    # Substituindo: (x0 + t*dx)^2 + (y0 + t*dy)^2 + (z0 + t*dz)^2 = R^2
    a = dx**2 + dy**2 + dz**2
    b = 2 * (x0 * dx + y0 * dy + z0 * dz)
    c = x0**2 + y0**2 + z0**2 - earth_radius**2

    # Resolver a equação quadrática
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        raise ValueError("O vetor não atinge a superfície da Terra.")

    # Escolher a menor solução positiva para t (interseção com o solo)
    t = (-b - np.sqrt(discriminant)) / (2 * a)
    if t < 0:
        raise ValueError("O vetor não aponta para a superfície da Terra.")

    # Coordenadas do ponto de interseção em ECEF
    xi = x0 + t * dx
    yi = y0 + t * dy
    zi = z0 + t * dz

    # Converter de ECEF de volta para latitude e longitude
    new_lat = np.degrees(np.arcsin(zi / earth_radius))
    new_lon = np.degrees(np.arctan2(yi, xi))

    return new_lat, new_lon

def reta3D(K_inv, R_t, t, pixel):
    pixel_RP2 = [pixel[0], pixel[1], 1 ]
    p0 = - R_t @ np.transpose(t)
    pv = R_t @ K_inv @ np.transpose(pixel_RP2)
    return (p0, pv)

def desenhar_centro(image, center_x, center_y, cor):
    line_length = 10
    
    # Desenhar a linha horizontal do '+'
    cv2.line(image, (int(center_x - line_length // 2), center_y), (int(center_x + line_length // 2), center_y),  cor, 2)  # Verde

    # Desenhar a linha vertical do '+'
    cv2.line(image, (center_x, int(center_y - line_length // 2)), (center_x, int(center_y + line_length // 2)),  cor, 2)

def print_on_pixel(image, label, x, y, cor):
    font_scale = 1  # Tamanho da fonte
    font_thickness = 2  # Espessura da fonte
    font = cv2.FONT_HERSHEY_SIMPLEX  # Fonte
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
    image_height, image_width, image_channels = image.shape
    text_x = x  # Alinhar à esquerda do retângulo
    text_y = y - baseline - 5  # Acima do retângulo (-5 para espaçamento)

    if text_y < 0:
        text_y = text_height + 5
    if text_x + text_width > image_width:  # Ultrapassa a borda direita
        text_x = image_width - text_width - 5  # Ajustar para a borda direita
    if text_x < 0:  # Ultrapassa a borda esquerda
        text_x = 5  # Ajustar para a borda esquerda


    cv2.putText(image, label, (text_x, text_y), font, font_scale, cor, font_thickness)

def mouse_click(event, x, y, flags, param):
    global clicks
    if event == cv2.EVENT_LBUTTONDOWN:  # Clique com o botão esquerdo
        original_x = int(x * scale_x)
        original_y = int(y * scale_y)
        clicks.append((original_x, original_y))
    elif event == cv2.EVENT_RBUTTONDOWN:  # Clique com o botão direito
        clicks.popleft()

with open("parameters.json", "r") as json_file:
    parameters = json.load(json_file)

K = [[2.85894134e+03, 0.00000000e+00, 1.97587073e+03],
 [0.00000000e+00, 2.85951996e+03, 1.49243924e+03],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
K_inv = np.linalg.inv(K)

project_id = "bus-detection-nrsmx"
model_version = 1
api_key = parameters["api_key"]
api_url = parameters["api_url"]

client = InferenceHTTPClient(api_url=api_url, api_key=api_key)

source = parameters["video_path"]
cap = cv2.VideoCapture(source)

frame_info = parse_srt(parameters["video_data_path"])
frame_index = 0

original_width = 3840
original_height = 2160
resized_width = parameters["resized_width"]
resized_height = parameters["resized_height"]
scale_x = original_width / resized_width
scale_y = original_height / resized_height
window_name = "Locate"

scale_reduct_inference = 6

clicks = deque(maxlen=5)

while True:
    ret, image = cap.read()
    if not ret:
        break

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('s'):
        continue

    yaw = float(frame_info[frame_index]['gb_yaw'])
    pitch = float(frame_info[frame_index]['gb_pitch'])
    roll = float(frame_info[frame_index]['gb_roll'])
    R_t = np.transpose(yaw_pitch_roll_to_rotation_matrix(yaw, pitch, roll))

    h = float(frame_info[frame_index]['rel_alt'])
    lat = float(frame_info[frame_index]['latitude'])
    long = float(frame_info[frame_index]['longitude'])

    for click in clicks:
        desenhar_centro(image, click[0], click[1], (255, 0, 0))
        reta = reta3D(K_inv, R_t, np.zeros(3), (click[0], click[1]))
        click_lat_long = find_ground_intersection(lat, long, h, reta[1])
        print_on_pixel(image, f"LAT:{click_lat_long[0]}, LONG:{click_lat_long[1]}", click[0], click[1], (255, 0, 0))

    short_image = cv2.resize(image, (int(original_width / scale_reduct_inference), int(original_height / scale_reduct_inference)))
    results = client.infer(short_image, model_id=f"{project_id}/{model_version}")

    for prediction in results['predictions']:
                        
        width, height = int(prediction['width'] * scale_reduct_inference), int(prediction['height'] * scale_reduct_inference)
        prediction_x = int(prediction['x'] * scale_reduct_inference)
        prediction_y = int(prediction['y'] * scale_reduct_inference)

        x, y = int(prediction_x - width/2) , int(prediction_y - height/2)
        
        class_id = prediction['class_id']

        # Calculate the bottom right x and y coordinates
        x2 = int(x + width)
        y2 = int(y + height)

        if class_id == 0:
            cv2.rectangle(image, (x, y), (x2, y2), (0, 0, 255), 3)
            desenhar_centro(image, int(prediction_x), int(prediction_y), (0, 0, 255))

            reta = reta3D(K_inv, R_t, np.zeros(3), (prediction_x, prediction_y))
            pred_lat_long = find_ground_intersection(lat, long, h, reta[1])
            print_on_pixel(image, f"LAT:{pred_lat_long[0]}, LONG:{pred_lat_long[1]}", x, y, (0, 0, 255))
            

    rez_img = cv2.resize(image, (resized_width, resized_height))
    cv2.imshow(window_name, rez_img)
    cv2.setMouseCallback(window_name, mouse_click)
    frame_index += 1