import cv2
from inference_sdk import InferenceHTTPClient
import numpy as np
import re
from collections import deque
import json
import utm
import glfw
from OpenGL.GL import *
from OpenGL.GLU import *

droneToMundoR = np.array([[0,1,0],[1,0,0],[0,0,-1]])
mundoToDroneR = np.transpose(droneToMundoR)
cameraToDroneR = np.array([[0,0,1],[1,0,0],[0,1,0]])
droneToCameraR = np.transpose(cameraToDroneR)
cameraToMundoR = np.array([[1,0,0],[0,0,1],[0,-1,0]])
mundoToCameraR = np.transpose(cameraToMundoR)
cameraToOpenglR = np.array([[1,0,0],[0,-1,0],[0,0,-1]])

near = 0.1
far = 1000.0
cone_height = 5.0
cone_radius = 1.5

def inv_K(K):
    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]
    K_inv = np.array([[1/fx, 0, -cx/fx],
             [0, 1/fy, -cy/fy],
             [0, 0, 1]])
    return K_inv

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

def find_ground_intersection_UTM(north, east, alt_rel, alt_abs, vec):

    # Descompactar vetor
    x = vec[0,0]
    y = vec[1,0]
    z = vec[2,0]

    # Evitar divisão por zero no vetor
    if z == 0:
        raise ValueError("O vetor é paralelo ao solo e nunca tocará o chão.")

    # Calcular t (tempo escalar para atingir o solo)
    t = -alt_rel / z

    # Coordenadas deslocadas no plano cartesiano
    x_t = t * x
    y_t = t * y

    # Conversão de deslocamento para UTM
    new_north = north + y_t
    new_east = east + x_t

    return np.array([[new_east], [new_north], [alt_abs - alt_rel]])

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
    pixel_RP2 = np.array([[pixel[0]], [pixel[1]], [1]])
    p0 = - R_t @ t
    pv = R_t @ K_inv @ pixel_RP2
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
    clicks, clicks_UTM = param
    if event == cv2.EVENT_LBUTTONDOWN:  # Clique com o botão esquerdo
        original_x = int(x * scale_x)
        original_y = int(y * scale_y)
        clicks.append((original_x, original_y))
    elif event == cv2.EVENT_RBUTTONDOWN:  # Clique com o botão direito
        if (len(clicks_UTM) > 0):
            clicks_UTM.popleft()

def build_projection_matrix(K, width, height, near=near, far=far):
    """ Converte a matriz K para o formato de projeção do OpenGL. """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    proj = np.zeros((4, 4))
    proj[0, 0] = 2 * fx / width
    proj[1, 1] = 2 * fy / height
    proj[0, 2] = 1 - (2 * cx / width)
    proj[1, 2] = (2 * cy / height) - 1
    proj[2, 2] = -(far + near) / (far - near)
    proj[2, 3] = -2 * far * near / (far - near)
    proj[3, 2] = -1
    return proj

def build_view_matrix(R, t):
    """ Converte R e t para uma matriz de visualização do OpenGL. """
    Rt = np.concatenate((R, t), axis=1)
    view = np.eye(4)  # Matriz identidade 4x4
    view[:3, :4] = Rt  # Insere [R | t] na matriz 4x4
    return view

def draw_cone_sphere(x, y, z, pitch, color):

    color_array = [0.0, 0.0, 0.0, 1.0]
    if color == "red":
        color_array = [1.0, 0.0, 0.0, 1.0]
    elif color == "blue":
        color_array = [0.0, 0.0, 1.0, 1.0]

    # Esfera vermelha
    glMaterialfv(GL_FRONT, GL_AMBIENT, color_array)
    glMaterialfv(GL_FRONT, GL_DIFFUSE, color_array)
    glMaterialfv(GL_FRONT, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
    glMaterialf(GL_FRONT, GL_SHININESS, 50.0)

    sphere_quadric = gluNewQuadric()
    glPushMatrix()
    glTranslatef(x, y, z)  # **Posicionar no local correto**
    gluSphere(sphere_quadric, cone_radius, 20, 20)
    glPopMatrix()

    # Desabilitar o plano de corte
    glDisable(GL_CLIP_PLANE0)

    # Cone
    cone_quadric = gluNewQuadric()
    glPushMatrix()
    glTranslatef(x, y, z)  # **Mesmo posicionamento para o cone**
    glRotatef(90 - pitch, 1, 0, 0)
    gluCylinder(cone_quadric, cone_radius, 0, cone_height, 20, 20)
    glPopMatrix()

def render(draw_func):
    
    glLoadIdentity()
    draw_func()

    glfw.swap_buffers(window)

def draw_opengl(pixels_opengl, imagem_fundo):
    # Capturar a tela do OpenGL
    imagem_renderizada = np.frombuffer(pixels_opengl, dtype=np.uint8).reshape(1080, 1920, 3)
    imagem_renderizada = cv2.flip(imagem_renderizada, 0)
    imagem_renderizada = cv2.cvtColor(imagem_renderizada, cv2.COLOR_RGB2BGR)  # Converter RGB → BGR

    # Criar uma máscara onde os pixels pretos indicam transparência
    gray = cv2.cvtColor(imagem_renderizada, cv2.COLOR_BGR2GRAY)  # Converter para tons de cinza
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)  # Criar máscara: 0 para preto, 255 para o resto

    # Inverter a máscara para pegar apenas o fundo
    mask_inv = cv2.bitwise_not(mask)

    # Criar uma versão da imagem de fundo com buraco onde os objetos estão
    fundo_com_buraco = cv2.bitwise_and(imagem_fundo, imagem_fundo, mask=mask_inv)

    # Criar uma versão da renderização que mantém apenas os objetos
    objetos_renderizados = cv2.bitwise_and(imagem_renderizada, imagem_renderizada, mask=mask)

    # Combinar as duas imagens corretamente
    resultado = cv2.add(fundo_com_buraco, objetos_renderizados)
    return resultado

with open("parameters.json", "r") as json_file:
    parameters = json.load(json_file)

K_path = parameters["K_path"]
with open(K_path, "r") as json_file:
    K = np.array(json.load(json_file), dtype=np.float64)

# Inicializar GLFW
if not glfw.init():
    raise Exception("GLFW não pôde ser inicializado!")

# Criar janela OpenGL
window = glfw.create_window(1920, 1080, "Render 3D", None, None)
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
proj_matrix = build_projection_matrix(K, 1920, 1080)
glMatrixMode(GL_PROJECTION)
glLoadMatrixf(np.transpose(proj_matrix))

K_inv = inv_K(K)

project_id = "car-models-rr7w5"
model_version = 1
api_key = parameters["api_key"]
api_url = parameters["api_url"]

client = InferenceHTTPClient(api_url=api_url, api_key=api_key)

source = parameters["video_path"]
cap = cv2.VideoCapture(source)

frame_info = parse_srt(parameters["video_data_path"])
frame_index = 0

original_width = 1920
original_height = 1080
resized_width = parameters["resized_width"]
resized_height = parameters["resized_height"]
scale_x = original_width / resized_width
scale_y = original_height / resized_height
window_name = "Locate"

scale_reduct_inference = 6

clicks = deque(maxlen=5)
clicks_UTM = deque(maxlen=5)
# Localizacao carro: [latitude: -22.905551] [longitude: -43.221218] [rel_alt: 2.847 abs_alt: 15.331]
car_x, car_y, car_zn, car_zl = utm.from_latlon(-22.905551, -43.221218)
car_z = 15.331 - 2.847
t_car_mundo = np.array([[car_x],[car_y],[car_z]])

images = []
while not glfw.window_should_close(window):
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    ret, image = cap.read()
    if ret:
        images.append(image)
        frame_index += 1

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
    
    image = images[frame_index - 1 if frame_index > 0 else 0]

    yaw = float(frame_info[frame_index]['gb_yaw'])
    pitch = float(frame_info[frame_index]['gb_pitch'])
    roll = float(frame_info[frame_index]['gb_roll'])
    R_drone = yaw_pitch_roll_to_rotation_matrix(yaw, pitch, roll)
    R_drone_T = np.transpose(R_drone)

    h = float(frame_info[frame_index]['rel_alt'])
    h_abs = float(frame_info[frame_index]['abs_alt'])
    lat = float(frame_info[frame_index]['latitude'])
    long = float(frame_info[frame_index]['longitude'])

    easting, northing, zone_number, zone_letter = utm.from_latlon(lat, long)
    t_drone_mundo = np.array([[easting], [northing], [h_abs]])
    print_on_pixel(image, f"index:{frame_index}, N:{int(northing)}, E:{int(easting)}, h_rel:{h}, yaw:{yaw}, pitch:{pitch}, roll:{roll}", 10, 10, (0,0,0))

    R = droneToCameraR @ R_drone_T @ mundoToDroneR
    t =  - droneToCameraR @ R_drone_T @ mundoToDroneR @ t_drone_mundo
    view_matrix = build_view_matrix(cameraToOpenglR @ R, np.array([[0],[0],[0]]))
    glMatrixMode(GL_MODELVIEW)
    glLoadMatrixf(np.transpose(view_matrix))

    t_car_opengl = cameraToOpenglR @ droneToCameraR @ R_drone_T @ mundoToDroneR @ (t_car_mundo - t_drone_mundo + [[0],[0],[cone_height]])

    render(lambda: draw_cone_sphere(t_car_opengl[0,0], t_car_opengl[1,0], t_car_opengl[2,0], pitch, "red"))
    glfw.poll_events()

    for click in clicks:
        reta = reta3D(K_inv, droneToMundoR @ R_drone @ cameraToDroneR, t_drone_mundo, (click[0], click[1]))
        click_UTM = find_ground_intersection_UTM(northing, easting, h, h_abs, reta[1])
        clicks_UTM.append(click_UTM)

    clicks.clear()
    clicks_UTM_copy = clicks_UTM.copy()

    for utm_click in clicks_UTM_copy:
        pixel_click = K @ np.concatenate((R, t), axis=1) @ np.vstack((utm_click, [1]))
        pixel_click = pixel_click.flatten()
        pixel_click = pixel_click / pixel_click[2]
        if pixel_click[0] >= 0 and pixel_click[0] <= original_width and pixel_click[1] >= 0 and pixel_click[1] <= original_height:
            desenhar_centro(image, int(pixel_click[0]), int(pixel_click[1]), (255, 0, 0))
            print_on_pixel(image, f"N:{utm_click[1]}, E:{utm_click[0]}, ZN:{zone_number}, ZL:{zone_letter}", int(pixel_click[0]), int(pixel_click[1]), (255, 0, 0))
            t_click_opengl = cameraToOpenglR @ droneToCameraR @ R_drone_T @ mundoToDroneR @ (utm_click - t_drone_mundo + [[0],[0],[cone_height]])
            render(lambda: draw_cone_sphere(t_click_opengl[0,0], t_click_opengl[1,0], t_click_opengl[2,0], pitch, "blue"))
            glfw.poll_events()
    
    pixels = glReadPixels(0, 0, 1920, 1080, GL_RGB, GL_UNSIGNED_BYTE)
    image = draw_opengl(pixels, image)
    
    # short_image = cv2.resize(image, (int(original_width / scale_reduct_inference), int(original_height / scale_reduct_inference)))
    # results = client.infer(short_image, model_id=f"{project_id}/{model_version}")

    # for prediction in results['predictions']:
                        
    #     width, height = int(prediction['width'] * scale_reduct_inference), int(prediction['height'] * scale_reduct_inference)
    #     prediction_x = int(prediction['x'] * scale_reduct_inference)
    #     prediction_y = int(prediction['y'] * scale_reduct_inference)

    #     x, y = int(prediction_x - width/2) , int(prediction_y - height/2)
        
    #     class_id = prediction['class_id']

    #     # Calculate the bottom right x and y coordinates
    #     x2 = int(x + width)
    #     y2 = int(y + height)

    #     if class_id == 0:
    #         cv2.rectangle(image, (x, y), (x2, y2), (0, 0, 255), 3)
    #         desenhar_centro(image, int(prediction_x), int(prediction_y), (0, 0, 255))

    #         reta = reta3D(K_inv, droneToMundoR @ R_drone @ cameraToDroneR, t_drone_mundo, (prediction_x, prediction_y))
    #         pred_UTM = find_ground_intersection_UTM(northing, easting, h, h_abs, reta[1])
    #         print_on_pixel(image, f"N:{pred_UTM[1]}, E:{pred_UTM[0]}, ZN:{zone_number}, ZL:{zone_letter}", x, y, (0, 0, 255))
    
    pixel_car = K @ np.concatenate((R, t), axis=1) @ np.vstack((t_car_mundo, [1]))
    pixel_car = pixel_car.flatten()
    pixel_car = pixel_car / pixel_car[2]
    desenhar_centro(image, int(pixel_car[0] / scale_x), int(pixel_car[1] / scale_y), (0,0,255))
    print_on_pixel(image, f"N:{t_car_mundo[1,0]}, E:{t_car_mundo[0,0]}, ZN:{car_zn}, ZL:{car_zl}", int(pixel_car[0] / scale_x), int(pixel_car[1] / scale_y), (0,0,255))

    rez_img = cv2.resize(image, (resized_width, resized_height))
    cv2.imshow(window_name, rez_img)
    cv2.setMouseCallback(window_name, mouse_click, (clicks, clicks_UTM))