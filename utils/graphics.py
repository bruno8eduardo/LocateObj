import cv2
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *

class Graphics:

    def __init__(self, geometry, near = 0.1, far = 1000.0, cone_height = 5.0, cone_radius = 1.5, glMode = True):
        self.cameraToOpenglR = geometry.cameraToOpenglR
        self.near = near
        self.far = far
        self.cone_height = cone_height
        self.cone_radius = cone_radius
        self.glMode = glMode

    

    def build_projection_matrix(self, K, width, height):
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        proj = np.zeros((4, 4))
        proj[0, 0] = 2 * fx / width
        proj[1, 1] = 2 * fy / height
        proj[0, 2] = 1 - (2 * cx / width)
        proj[1, 2] = 1 - (2 * cy / height)
        proj[2, 2] = -(self.far + self.near) / (self.far - self.near)
        proj[2, 3] = -2 * self.far * self.near / (self.far - self.near)
        proj[3, 2] = -1
        return proj

    def build_view_matrix(self, R, t):
        """ Converte R e t para uma matriz de visualização do OpenGL. """
        R_T = np.transpose(R)
        Rt = np.concatenate((R_T, -R_T @ t), axis=1)
        view = np.eye(4)  # Matriz identidade 4x4
        view[:3, :4] = Rt  # Insere [R | t] na matriz 4x4
        return view

    def draw_cone_sphere(self, x, y, z, pitch, color):

        color_array = [0.0, 0.0, 0.0, 1.0]
        if color == "red":
            color_array = [1.0, 0.0, 0.0, 1.0]
        elif color == "blue":
            color_array = [0.0, 0.0, 1.0, 1.0]
        elif color == "green":
            color_array = [0.0, 1.0, 0.0, 1.0]
        elif color == "black":
            color_array = [0.1, 0.1, 0.1, 1.0]

        # Esfera vermelha
        glMaterialfv(GL_FRONT, GL_AMBIENT, color_array)
        glMaterialfv(GL_FRONT, GL_DIFFUSE, color_array)
        glMaterialfv(GL_FRONT, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        glMaterialf(GL_FRONT, GL_SHININESS, 50.0)

        sphere_quadric = gluNewQuadric()
        glPushMatrix()
        glTranslatef(x, y, z)  # **Posicionar no local correto**
        gluSphere(sphere_quadric, self.cone_radius, 20, 20)
        glPopMatrix()

        # Desabilitar o plano de corte
        glDisable(GL_CLIP_PLANE0)

        # Cone
        cone_quadric = gluNewQuadric()
        glPushMatrix()
        glTranslatef(x, y, z)  # **Mesmo posicionamento para o cone**
        glRotatef(90 - pitch, 1, 0, 0)
        gluCylinder(cone_quadric, self.cone_radius, 0, self.cone_height, 20, 20)
        glPopMatrix()

    def render(self, draw_func):

        if self.glMode:
            glLoadIdentity()
            draw_func()

    def draw_opengl(self, pixels_opengl, imagem_fundo):
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

    def instantiate(self, image, K, R, t, point, color, t_drone_ENU, pitch):
        pixel = K @ np.concatenate((R, t), axis=1) @ np.vstack((point, [1]))
        pixel = pixel.flatten()
        pixel = pixel / pixel[2]
        if color == "red":
            colorN = (0,0,255)
        elif color == "black":
            colorN = (0,0,0)
        elif color == "blue":
            colorN = (255,0,0)
        elif color == "green":
            colorN = (0,255,0)

        self.desenhar_centro(image, int(pixel[0]), int(pixel[1]), colorN)
        self.print_on_pixel(image, f"N:{point[1,0]:.3f}, E:{point[0,0]:.3f}, Up: {point[2,0]:.3f}", int(pixel[0]), int(pixel[1]), colorN)
        t_opengl = self.cameraToOpenglR @ R @ (point - t_drone_ENU + [[0],[0],[self.cone_height]])
        view_matrix = self.build_view_matrix(self.cameraToOpenglR @ R, np.array([[0],[0],[0]]))
        glMatrixMode(GL_MODELVIEW)
        glLoadMatrixf(view_matrix)
        self.render(lambda: self.draw_cone_sphere(t_opengl[0,0], t_opengl[1,0], t_opengl[2,0], pitch, color))


    def desenhar_centro(self, image, center_x, center_y, cor, roi_flag=False):
        if (not self.glMode) or roi_flag:
            line_length = 10
            
            # Desenhar a linha horizontal do '+'
            cv2.line(image, (int(center_x - line_length // 2), center_y), (int(center_x + line_length // 2), center_y),  cor, 2)  # Verde

            # Desenhar a linha vertical do '+'
            cv2.line(image, (center_x, int(center_y - line_length // 2)), (center_x, int(center_y + line_length // 2)),  cor, 2)

    def print_on_pixel(self, image, label, x, y, cor):
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
