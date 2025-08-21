import tkinter as tk
from tkinter import simpledialog
import cv2

def get_roi_data(i):
    root = tk.Tk()
    root.withdraw()
    lat_roi = simpledialog.askfloat(f"Entrada de dados {i}", f"Insira LATITUDE do ROI {i}: ")
    long_roi = simpledialog.askfloat(f"Entrada de dados {i}", f"Insira LONGITUDE do ROI {i}: ")
    h_abs_roi = simpledialog.askfloat(f"Entrada de dados {i}", f"Insira ALTITUDE do ROI {i} em relação ao nível do mar: ")
    return lat_roi, long_roi, h_abs_roi

def mouse_click(event, x, y, flags, param):
    clicks, clicks_ENU = param
    if event == cv2.EVENT_LBUTTONDOWN:  # Clique com o botão esquerdo
        original_x = int(x)
        original_y = int(y)
        clicks.append((original_x, original_y))
    elif event == cv2.EVENT_RBUTTONDOWN:  # Clique com o botão direito
        if (len(clicks_ENU) > 0):
            clicks_ENU.popleft()
