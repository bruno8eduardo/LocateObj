import zmq
import json
import cv2
from utils.geometry import Geometry
import numpy as np
import threading
from collections import deque

class ReconstructionClient:

    connectionString = "tcp://127.0.0.1:5555"
    context = zmq.Context()

    max_items = 7

    img_list = deque(maxlen=max_items)
    t_world_list = deque(maxlen=max_items)
    R_list = deque(maxlen=max_items)

    lat0 = None
    lon0 = None
    h0 = None

    geodetic = None

    @staticmethod
    def choose_frames_for_rec(img, t_world, R, lat0, lon0, h0, frame_index=None):

        img = cv2.imencode(".png", img)[1]
        img = img.tobytes()

        if len(ReconstructionClient.img_list) == 0:
            ReconstructionClient.img_list.append(img)
            ReconstructionClient.t_world_list.append(t_world)
            ReconstructionClient.R_list.append(R)
            return True, False

        for i in range(0, len(ReconstructionClient.img_list)):

            distance = np.linalg.norm(t_world - ReconstructionClient.t_world_list[i])
            rec_h = ReconstructionClient.t_world_list[i][2,0]
            distance_test = distance > 0.2 * rec_h
            if not distance_test:
                return False, False
        
        ReconstructionClient.img_list.append(img)
        ReconstructionClient.t_world_list.append(t_world)
        ReconstructionClient.R_list.append(R)

        if len(ReconstructionClient.img_list) > 3:
            try:
                t = threading.Thread(target=ReconstructionClient.send_images_with_metadata, args=(lat0, lon0, h0, frame_index,))
                t.daemon = True
                t.start()
            except Exception as e:
                print(f"Error starting thread: {e}")
            return True, True

        return True, False


    @staticmethod
    def send_images_with_metadata(lat0, lon0, h0, index=None):

        socket = ReconstructionClient.context.socket(zmq.REQ)
        socket.connect(ReconstructionClient.connectionString)

        meta = {
            "num_images": len(ReconstructionClient.img_list),
            "R_drone_list": [r.tolist() for r in ReconstructionClient.R_list],
            "T_drone_list": [t.tolist() for t in ReconstructionClient.t_world_list],
            "lat0": lat0,
            "lon0": lon0,
            "h0": h0
        }

        # Prepare multipart message: first part is meta, rest are images
        msg_parts = [json.dumps(meta).encode()] + list(ReconstructionClient.img_list)
        socket.send_multipart(msg_parts)

        # Recebe resposta
        reply = socket.recv_string()
        print(f"Index {index}:", reply)

        if ReconstructionClient.geodetic is not None:
            t = threading.Thread(target=ReconstructionClient.geodetic.update_ENU_DEM)
            t.daemon = True
            t.start()