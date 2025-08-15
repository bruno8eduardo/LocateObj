import numpy as np
import cv2

class Geometry:

    def __init__(self, minimal_distance_param = 0.01):
        self.minimal_distance_param = minimal_distance_param

        self.droneToMundoR = np.array([[0,1,0],[1,0,0],[0,0,-1]])
        self.mundoToDroneR = np.transpose(self.droneToMundoR)
        self.cameraToDroneR = np.array([[0,0,1],[1,0,0],[0,1,0]])
        self.droneToCameraR = np.transpose(self.cameraToDroneR)
        self.cameraToMundoR = np.array([[1,0,0],[0,0,1],[0,-1,0]])
        self.mundoToCameraR = np.transpose(self.cameraToMundoR)
        self.cameraToOpenglR = np.array([[1,0,0],[0,-1,0],[0,0,-1]])

    def inv_K(self, K):
        fx = K[0][0]
        fy = K[1][1]
        cx = K[0][2]
        cy = K[1][2]
        K_inv = np.array([[1/fx, 0, -cx/fx],
                [0, 1/fy, -cy/fy],
                [0, 0, 1]])
        return K_inv

    def norm_vec(self, v):
        v_copy = v.copy()
        norm_v = v_copy / np.linalg.norm(v_copy)
        return norm_v


    def yaw_pitch_roll_to_rotation_matrix(self, yaw, pitch, roll):
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


    def reta3D(self, K_inv, R_t, t, pixel):
        pixel_RP2 = np.array([[pixel[0]], [pixel[1]], [1]])
        p0 = - R_t @ t
        pv = R_t @ K_inv @ pixel_RP2
        return (p0, pv)

    def get_homography(self, frame_base, frame_obj, detector, matcher):
        kp_base, des_base = detector.detectAndCompute(frame_base, None)
        kp_obj, des_obj = detector.detectAndCompute(frame_obj, None)
        
        matches = matcher.match(des_base, des_obj)
        matches = sorted(matches, key=lambda x: x.distance)
        num_matches = 50
        matches = matches[:num_matches]

        src_pts = np.float32([kp_base[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp_obj[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        return H

    def distance_is_minimal(self, east_0, north_0, h_0, east_1, north_1, h_1):
        point_0 = np.array([east_0, north_0, h_0])
        point_1 = np.array([east_1, north_1, h_1])
        distance = np.linalg.norm(point_1 - point_0)
        if distance < h_1 * self.minimal_distance_param:
            return True
        else:
            return False

    def get_R_one_roi(self, roi_enu, roi_pixel, R, K_inv, t_drone_ENU):    
        theta_1, R_1 = get_rotation_from_vectors(R @ (roi_enu - t_drone_ENU), K_inv @ roi_pixel)
        theta_2, R_2 = get_rotation_from_vectors(R @ (roi_enu - t_drone_ENU), - K_inv @ roi_pixel)
        if theta_1 <= theta_2:
            R_corr = R_1
        else:
            R_corr = R_2
        
        return R_corr @ R

    def get_R_roi(self, roi_enus, roi_pixels, K_inv, t_drone_ENU):
        if len(roi_enus) > 2:
            print("CORRECAO PARA 3 OU MAIS ROI AINDA A IMPLEMENTAR")
            print("CONSIDERAMOS SOMENTE OS DOIS PRIMEIROS ROI")
        
        a_list = [(lambda a: norm_vec(a - t_drone_ENU))(a) for a in roi_enus]
        b_list = [(lambda b: norm_vec(K_inv @ b))(b) for b in roi_pixels]

        u_0 = a_list[0]
        u_1 = norm_vec(a_list[1] - (np.dot(a_list[1].copy().flatten(), u_0.copy().flatten())) * u_0)
        u_2_flat = np.cross(u_0.copy().flatten(), u_1.copy().flatten())
        u_2 = np.array([[u_2_flat[0]],[u_2_flat[1]],[u_2_flat[2]]])
        
        v_0 = b_list[0]
        v_1 = norm_vec(b_list[1] - (np.dot(b_list[1].copy().flatten(), v_0.copy().flatten())) * v_0)
        v_2_flat = np.cross(v_0.copy().flatten(), v_1.copy().flatten())
        v_2 = np.array([[v_2_flat[0]],[v_2_flat[1]],[v_2_flat[2]]])

        A = np.concatenate((u_0, u_1, u_2), axis=1)
        B = np.concatenate((v_0, v_1, v_2), axis=1)

        return B @ np.transpose(A)

    # R x = b
    def get_rotation_from_vectors(self, x, b):
        x_norm = norm_vec(x.flatten())
        b_norm = norm_vec(b.flatten())
        v = np.cross(x_norm, b_norm)
        v = norm_vec(v)
        theta = np.arccos(np.clip(np.dot(x_norm, b_norm), -1.0, 1.0))
        theta_op = 2 * np.pi - theta
        if theta <= theta_op:
            rot_vec = theta * v
            R_theta, _ = cv2.Rodrigues(rot_vec)
            return theta, R_theta
        else:
            rot_vec = theta_op * v
            R_theta, _ = cv2.Rodrigues(rot_vec)
            return theta_op, R_theta
