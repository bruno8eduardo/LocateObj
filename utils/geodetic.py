import numpy as np
from utils.geometry import Geometry

class Geodetic:

    def __init__(self, dem_elevation_data, dem_transform, dem_crs, dem_interception_epsilon = 0.01, dem_interception_count = 50):
        self.dem_elevation_data = dem_elevation_data
        self.dem_transform = dem_transform
        self.dem_crs = dem_crs

        self.dem_interception_epsilon = dem_interception_epsilon
        self.dem_interception_count = dem_interception_count

    def get_DEM_alt(self, east_utm, north_utm):
        row, col = ~self.dem_transform * (east_utm, north_utm)
        row = int(round(row))
        col = int(round(col))
        if 0 <= row < self.dem_elevation_data.shape[0] and 0 <= col < self.dem_elevation_data.shape[1]:
            return self.dem_elevation_data[row, col]
        else:
            return None  # Fora da imagem
    
    def get_intersection_from_click(self, click, K_inv, R_drone, t_drone_mundo, dem_elevation_data, h_abs, h0, utm0_x, utm0_y, h_dem_offset):
        easting = t_drone_mundo[0,0]
        northing = t_drone_mundo[1,0]
        h_enu = t_drone_mundo[2,0]
        reta = Geometry.reta3D(K_inv, Geometry.droneToMundoR @ R_drone @ Geometry.cameraToDroneR, t_drone_mundo, (click[0], click[1]))
        # click_ENU = find_ground_intersection_ENU(northing, easting, h_enu, reta[1])
        vec_DEM = Geometry.norm_vec(reta[1].flatten())
        if vec_DEM[2] < 0:
            vec_DEM = (-1) * vec_DEM
        if dem_elevation_data is not None:
            click_ENU = self.find_DEM_intersection(easting + utm0_x, northing + utm0_y, h_abs - h_dem_offset, vec_DEM)
        else:
            click_ENU = self.find_ground_intersection_ENU(northing, easting, h_enu, vec_DEM)
        
        if click_ENU is not None:
            if dem_elevation_data is not None:
                click_ENU[0,0] -= utm0_x
                click_ENU[1,0] -= utm0_y
                click_ENU[2,0] += h_dem_offset - h0
        
        return click_ENU


    def find_DEM_intersection(self, utm_east, utm_north, utm_up, vec_flat_norm):
        count = 0
        while True:
            alt = self.get_DEM_alt(utm_east, utm_north)
            if alt is None:
                return None
            
            gap = utm_up - alt
            if np.abs(gap) <= self.dem_interception_epsilon:
                return np.array([[utm_east], [utm_north], [utm_up]])
            
            vec = gap * vec_flat_norm
            utm_east -= vec[0]
            utm_north -= vec[1]
            utm_up -= vec[2]
            
            count += 1
            if count > self.dem_interception_count:
                return None

    def find_ground_intersection(self, lat, lon, alt, vec):

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

    def find_ground_intersection_UTM(self, north, east, alt_rel, alt_abs, vec):

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

    def find_ground_intersection_ENU(self, north, east, alt, vec):

        # Descompactar vetor
        x = vec[0]
        y = vec[1]
        z = vec[2]

        # Evitar divisão por zero no vetor
        if z == 0:
            raise ValueError("O vetor é paralelo ao solo e nunca tocará o chão.")

        # Calcular t (tempo escalar para atingir o solo)
        t = -alt / z

        # Coordenadas deslocadas no plano cartesiano
        x_t = t * x
        y_t = t * y

        # Conversão de deslocamento
        new_north = north + y_t
        new_east = east + x_t

        return np.array([[new_east], [new_north], [0]])

    def find_ground_intersection_ECEF(self, lat, lon, alt, vec, earth_radius=6371000):
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
