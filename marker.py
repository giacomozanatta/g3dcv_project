import numpy as np

Marker_0 = np.array([
                        (70, 0, 0),     # A
                        (65, 5, 0),     # B
                        (98, 5, 0),     # C
                        (98, -5, 0),    # D
                        (65, -5, 0),    # E
                      ], dtype="double")

Marker_0_2D = np.array([
                        (70, 0),     # A
                        (65, 5),     # B
                        (98, 5),     # C
                        (98, -5),    # D
                        (65, -5),    # E
                      ], dtype="double")

Marker_circles_0_2D = np.array([
                        (75, 0), #D1
                        (79.5, 0), #D2
                        (84, 0), #D3
                        (88.5, 0), #D4
                        (93, 0), #D5
                    ], dtype="double")              
Marker_circles_0 = np.array([
                                (75, 0, 0), #D1
                                (79.5, 0, 0), #D2
                                (84, 0, 0), #D3
                                (88.5, 0, 0), #D4
                                (93, 0, 0), #D5
                            ], dtype="double")

mag_A = np.linalg.norm(Marker_0[0])
mag_B = np.linalg.norm(Marker_0[1])
mag_C = np.linalg.norm(Marker_0[2])
mag_D = np.linalg.norm(Marker_0[3])
mag_E = np.linalg.norm(Marker_0[4])




angle_A = 0
angle_B = np.degrees(np.arctan(Marker_0[1][1] / Marker_0[1][0]))
angle_C = np.degrees(np.arctan(Marker_0[2][1] / Marker_0[2][0]))
angle_D = np.degrees(np.arctan(Marker_0[3][1] / Marker_0[3][0]))
angle_E = np.degrees(np.arctan(Marker_0[4][1] / Marker_0[4][0]))

angle_circles = 0

def get_marker_position(position):
    _angle_A = angle_A + (-15 * position)
    _angle_B = angle_B + (-15 * position)
    _angle_C = angle_C + (-15 * position)
    _angle_D = angle_D + (-15 * position)
    _angle_E = angle_E + (-15 * position)

    return np.array([
        (round(mag_A * (np.cos(np.radians(_angle_A)))), round(mag_A * (np.sin(np.radians(_angle_A)))), 0),  # A
        (round(mag_B * (np.cos(np.radians(_angle_B)))), round(mag_B * (np.sin(np.radians(_angle_B)))), 0),  # B
        (round(mag_C * (np.cos(np.radians(_angle_C)))), round(mag_C * (np.sin(np.radians(_angle_C)))), 0),  # C
        (round(mag_D * (np.cos(np.radians(_angle_D)))), round(mag_D * (np.sin(np.radians(_angle_D)))), 0),  # D
        (round(mag_E * (np.cos(np.radians(_angle_E)))), round(mag_E * (np.sin(np.radians(_angle_E)))), 0),  # E
    ], dtype="double")
