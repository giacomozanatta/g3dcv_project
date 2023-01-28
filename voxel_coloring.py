from backgroundremoval import *
from poseestimation import *
from undistort import *
import pickle
import configs
from lib.voxel import *
from lib.videocapture import *

x_axis = np.float32([[0,0,0], [-50,0,0]]).reshape(-1,3)

def voxel_set_to_3D_matrix(voxel_set):
    m = {}
    for i in range(0, len(voxel_set.set)):
        if not voxel_set.set[i][0] in m:
             m[voxel_set.set[i][0]] = {}
        if not voxel_set.set[i][1] in m[voxel_set.set[i][0]]:
            m[voxel_set.set[i][0]][voxel_set.set[i][1]] = {}
        m[voxel_set.set[i][0]][voxel_set.set[i][1]][voxel_set.set[i][2]] = {
            'R': 0,
            'G': 0,
            'B': 0,
            'acc': 0
        }
    return m
voxel_set = None
print('[INFO] Loading voxel_set from file')
with open(r'output/voxels_' + configs.working_object + '.pkl', 'rb') as f:
    voxel_set = pickle.load(f)

print('[INFO] voxel_set loaded. Number of voxels: ' + str(len(voxel_set.set)))

M = voxel_set_to_3D_matrix(voxel_set)

dist = np.load("output/dist.npy")
K = np.load("output/K.npy")

def get_min_y_given_xz(M, x,z):
        for _y in M[x].keys():
            if z in M[x][_y]:
                return _y
def get_max_y_given_xz(M, x,z):
        for _y in reversed(M[x].keys()):
            if z in M[x][_y]:
                return _y

def get_min_x_given_yz(M, y, z):
    for _x in M.keys():
        if y in M[_x]:
            if z in M[_x][y]:
                return _x

def get_max_x_given_yz(M, y, z):
    for _x in reversed(M.keys()):
        if y in M[_x]:
            if z in M[_x][y]:
                return _x
def save_ply(name, data, M):
    offset = data.offset
    with open('output/' + name + '.ply', 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('comment made by Giacomo Zanatta\n')
        f.write('element vertex ' + str(len(data.set) * 8) + '\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('element face ' + str(len(data.set) * 6) +'\n')
        f.write('property list uchar int vertex_index\n')
        f.write('end_header\n')
        for point in data.set:
            acc = 1
            M_entry = M[point[0]][point[1]][point[2]]
            # add coloring
            if M_entry['acc'] > 0:
                acc = M_entry['acc']
            f.write(str(point[0]-(offset/2)) + ' ' + str(point[1]-(offset/2)) + ' ' + str(point[2]-(offset/2)) + ' ' + str(int(M_entry['R']/acc)) + ' ' + str(int(M_entry['G']/acc)) + ' ' + str(int(M_entry['B']/acc)) + '\n')
            f.write(str(point[0]-(offset/2)) + ' ' + str(point[1]-(offset/2)) + ' ' + str(point[2]+(offset/2)) + ' ' + str(int(M_entry['R']/acc)) + ' ' + str(int(M_entry['G']/acc)) + ' ' + str(int(M_entry['B']/acc)) + '\n')
            f.write(str(point[0]-(offset/2)) + ' ' + str(point[1]+(offset/2)) + ' ' + str(point[2]+(offset/2)) + ' ' + str(int(M_entry['R']/acc)) + ' ' + str(int(M_entry['G']/acc)) + ' ' + str(int(M_entry['B']/acc)) + '\n')
            f.write(str(point[0]-(offset/2)) + ' ' + str(point[1]+(offset/2)) + ' ' + str(point[2]-(offset/2)) + ' ' + str(int(M_entry['R']/acc)) + ' ' + str(int(M_entry['G']/acc)) + ' ' + str(int(M_entry['B']/acc)) + '\n')
            f.write(str(point[0]+(offset/2)) + ' ' + str(point[1]-(offset/2)) + ' ' + str(point[2]-(offset/2)) + ' ' + str(int(M_entry['R']/acc)) + ' ' + str(int(M_entry['G']/acc)) + ' ' + str(int(M_entry['B']/acc)) + '\n')
            f.write(str(point[0]+(offset/2)) + ' ' + str(point[1]-(offset/2)) + ' ' + str(point[2]+(offset/2)) + ' ' + str(int(M_entry['R']/acc)) + ' ' + str(int(M_entry['G']/acc)) + ' ' + str(int(M_entry['B']/acc)) + '\n')
            f.write(str(point[0]+(offset/2)) + ' ' + str(point[1]+(offset/2)) + ' ' + str(point[2]+(offset/2)) + ' ' + str(int(M_entry['R']/acc)) + ' ' + str(int(M_entry['G']/acc)) + ' ' + str(int(M_entry['B']/acc)) + '\n')
            f.write(str(point[0]+(offset/2)) + ' ' + str(point[1]+(offset/2)) + ' ' + str(point[2]-(offset/2)) + ' ' + str(int(M_entry['R']/acc)) + ' ' + str(int(M_entry['G']/acc)) + ' ' + str(int(M_entry['B']/acc)) + '\n')
        # number of edges = len(data.set) * 8
        # number of faces = len(data.set) * 6
        for i, point in enumerate(data.set):
            f.write('4 ' + str(0+(8*i)) + ' ' + str(1+(8*i)) + ' ' + str(2+(8*i)) + ' ' + str((3+(8*i))) + '\n')
            f.write('4 ' + str(7+(8*i)) + ' ' + str(6+(8*i)) + ' ' + str(5+(8*i)) + ' ' + str((4+(8*i))) + '\n')
            f.write('4 ' + str(0+(8*i)) + ' ' + str(4+(8*i)) + ' ' + str(5+(8*i)) + ' ' + str((1+(8*i))) + '\n')
            f.write('4 ' + str(1+(8*i)) + ' ' + str(5+(8*i)) + ' ' + str(6+(8*i)) + ' ' + str((2+(8*i))) + '\n')
            f.write('4 ' + str(2+(8*i)) + ' ' + str(6+(8*i)) + ' ' + str(7+(8*i)) + ' ' + str((3+(8*i))) + '\n')
            f.write('4 ' + str(3+(8*i)) + ' ' + str(7+(8*i)) + ' ' + str(4+(8*i)) + ' ' + str((0+(8*i))) + '\n')
        f.close()


def process_voxing(frame):
    global voxel_set
    global M
    frame = undistort_frame(frame, K, dist)
    background_removal(frame, configs)
    old_frame = frame.copy()
    rvec, tvec = pose_estimation(old_frame, configs, K)
    imgpts = cv2.projectPoints(x_axis, rvec, tvec, K, np.array([]))
    cv2.line(frame, np.array(imgpts[0][0][0], dtype=np.int32), np.array(imgpts[0][1][0], dtype=np.int32), (255,0,0), 5)
    img_x_axis = np.int32([[imgpts[0][0][0][0],imgpts[0][0][0][1]], [imgpts[0][0][0][0]+50,imgpts[0][0][0][1]]]).reshape(-1,2)

    cv2.line(frame, np.array(imgpts[0][0][0], dtype=np.int32), np.array(imgpts[0][1][0], dtype=np.int32), (255,0,0), 5)

    voxel_pts = cv2.projectPoints(np.array(voxel_set.set), rvec, tvec, K, np.array([]))

    cv2.line(frame, img_x_axis[0], img_x_axis[1], (255,255,0), 5)

    delta_X = imgpts[0][1][0][0] - imgpts[0][0][0][0]
    delta_Y = imgpts[0][1][0][1] - imgpts[0][0][0][1]
    # CHECK WHICH VOXEL NEEDS TO BE COLORED
    # get only the external voxel:
    # 
    if delta_X >= 0 and delta_Y >= 0:
        if delta_X > delta_Y:
            for j in range(len(voxel_pts[0])):
                voxel_3d_point = voxel_set.set[j]
                y = get_min_y_given_xz(M, voxel_3d_point[0], voxel_3d_point[2])
                color = frame[np.int32(voxel_pts[0][j][0][1]),np.int32(voxel_pts[0][j][0][0])]
                M[voxel_3d_point[0]][y][voxel_3d_point[2]]['R'] += int(color[2])
                M[voxel_3d_point[0]][y][voxel_3d_point[2]]['G'] += int(color[1])
                M[voxel_3d_point[0]][y][voxel_3d_point[2]]['B'] += int(color[0])
                M[voxel_3d_point[0]][y][voxel_3d_point[2]]['acc'] += 1
            cv2.putText(frame, '1 - min y', np.int32(imgpts[0][1][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
        else:
            for j in range(len(voxel_pts[0])):
                voxel_3d_point = voxel_set.set[j]
                x = get_min_x_given_yz(M, voxel_3d_point[1], voxel_3d_point[2])
                color = frame[np.int32(voxel_pts[0][j][0][1]),np.int32(voxel_pts[0][j][0][0])]
                M[x][voxel_3d_point[1]][voxel_3d_point[2]]['R'] += int(color[2])
                M[x][voxel_3d_point[1]][voxel_3d_point[2]]['G'] += int(color[1])
                M[x][voxel_3d_point[1]][voxel_3d_point[2]]['B'] += int(color[0])
                M[x][voxel_3d_point[1]][voxel_3d_point[2]]['acc'] += 1
            cv2.putText(frame, '4 - min x', np.int32(imgpts[0][1][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)

    if delta_X >= 0 and delta_Y < 0:
        #for j in range(len(voxel_pts[0])):
        if delta_X > - delta_Y:
            for j in range(len(voxel_pts[0])):
                voxel_3d_point = voxel_set.set[j]
                y = get_min_y_given_xz(M, voxel_3d_point[0], voxel_3d_point[2])
                color = frame[np.int32(voxel_pts[0][j][0][1]),np.int32(voxel_pts[0][j][0][0])]
                M[voxel_3d_point[0]][y][voxel_3d_point[2]]['R'] += int(color[2])
                M[voxel_3d_point[0]][y][voxel_3d_point[2]]['G'] += int(color[1])
                M[voxel_3d_point[0]][y][voxel_3d_point[2]]['B'] += int(color[0])
                M[voxel_3d_point[0]][y][voxel_3d_point[2]]['acc'] += 1
            cv2.putText(frame, '1 - min y', np.int32(imgpts[0][1][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
        else:   
            for j in range(len(voxel_pts[0])):
                voxel_3d_point = voxel_set.set[j]
                x = get_max_x_given_yz(M, voxel_3d_point[1], voxel_3d_point[2])
                color = frame[np.int32(voxel_pts[0][j][0][1]),np.int32(voxel_pts[0][j][0][0])]
                M[x][voxel_3d_point[1]][voxel_3d_point[2]]['R'] += int(color[2])
                M[x][voxel_3d_point[1]][voxel_3d_point[2]]['G'] += int(color[1])
                M[x][voxel_3d_point[1]][voxel_3d_point[2]]['B'] += int(color[0])
                M[x][voxel_3d_point[1]][voxel_3d_point[2]]['acc'] += 1
            cv2.putText(frame, '2 - max x', np.int32(imgpts[0][1][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
    if delta_X < 0 and delta_Y < 0:
        if delta_X > delta_Y:
            for j in range(len(voxel_pts[0])):
                voxel_3d_point = voxel_set.set[j]
                x = get_max_x_given_yz(M, voxel_3d_point[1], voxel_3d_point[2])
                color = frame[np.int32(voxel_pts[0][j][0][1]),np.int32(voxel_pts[0][j][0][0])]
                M[x][voxel_3d_point[1]][voxel_3d_point[2]]['R'] += int(color[2])
                M[x][voxel_3d_point[1]][voxel_3d_point[2]]['G'] += int(color[1])
                M[x][voxel_3d_point[1]][voxel_3d_point[2]]['B'] += int(color[0])
                M[x][voxel_3d_point[1]][voxel_3d_point[2]]['acc'] += 1
            cv2.putText(frame, '2 - max x', np.int32(imgpts[0][1][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
        else:
            for j in range(len(voxel_pts[0])):
                voxel_3d_point = voxel_set.set[j]
                y = get_max_y_given_xz(M, voxel_3d_point[0], voxel_3d_point[2])
                color = frame[np.int32(voxel_pts[0][j][0][1]),np.int32(voxel_pts[0][j][0][0])]
                M[voxel_3d_point[0]][y][voxel_3d_point[2]]['R'] += int(color[2])
                M[voxel_3d_point[0]][y][voxel_3d_point[2]]['G'] += int(color[1])
                M[voxel_3d_point[0]][y][voxel_3d_point[2]]['B'] += int(color[0])
                M[voxel_3d_point[0]][y][voxel_3d_point[2]]['acc'] += 1
            cv2.putText(frame, '3 - max y', np.int32(imgpts[0][1][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
        print('THIRD quadrant')
    if delta_X < 0 and delta_Y >= 0:
        if -delta_X < delta_Y:
            for j in range(len(voxel_pts[0])):
                voxel_3d_point = voxel_set.set[j]
                x = get_min_x_given_yz(M, voxel_3d_point[1], voxel_3d_point[2])
                color = frame[np.int32(voxel_pts[0][j][0][1]),np.int32(voxel_pts[0][j][0][0])]
                M[x][voxel_3d_point[1]][voxel_3d_point[2]]['R'] += int(color[2])
                M[x][voxel_3d_point[1]][voxel_3d_point[2]]['G'] += int(color[1])
                M[x][voxel_3d_point[1]][voxel_3d_point[2]]['B'] += int(color[0])
                M[x][voxel_3d_point[1]][voxel_3d_point[2]]['acc'] += 1
            cv2.putText(frame, '4 - min y', np.int32(imgpts[0][1][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
        else:
            for j in range(len(voxel_pts[0])):
                voxel_3d_point = voxel_set.set[j]
                y = get_max_y_given_xz(M, voxel_3d_point[0], voxel_3d_point[2])
                color = frame[np.int32(voxel_pts[0][j][0][1]),np.int32(voxel_pts[0][j][0][0])]
                M[voxel_3d_point[0]][y][voxel_3d_point[2]]['R'] += int(color[2])
                M[voxel_3d_point[0]][y][voxel_3d_point[2]]['G'] += int(color[1])
                M[voxel_3d_point[0]][y][voxel_3d_point[2]]['B'] += int(color[0])
                M[voxel_3d_point[0]][y][voxel_3d_point[2]]['acc'] += 1
            cv2.putText(frame, '3 - max y', np.int32(imgpts[0][1][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
    cv2.imshow('QUADRANTS', frame)
    cv2.waitKey(25)

video_capture = VideoCapture('data/' + configs.working_object + '.mp4')
video_capture.process_video(process_voxing)
save_ply(configs.working_object + '-color', voxel_set, M)