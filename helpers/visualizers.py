# visualize depth
import os
import mayavi.mlab as mlab
import argparse
from matplotlib import pyplot as plt
import numpy as np
# import matplotlib
import cv2

filedir = os.path.dirname(os.path.realpath(__file__))
project_dir = os.getcwd()

# SDN
depth_map_relative_path = '/results/sdn_kitti_test_sub/depth_maps/test'
pl_relative_path = '/results/sdn_kitti_test_sub/pseudo_lidar'

# SDN + GDC
depth_map_relative_path2 = '/results/sdn_kitti_test_sub/depth_maps/test/corrected'
pl_relative_path2 = '/results/sdn_kitti_test_sub/pseudo_lidar/corrected'
# pl_relative_path2 = '/kitti/testing/velodyne'


# pts_mode='sphere'
kitti_relative_path = '/kitti'


# colors
color_lime_green = (50, 205, 50)


def norm_color(input_color):
    return tuple([x / 255.0 for x in input_color])


def draw_lidar(
    pc,
    color=None,
    fig=None,
    bgcolor=(0, 0, 0),
    pts_scale=0.3,
    pts_mode="sphere",
    pts_color=None,
    color_by_intensity=False,
    pc_label=False,
):
    """ Draw lidar points
    Args:
        pc: numpy array (n,3) of XYZ
        color: numpy array (n) of intensity or whatever
        fig: mayavi figure handler, if None create new one otherwise will use it
    Returns:
        fig: created or used fig
    """
    # ind = (pc[:,2]< -1.65)
    # pc = pc[ind]
    pts_mode = "point"
    print("====================", pc.shape)
    if fig is None:
        fig = mlab.figure(
            figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(1600, 1000)
        )
    if color is None:
        color = pc[:, 2]
    if pc_label:
        color = pc[:, 4]
    if color_by_intensity:
        color = pc[:, 2]

    mlab.points3d(
        pc[:, 0],
        pc[:, 1],
        pc[:, 2],
        color,
        color=pts_color,
        mode=pts_mode,
        colormap="gnuplot",
        scale_factor=pts_scale,
        figure=fig,
    )

    # draw origin
    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode="sphere", scale_factor=0.2)

    # draw axis
    axes = np.array(
        [[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]],
        dtype=np.float64,
    )
    mlab.plot3d(
        [0, axes[0, 0]],
        [0, axes[0, 1]],
        [0, axes[0, 2]],
        color=(1, 0, 0),
        tube_radius=None,
        figure=fig,
    )
    mlab.plot3d(
        [0, axes[1, 0]],
        [0, axes[1, 1]],
        [0, axes[1, 2]],
        color=(0, 1, 0),
        tube_radius=None,
        figure=fig,
    )
    mlab.plot3d(
        [0, axes[2, 0]],
        [0, axes[2, 1]],
        [0, axes[2, 2]],
        color=(0, 0, 1),
        tube_radius=None,
        figure=fig,
    )

    # draw fov (todo: update to real sensor spec.)
    fov = np.array(
        [[20.0, 20.0, 0.0, 0.0], [20.0, -20.0, 0.0, 0.0]], dtype=np.float64  # 45 degree
    )

    mlab.plot3d(
        [0, fov[0, 0]],
        [0, fov[0, 1]],
        [0, fov[0, 2]],
        color=(1, 1, 1),
        tube_radius=None,
        line_width=1,
        figure=fig,
    )
    mlab.plot3d(
        [0, fov[1, 0]],
        [0, fov[1, 1]],
        [0, fov[1, 2]],
        color=(1, 1, 1),
        tube_radius=None,
        line_width=1,
        figure=fig,
    )

    # draw square region
    TOP_Y_MIN = -20
    TOP_Y_MAX = 20
    TOP_X_MIN = 0
    TOP_X_MAX = 40
    TOP_Z_MIN = -2.0
    TOP_Z_MAX = 0.4

    x1 = TOP_X_MIN
    x2 = TOP_X_MAX
    y1 = TOP_Y_MIN
    y2 = TOP_Y_MAX
    mlab.plot3d(
        [x1, x1],
        [y1, y2],
        [0, 0],
        color=(0.5, 0.5, 0.5),
        tube_radius=0.1,
        line_width=1,
        figure=fig,
    )
    mlab.plot3d(
        [x2, x2],
        [y1, y2],
        [0, 0],
        color=(0.5, 0.5, 0.5),
        tube_radius=0.1,
        line_width=1,
        figure=fig,
    )
    mlab.plot3d(
        [x1, x2],
        [y1, y1],
        [0, 0],
        color=(0.5, 0.5, 0.5),
        tube_radius=0.1,
        line_width=1,
        figure=fig,
    )
    mlab.plot3d(
        [x1, x2],
        [y2, y2],
        [0, 0],
        color=(0.5, 0.5, 0.5),
        tube_radius=0.1,
        line_width=1,
        figure=fig,
    )

    # mlab.orientation_axes()
    mlab.view(
        azimuth=180,
        elevation=70,
        focalpoint=[12.0909996, -1.04700089, -2.03249991],
        distance=62.0,
        figure=fig,
    )
    return fig


def load_lidar_pc(pc_filename, dtype=np.float32, n_vec=4):
    scan = np.fromfile(pc_filename, dtype=dtype)
    scan = scan.reshape((-1, n_vec))
    return scan


def load_image(img_filename):
    return cv2.imread(img_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize_PL')
    parser.add_argument('-i', '--index', required=True, help='image index')
    args = parser.parse_args()

    # print(filedir)
    # print(project_dir)
    for key, value in sorted(vars(args).items()):
        print("{} : {}".format(key, value))

    image_full_path = project_dir + depth_map_relative_path2 + '/' + args.index + '.npy'
    print("showing image: {}".format(image_full_path))
    img_array = np.load(image_full_path)

    plt.imshow(img_array, cmap='gray')

    fig = mlab.figure(
        figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500)
    )
    lidar_full_path = project_dir + pl_relative_path + '/' + args.index + '.bin'
    print("showing lidar: {}".format(lidar_full_path))
    pc = load_lidar_pc(lidar_full_path)
    # draw_lidar(pc, fig=fig)

    lidar_full_path2 = project_dir + pl_relative_path2 + '/' + args.index + '.bin'
    print("showing lidar: {}".format(lidar_full_path2))
    pc_corrected = load_lidar_pc(lidar_full_path2)
    # draw_lidar(pc_corrected, fig=fig)
    draw_lidar(pc_corrected, fig=fig)
    left_image_path = project_dir + kitti_relative_path + \
        '/testing' + '/image_2/' + args.index + '.png'
    left_image = load_image(left_image_path)
    right_image_path = project_dir + kitti_relative_path + \
        '/testing' + '/image_3/' + args.index + '.png'
    right_image = load_image(right_image_path)

    cv2.imshow("left_image", left_image)
    cv2.imshow("right_image", right_image)

    # mlab.savefig("pc.jpg", figure=fig)
    plt.show()
    mlab.show()
    # mlab.clf()
