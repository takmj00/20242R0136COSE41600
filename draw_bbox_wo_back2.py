import open3d as o3d
import numpy as np
import os
import matplotlib.pyplot as plt

# 전역 변수로 배경 모델을 유지
background_model = None

def create_background_model(pcd_dir, num_frames=10):
    
    pcd_files = sorted([f for f in os.listdir(pcd_dir) if f.endswith('.pcd')])[:5]
    all_points = []

    for pcd_file in pcd_files:
        pcd = o3d.io.read_point_cloud(os.path.join(pcd_dir, pcd_file))
        all_points.append(np.asarray(pcd.points))
    
    all_points = np.asarray(all_points)

    mean_points = np.mean(all_points, axis=0)
    background_pcd = o3d.geometry.PointCloud()
    background_pcd.points = o3d.utility.Vector3dVector(mean_points.reshape(-1, 3))
    

    '''
    point_size = 2.0
    frame_width = 1920
    frame_height = 1054
    view_param_file='view_params.json'
    custom_view_parameters = o3d.io.read_pinhole_camera_parameters(view_param_file)

    background_pcd.paint_uniform_color([0, 0, 1])
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=frame_width, height=frame_height)
    vis.get_render_option().point_size = point_size
    view_ctl = vis.get_view_control()
    vis.add_geometry(background_pcd)
    view_ctl.convert_from_pinhole_camera_parameters(custom_view_parameters, allow_arbitrary=True)
    vis.get_render_option().point_size = 1
    vis.run()
    vis.destroy_window()
    '''
    return background_pcd

def draw_bbox(file_path, pcd_dir):
    
    global background_model

    if background_model is None:
        print("Creating background model...")
        background_model = create_background_model(pcd_dir)
        print("Background model created.")

    current_pcd = o3d.io.read_point_cloud(file_path)
    current_points = np.asarray(current_pcd.points)

    background_points = np.asarray(background_model.points)

    distance_threshold = 2.0  
    distances = np.linalg.norm(current_points - background_points, axis=1)
    foreground_indices = np.where(distances > distance_threshold)[0]


    

    if len(foreground_indices) == 0:
        return current_pcd, []

    foreground_points = current_points[foreground_indices]
    foreground_pcd = o3d.geometry.PointCloud()
    foreground_pcd.points = o3d.utility.Vector3dVector(foreground_points)

    voxel_size = 0.05 
    foreground_pcd = foreground_pcd.voxel_down_sample(voxel_size=voxel_size)
    
    cl, ind = foreground_pcd.remove_radius_outlier(nb_points=6, radius=1.2)
    foreground_pcd = foreground_pcd.select_by_index(ind)
    plane_model, inliers = foreground_pcd.segment_plane(distance_threshold=0.1,
                                                ransac_n=3,
                                                num_iterations=2000)
    final_point = foreground_pcd.select_by_index(inliers, invert=True)

    eps = 0.33  
    min_points = 10  
    labels = np.array(final_point.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    
    max_label = labels.max()
    bounding_boxes = []

    if max_label < 0:
        return current_pcd, []

    # 노이즈 포인트는 검정색, 클러스터 포인트는 파란색으로 지정
    colors = np.zeros((len(labels), 3))  # 기본 검정색 (노이즈)
    colors[labels >= 0] = [0, 0, 1]  # 파란색으로 지정

    final_point.colors = o3d.utility.Vector3dVector(colors)

    # 필터링 기준 설정
    min_points_in_cluster = 10   # 클러스터 내 최소 포인트 수, 3(수영)
    max_points_in_cluster = 40  # 클러스터 내 최대 포인트 수 2(2명), 4(나무랑 겹칠 때)

    max_z_value = 5.0           # 클러스터 내 최대 Z값
    min_height = 0.2            # Z값 차이의 최소값, 2, 3(수영?)에서 해야 할 문제
    max_height = 1.3            # Z값 차이의 최대값, *****고정*****
    max_distance = 80.0         # 원점으로부터의 최대 거리
    max_depth = 2.0
    max_width = 3.0
    # 1번, 2번, 3번 조건을 모두 만족하는 클러스터 필터링 및 바운딩 박스 생성
    for i in range(labels.max() + 1):
        cluster_indices = np.where(labels == i)[0]
        if min_points_in_cluster <= len(cluster_indices) <= max_points_in_cluster:

            cluster_pcd = final_point.select_by_index(cluster_indices)
            points = np.asarray(cluster_pcd.points)
            y_values = points[:, 2]
            x_values = points[:, 2]
            if y_values.max()-y_values.min() <= max_depth and x_values.max()-x_values.min() <= max_width:
                    
                z_values = points[:, 2]
                z_min = z_values.min()
                z_max = z_values.max()
                if z_max <= max_z_value:
                    height_diff = z_max - z_min
                    if min_height <= height_diff <= max_height:
                        distances = np.linalg.norm(points, axis=1)
                        if distances.max() <= max_distance:
                            bbox = cluster_pcd.get_axis_aligned_bounding_box()
                            bbox.color = (1, 0, 0) 
                            bounding_boxes.append(bbox)

    return foreground_pcd, bounding_boxes
