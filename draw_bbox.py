# 시각화에 필요한 라이브러리 불러오기
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def draw_bbox(file_path):
    # PCD 파일 읽기
    original_pcd = o3d.io.read_point_cloud(file_path)

    # Voxel Downsampling 수행
    voxel_size = 0.2  # 필요에 따라 voxel 크기를 조정하세요.
    downsample_pcd = original_pcd.voxel_down_sample(voxel_size=voxel_size)

    '''
    # Radius Outlier Removal (ROR) 적용
    cl, ind = downsample_pcd.remove_radius_outlier(nb_points=6, radius=1.2)
    ror_pcd = downsample_pcd.select_by_index(ind)
    '''
    plane_model, inliers = downsample_pcd.segment_plane(distance_threshold=0.1,
                                                ransac_n=3,
                                                num_iterations=2000)
    final_point = downsample_pcd.select_by_index(inliers, invert=True)

    # DBSCAN 클러스터링 적용
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(final_point.cluster_dbscan(eps=0.33, min_points=10, print_progress=True))

    # 노이즈 포인트는 검정색, 클러스터 포인트는 파란색으로 지정
    colors = np.zeros((len(labels), 3))  # 기본 검정색 (노이즈)
    colors[labels >= 0] = [0, 0, 1]  # 파란색으로 지정

    final_point.colors = o3d.utility.Vector3dVector(colors)

    # 필터링 기준 설정
    min_points_in_cluster = 10   # 클러스터 내 최소 포인트 수, 3(수영)
    max_points_in_cluster = 40  # 클러스터 내 최대 포인트 수 2(2명), 4(나무랑 겹칠 때)
    min_z_value = 0          # 클러스터 내 최소 Z값, (4,5,6,7에서도 먹혀야 함, 끝까지 오는 사람한테도 먹혀야 함(3))
    max_z_value = 5.0           # 클러스터 내 최대 Z값
    min_height = 0.2            # Z값 차이의 최소값, 2, 3(수영?)에서 해야 할 문제
    max_height = 1.3            # Z값 차이의 최대값, *****고정*****
    max_distance = 80.0         # 원점으로부터의 최대 거리
    max_depth = 2.0
    max_width = 3.0
    # 1번, 2번, 3번 조건을 모두 만족하는 클러스터 필터링 및 바운딩 박스 생성
    bboxes_1234 = []
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
                            bboxes_1234.append(bbox)
                    
    return final_point, bboxes_1234
