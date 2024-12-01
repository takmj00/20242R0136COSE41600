import open3d as o3d
import numpy as np
import os
import matplotlib.pyplot as plt

class cluster:
    def __init__(self, centroid, prev, fixed):
        self.centroid = centroid
        self.prev = prev
        self.fixed = fixed
        
background_model = []


def draw_bbox(file_path, file_num, prev_clusters=[]):
    print(len(prev_clusters))
    global background_model

    current_pcd = o3d.io.read_point_cloud(file_path)
    current_points = np.asarray(current_pcd.points)
    foreground_pcd = current_pcd

    if len(background_model) < 5:
        background_model.append(current_points)

    elif file_num > len(background_model) + 2 : 
        
        distance_threshold = 0.5  
        background_indices = []

        for background_points in background_model:
            distances = np.linalg.norm(current_points - background_points, axis=1)
            background_indices.extend(np.where(distances < distance_threshold)[0])
        
        background_indices = np.unique(background_indices)
        all_indices = np.arange(len(current_points))
        foreground_indices = np.setdiff1d(all_indices, background_indices)
        
        if len(foreground_indices) == 0:
            return current_pcd, []

        foreground_points = current_points[foreground_indices]
        foreground_pcd = o3d.geometry.PointCloud()
        foreground_pcd.points = o3d.utility.Vector3dVector(foreground_points)
        
    voxel_size = 0.2
    foreground_pcd = foreground_pcd.voxel_down_sample(voxel_size=voxel_size)
    
    cl, ind = foreground_pcd.remove_radius_outlier(nb_points=6, radius=1.2)
    foreground_pcd = foreground_pcd.select_by_index(ind)
    
    final_point = foreground_pcd
    eps = 0.4  
    min_points = 7  
    labels = np.array(final_point.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    
    max_label = labels.max()
    bounding_boxes = []
    current_clusters = []
    if max_label < 0:
        return current_pcd, []

    # 노이즈 포인트는 검정색, 클러스터 포인트는 파란색으로 지정
    colors = np.zeros((len(labels), 3))  # 기본 검정색 (노이즈)
    colors[labels >= 0] = [0, 0, 1]  # 파란색으로 지정

    final_point.colors = o3d.utility.Vector3dVector(colors)

    # 필터링 기준 설정
    min_points_in_cluster = 10   # 클러스터 내 최소 포인트 수, 3(수영)
    max_points_in_cluster = 80  # 클러스터 내 최대 포인트 수 2(2명), 4(나무랑 겹칠 때)

    max_z_value = 10.0           # 클러스터 내 최대 Z값
    min_height = 0.2            # Z값 차이의 최소값, 2, 3(수영?)에서 해야 할 문제
    max_height = 1.3            # Z값 차이의 최대값, *****고정*****
    max_distance = 80.0         # 원점으로부터의 최대 거리
    max_depth = 1.5
    max_width = 1.5

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
                            cluster_centroid = np.mean(points, axis=0)
                            prev = None
                            fixed = False

                            min_distance = 5.0
                            closest_prev_cluster = None
                            # matching 
                            for prev_cluster in prev_clusters:
                                prev_distance = np.linalg.norm(cluster_centroid - prev_cluster.centroid)
                                if prev_distance < min_distance:
                                    min_distance = prev_distance
                                    closest_prev_cluster = prev_cluster

                            # preprev => prev => current
                            # 현재 클러스터와 시작점의 거리가 이전 클러스터와의 거리보다 가까우면 고정 클러스터
                            # 거리가 더 짧다는 건 전진하지 않았음을 의미
                            # 그러나 이전 클러스터가 고정으로 한 번이라도 판단되었다면 계속 고정 클러스터로 판단 - 포기. 오리걸음을 못 잡음
                            if closest_prev_cluster and closest_prev_cluster.prev is not None:
                                prev_cluster = closest_prev_cluster
                                while prev_cluster.prev is not None:
                                    prev_cluster = prev_cluster.prev
                                prev_distance = np.linalg.norm(cluster_centroid - prev_cluster.centroid)
                                
                                # 고정 클러스터 
                                if prev_distance < min_distance+0.2:                                     
                                #if prev_distance < min_distance+0.2 or closest_prev_cluster.fixed:                                     
                                    new_cluster = cluster(centroid=cluster_centroid, prev=closest_prev_cluster, fixed=True)
                                    current_clusters.append(new_cluster)
                                    continue
                                # 
                                prev = closest_prev_cluster
                                fixed = False
                                    
                            # Not matched. 판단이 안 서니 일단 bbox
                            elif closest_prev_cluster is None:
                                prev = None
                                fixed = False
                            # None => prev => current. 전 클러스터와의 연결로는 이동하는 것인지 판단 불가이니 일단 bbox
                            elif closest_prev_cluster.prev is None:
                                prev = closest_prev_cluster
                                fixed = False

                            bbox = cluster_pcd.get_axis_aligned_bounding_box()
                            bbox.color = (1, 0, 0)
                            bounding_boxes.append(bbox)

                            new_cluster = cluster(cluster_centroid, prev, fixed)
                            current_clusters.append(new_cluster)
    return current_pcd, bounding_boxes, current_clusters
