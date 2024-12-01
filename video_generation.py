import open3d as o3d
import cv2
import numpy as np
import os

#from draw_bbox import draw_bbox
from draw_bbox_wo_back import draw_bbox

def video_generation(pcd_dir, output_video):

    view_param_file='view_params.json'
    point_size = 1.0
    frame_width = 1920
    frame_height = 1054
    fps = 20

    pcd_files = sorted([f for f in os.listdir(pcd_dir) if f.endswith('.pcd')])

    # setting view parameters
    custom_view_parameters = None

    if os.path.exists(view_param_file):
        custom_view_parameters = o3d.io.read_pinhole_camera_parameters(view_param_file)
        print("Loaded saved view parameters.")
    else:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=frame_width, height=frame_height)
        vis.get_render_option().point_size = point_size
        view_ctl = vis.get_view_control()

        pcd = o3d.io.read_point_cloud(os.path.join(pcd_dir, pcd_files[0]))
        vis.add_geometry(pcd)
        
        print("Adjust the view as desired and close the window to save the view parameters.")
        vis.run() 

        custom_view_parameters = view_ctl.convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters(view_param_file, custom_view_parameters)
        print(f"View parameters saved to '{view_param_file}' file.")
        
        vis.clear_geometries()
        vis.destroy_window()


    # making a video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=frame_width, height=frame_height, visible=False)
    vis.get_render_option().point_size = point_size
    view_ctl = vis.get_view_control()

    clusters = []
    for i,pcd_file in enumerate(pcd_files[:]):
        vis.clear_geometries()
        file_path = os.path.join(pcd_dir, pcd_file)

        
        pcd, bounding_boxes, clusters = draw_bbox(file_path, i, clusters)
        vis.add_geometry(pcd)
        for bbox in bounding_boxes:
            vis.add_geometry(bbox)
        
        
        view_ctl.convert_from_pinhole_camera_parameters(custom_view_parameters, allow_arbitrary=True)

        vis.poll_events()
        vis.update_renderer()

        image = vis.capture_screen_float_buffer(do_render=True)
        image = np.asarray(image)
        image = (image * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        video_writer.write(image)
        print(f"Completed: {i+1}/{len(pcd_files)} pcd files.")

    video_writer.release()
    vis.destroy_window()

if __name__ == "__main__":
    scenarios = ['01_straight_walk','02_straight_duck_walk','03_straight_crawl',
                 '04_zigzag_walk','05_straight_duck_walk','06_straight_crawl',
                 '07_straight_walk']
    for i in range(3):
        video_generation(f"data/{scenarios[i]}/pcd", f"{scenarios[i]}_wo_back4.mp4")

