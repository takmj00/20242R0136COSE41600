import open3d as o3d
import cv2
import numpy as np
import os

def video_generation(pcd_dir, output_video, view_param_file='view_params.json'):

    frame_width = 1920
    frame_height = 1054
    fps = 30

    pcd_files = sorted([f for f in os.listdir(pcd_dir) if f.endswith('.pcd')])

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))


    vis = o3d.visualization.Visualizer()
    vis.create_window(width=frame_width, height=frame_height)
    vis.get_render_option().point_size = 1

    view_ctl = vis.get_view_control()

    custom_view_parameters = None

    if os.path.exists(view_param_file):
        custom_view_parameters = o3d.io.read_pinhole_camera_parameters(view_param_file)
        print("Loaded saved view parameters.")
    else:
        pcd = o3d.io.read_point_cloud(os.path.join(pcd_dir, pcd_files[0]))
        vis.add_geometry(pcd)
        
        print("Adjust the view as desired and close the window to save the view parameters.")
        vis.run() 

        custom_view_parameters = view_ctl.convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters(view_param_file, custom_view_parameters)
        print(f"View parameters saved to '{view_param_file}' file.")
        
        vis.clear_geometries()

    vis.destroy_window()
    vis.create_window(width=frame_width, height=frame_height, visible=False)
    opt = vis.get_render_option()
    vis.get_render_option().point_size = 1

    view_ctl = vis.get_view_control()

    for pcd_file in pcd_files:
        vis.clear_geometries()
        pcd = o3d.io.read_point_cloud(os.path.join(pcd_dir, pcd_file))
        vis.add_geometry(pcd)

        view_ctl.convert_from_pinhole_camera_parameters(custom_view_parameters, allow_arbitrary=True)

        vis.poll_events()
        vis.update_renderer()

        image = vis.capture_screen_float_buffer(do_render=True)
        image = np.asarray(image)
        image = (image * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        video_writer.write(image)

    video_writer.release()
    vis.destroy_window()

if __name__ == "__main__":
    video_generation("data/01_straight_walk/pcd", "output_video.mp4")