"""
Custom mesh visualization with emerald green color for hero videos.
Based on SAM-3d-body visualization utilities.
"""

import numpy as np
import cv2
from sam_3d_body.visualization.renderer import Renderer
from sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer
from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info

# Emerald green in RGB (normalized 0-1)
# #50C878 = RGB(80, 200, 120) = normalized (0.314, 0.784, 0.471)
EMERALD_GREEN = (80 / 255.0, 200 / 255.0, 120 / 255.0)  # #50C878

visualizer = SkeletonVisualizer(line_width=2, radius=5)
visualizer.set_pose_meta(mhr70_pose_info)


def visualize_sample_together_emerald(img_cv2, outputs, faces):
    """
    Render everything together with emerald green mesh color.
    Based on visualize_sample_together but with custom emerald green color.
    """
    img_keypoints = img_cv2.copy()
    img_mesh = img_cv2.copy()

    # First, sort by depth, furthest to closest
    all_depths = np.stack([tmp['pred_cam_t'] for tmp in outputs], axis=0)[:, 2]
    outputs_sorted = [outputs[idx] for idx in np.argsort(-all_depths)]

    # Then, draw all keypoints with emerald green
    # Update visualizer colors to emerald green
    emerald_green_rgb = tuple(reversed(EMERALD_GREEN))  # RGB to match visualizer
    visualizer.kpt_color = emerald_green_rgb
    visualizer.link_color = emerald_green_rgb
    
    for pid, person_output in enumerate(outputs_sorted):
        keypoints_2d = person_output["pred_keypoints_2d"]
        keypoints_2d = np.concatenate(
            [keypoints_2d, np.ones((keypoints_2d.shape[0], 1))], axis=-1
        )
        img_keypoints = visualizer.draw_skeleton(img_keypoints, keypoints_2d)

    # Then, put all meshes together as one super mesh
    all_pred_vertices = []
    all_faces = []
    for pid, person_output in enumerate(outputs_sorted):
        all_pred_vertices.append(person_output["pred_vertices"] + person_output["pred_cam_t"])
        all_faces.append(faces + len(person_output["pred_vertices"]) * pid)
    all_pred_vertices = np.concatenate(all_pred_vertices, axis=0)
    all_faces = np.concatenate(all_faces, axis=0)

    # Pull out a fake translation; take the closest two
    fake_pred_cam_t = (np.max(all_pred_vertices[-2*18439:], axis=0) + np.min(all_pred_vertices[-2*18439:], axis=0)) / 2
    all_pred_vertices = all_pred_vertices - fake_pred_cam_t
    
    # Render front view with emerald green mesh
    renderer = Renderer(focal_length=outputs_sorted[0]["focal_length"], faces=all_faces)
    img_mesh = (
        renderer(
            all_pred_vertices,
            fake_pred_cam_t,
            img_mesh,
            mesh_base_color=EMERALD_GREEN,  # Use emerald green instead of LIGHT_BLUE
            scene_bg_color=(1, 1, 1),
        )
        * 255
    )

    return img_mesh
