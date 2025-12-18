#!/usr/bin/env python3
"""
Interactive 3D renderer for SAM-3d-body JSON output.
Loads JSON data and displays interactive 3D meshes and skeletons that can be
rotated, zoomed, and panned with the mouse.
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np
import cv2
import tempfile
import shutil

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("Warning: PyVista not available. Trying Open3D...")
    try:
        import open3d as o3d
        OPEN3D_AVAILABLE = True
    except ImportError:
        OPEN3D_AVAILABLE = False
        print("Error: Neither PyVista nor Open3D is available.")
        print("Please install one of them:")
        print("  pip install pyvista")
        print("  or")
        print("  pip install open3d")
        sys.exit(1)


def load_json_data(json_path):
    """Load and parse JSON data file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def create_mesh_from_data(vertices, faces):
    """Create a mesh object from vertices and faces."""
    if PYVISTA_AVAILABLE:
        # PyVista mesh
        vertices = np.array(vertices, dtype=np.float32)
        faces = np.array(faces, dtype=np.int32)
        
        # PyVista expects faces in format: [n, i1, i2, i3, n, i4, i5, i6, ...]
        # where n is the number of vertices in the face (3 for triangles)
        if len(faces.shape) == 2 and faces.shape[1] == 3:
            # Convert to PyVista format
            n_faces = faces.shape[0]
            faces_pv = np.empty((n_faces, 4), dtype=np.int32)
            faces_pv[:, 0] = 3  # Triangle
            faces_pv[:, 1:] = faces
            faces_pv = faces_pv.flatten()
        else:
            faces_pv = faces
        
        mesh = pv.PolyData(vertices, faces_pv)
        return mesh
    elif OPEN3D_AVAILABLE:
        # Open3D mesh
        vertices = np.array(vertices, dtype=np.float32)
        faces = np.array(faces, dtype=np.int32)
        
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()
        return mesh
    else:
        raise RuntimeError("No visualization library available")


def create_mesh_from_person_data(person_data, json_data, show_skeleton=True):
    """Create PyVista mesh and skeleton from person data."""
    vertices = np.array(person_data["mesh"]["vertices"], dtype=np.float32)
    faces = np.array(json_data["mesh_info"]["faces"], dtype=np.int32)
    
    n_faces = faces.shape[0]
    faces_pv = np.empty((n_faces, 4), dtype=np.int32)
    faces_pv[:, 0] = 3  # Triangle
    faces_pv[:, 1:] = faces
    faces_pv = faces_pv.flatten()
    
    mesh = pv.PolyData(vertices, faces_pv)
    
    skeleton_spheres = []
    if show_skeleton and "skeleton" in person_data:
        keypoints_3d = np.array(person_data["skeleton"]["keypoints_3d"], dtype=np.float32)
        valid_keypoints = keypoints_3d[keypoints_3d[:, 2] > 0] if len(keypoints_3d) > 0 else keypoints_3d
        
        if len(valid_keypoints) > 0:
            for kp in valid_keypoints:
                sphere = pv.Sphere(radius=0.02, center=kp)
                skeleton_spheres.append(sphere)
    
    return mesh, skeleton_spheres


def visualize_with_pyvista(json_data, frame_idx=0, person_idx=0, show_skeleton=True):
    """Visualize mesh and skeleton using PyVista."""
    if not PYVISTA_AVAILABLE:
        raise RuntimeError("PyVista not available")
    
    # Get frame data
    if frame_idx >= len(json_data["frames"]):
        frame_idx = len(json_data["frames"]) - 1
        print(f"Warning: Frame index out of range, using frame {frame_idx}")
    
    frame_data = json_data["frames"][frame_idx]
    
    if person_idx >= len(frame_data["persons"]):
        person_idx = len(frame_data["persons"]) - 1
        print(f"Warning: Person index out of range, using person {person_idx}")
    
    if len(frame_data["persons"]) == 0:
        print("Error: No persons in this frame")
        return
    
    person_data = frame_data["persons"][person_idx]
    
    # Create mesh and skeleton
    mesh, skeleton_spheres = create_mesh_from_person_data(person_data, json_data, show_skeleton)
    
    # Create plotter
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color='lightblue', opacity=1.0, show_edges=False)
    
    # Add skeleton spheres
    for sphere in skeleton_spheres:
        plotter.add_mesh(sphere, color='red', opacity=0.8)
    
    # Set up the plotter
    plotter.add_text(
        f"Frame {frame_idx} | Person {person_idx} | "
        f"Vertices: {len(mesh.points)} | Faces: {len(mesh.faces)//4}",
        font_size=12,
        position='upper_left'
    )
    
    plotter.show_axes()
    plotter.background_color = 'white'
    
    # Instructions
    print("\n" + "="*60)
    print("Interactive 3D Viewer Controls:")
    print("="*60)
    print("  Left Click + Drag: Rotate")
    print("  Right Click + Drag: Pan")
    print("  Scroll Wheel: Zoom in/out")
    print("  'q' or Close Window: Quit")
    print("="*60 + "\n")
    
    plotter.show()


def animate_all_frames(json_data, person_idx=0, show_skeleton=True, fps=30):
    """Animate through all frames using PyVista with interactive controls."""
    if not PYVISTA_AVAILABLE:
        raise RuntimeError("PyVista not available for animation")
    
    num_frames = len(json_data["frames"])
    if num_frames == 0:
        print("Error: No frames to animate")
        return
    
    print(f"Animating through {num_frames} frame(s) at {fps} FPS...")
    print("Loading frame data...")
    
    # Store mesh data for each frame
    all_vertices = []
    all_sphere_centers = []
    faces = np.array(json_data["mesh_info"]["faces"], dtype=np.int32)
    n_faces = faces.shape[0]
    faces_pv = np.empty((n_faces, 4), dtype=np.int32)
    faces_pv[:, 0] = 3
    faces_pv[:, 1:] = faces
    faces_pv = faces_pv.flatten()
    
    # Pre-load all frame data
    for frame_idx in range(num_frames):
        if frame_idx % 100 == 0:
            print(f"  Loading frame {frame_idx}/{num_frames}...")
        
        frame_data = json_data["frames"][frame_idx]
        if len(frame_data["persons"]) == 0:
            all_vertices.append(None)
            all_sphere_centers.append([])
            continue
        
        current_person_idx = person_idx if person_idx < len(frame_data["persons"]) else 0
        person_data = frame_data["persons"][current_person_idx]
        
        vertices = np.array(person_data["mesh"]["vertices"], dtype=np.float32)
        all_vertices.append(vertices)
        
        sphere_centers = []
        if show_skeleton and "skeleton" in person_data:
            keypoints_3d = np.array(person_data["skeleton"]["keypoints_3d"], dtype=np.float32)
            valid_keypoints = keypoints_3d[keypoints_3d[:, 2] > 0] if len(keypoints_3d) > 0 else keypoints_3d
            sphere_centers = valid_keypoints.tolist() if len(valid_keypoints) > 0 else []
        all_sphere_centers.append(sphere_centers)
    
    print("Frame data loaded. Starting animation...")
    
    # Create plotter with interactive mode enabled
    plotter = pv.Plotter()
    plotter.show_axes()
    plotter.background_color = 'white'
    
    # Store actors in a mutable container so callback can access them
    actors = {'mesh': None, 'spheres': []}
    
    # Create initial mesh
    if all_vertices[0] is not None:
        initial_mesh = pv.PolyData(all_vertices[0], faces_pv)
        actors['mesh'] = plotter.add_mesh(initial_mesh, color='lightblue', opacity=1.0, show_edges=False, name='mesh')
        
        # Add initial skeleton
        for center in all_sphere_centers[0]:
            sphere = pv.Sphere(radius=0.02, center=center)
            sphere_actor = plotter.add_mesh(sphere, color='red', opacity=0.8)
            actors['spheres'].append(sphere_actor)
    
    # Store text actor reference
    text_actor_ref = {'actor': None}
    
    # Add initial text
    text_actor_ref['actor'] = plotter.add_text(
        f"Frame 0/{num_frames-1} | Person {person_idx}",
        font_size=12,
        position='upper_left'
    )
    
    frame_count = [0]  # Use list to allow modification in callback
    
    def update_frame():
        """Update frame animation - called automatically by PyVista."""
        frame_count[0] = (frame_count[0] + 1) % num_frames
        current_frame = frame_count[0]
        
        # Update text by removing and re-adding (PyVista doesn't support direct text updates)
        if text_actor_ref['actor'] is not None:
            plotter.remove_actor(text_actor_ref['actor'])
        text_actor_ref['actor'] = plotter.add_text(
            f"Frame {current_frame}/{num_frames-1} | Person {person_idx}",
            font_size=12,
            position='upper_left'
        )
        
        if all_vertices[current_frame] is not None:
            # Update mesh by modifying the existing actor's data
            if actors['mesh'] is not None:
                new_mesh = pv.PolyData(all_vertices[current_frame], faces_pv)
                # Update the mesh data directly
                actors['mesh'].GetMapper().SetInputData(new_mesh)
            else:
                # Create new mesh if it doesn't exist
                new_mesh = pv.PolyData(all_vertices[current_frame], faces_pv)
                actors['mesh'] = plotter.add_mesh(new_mesh, color='lightblue', opacity=1.0, show_edges=False, name='mesh')
            
            # Update skeleton - remove old spheres, add new ones
            for actor in actors['spheres']:
                plotter.remove_actor(actor)
            actors['spheres'].clear()
            
            for center in all_sphere_centers[current_frame]:
                sphere = pv.Sphere(radius=0.02, center=center)
                sphere_actor = plotter.add_mesh(sphere, color='red', opacity=0.8)
                actors['spheres'].append(sphere_actor)
        else:
            # Empty frame - hide mesh
            if actors['mesh'] is not None:
                plotter.remove_actor(actors['mesh'])
                actors['mesh'] = None
            for actor in actors['spheres']:
                plotter.remove_actor(actor)
            actors['spheres'].clear()
    
    # Instructions
    print("\n" + "="*60)
    print("Interactive Animated 3D Viewer:")
    print("="*60)
    print("  Left Click + Drag: Rotate (works during animation!)")
    print("  Right Click + Drag: Pan (works during animation!)")
    print("  Scroll Wheel: Zoom in/out (works during animation!)")
    print("  Animation loops automatically at {} FPS".format(fps))
    print("  'q' or Close Window: Quit")
    print("="*60 + "\n")
    
    # Use timer event for animation (allows interaction during animation)
    interval_ms = int(1000 / fps)
    
    # Use a very large max_steps - our update_frame function handles looping
    # This allows the animation to loop indefinitely
    max_steps = 1000000  # Very large number - effectively infinite
    
    def timer_callback(step):
        """Timer callback for animation - called by PyVista."""
        # Update frame (our function manages the looping via modulo)
        update_frame()
    
    # Start the animation timer
    # duration is in milliseconds
    plotter.add_timer_event(max_steps=max_steps, duration=interval_ms, callback=timer_callback)
    plotter.show()


def visualize_with_open3d(json_data, frame_idx=0, person_idx=0, show_skeleton=True):
    """Visualize mesh and skeleton using Open3D."""
    if not OPEN3D_AVAILABLE:
        raise RuntimeError("Open3D not available")
    
    # Get frame data
    if frame_idx >= len(json_data["frames"]):
        frame_idx = len(json_data["frames"]) - 1
        print(f"Warning: Frame index out of range, using frame {frame_idx}")
    
    frame_data = json_data["frames"][frame_idx]
    
    if person_idx >= len(frame_data["persons"]):
        person_idx = len(frame_data["persons"]) - 1
        print(f"Warning: Person index out of range, using person {person_idx}")
    
    if len(frame_data["persons"]) == 0:
        print("Error: No persons in this frame")
        return
    
    person_data = frame_data["persons"][person_idx]
    
    # Get mesh data
    vertices = np.array(person_data["mesh"]["vertices"], dtype=np.float32)
    faces = np.array(json_data["mesh_info"]["faces"], dtype=np.int32)
    
    # Create mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.7, 0.8, 0.9])  # Light blue
    
    # Create visualization
    geometries = [mesh]
    
    # Add skeleton if requested
    if show_skeleton and "skeleton" in person_data:
        keypoints_3d = np.array(person_data["skeleton"]["keypoints_3d"], dtype=np.float32)
        valid_keypoints = keypoints_3d[keypoints_3d[:, 2] > 0] if len(keypoints_3d) > 0 else keypoints_3d
        
        if len(valid_keypoints) > 0:
            # Create point cloud for keypoints
            keypoint_cloud = o3d.geometry.PointCloud()
            keypoint_cloud.points = o3d.utility.Vector3dVector(valid_keypoints)
            keypoint_cloud.paint_uniform_color([1.0, 0.0, 0.0])  # Red
            
            # Create spheres for each keypoint
            for kp in valid_keypoints:
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
                sphere.translate(kp)
                sphere.paint_uniform_color([1.0, 0.0, 0.0])
                geometries.append(sphere)
    
    # Instructions
    print("\n" + "="*60)
    print("Interactive 3D Viewer Controls:")
    print("="*60)
    print("  Left Click + Drag: Rotate")
    print("  Right Click + Drag: Pan")
    print("  Scroll Wheel: Zoom in/out")
    print("  'Q' or Close Window: Quit")
    print("="*60 + "\n")
    
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"Frame {frame_idx} | Person {person_idx}",
        width=1024,
        height=768
    )


def export_video(json_data, output_path, person_idx=0, show_skeleton=True, fps=30, resolution=(1920, 1080)):
    """Export all frames as an MP4 video."""
    if not PYVISTA_AVAILABLE:
        raise RuntimeError("PyVista required for video export")
    
    num_frames = len(json_data["frames"])
    if num_frames == 0:
        print("Error: No frames to export")
        return
    
    print(f"Exporting {num_frames} frame(s) to video at {fps} FPS...")
    print(f"Resolution: {resolution[0]}x{resolution[1]}")
    
    # Prepare mesh data
    all_vertices = []
    all_sphere_centers = []
    faces = np.array(json_data["mesh_info"]["faces"], dtype=np.int32)
    n_faces = faces.shape[0]
    faces_pv = np.empty((n_faces, 4), dtype=np.int32)
    faces_pv[:, 0] = 3
    faces_pv[:, 1:] = faces
    faces_pv = faces_pv.flatten()
    
    # Load all frame data
    print("Loading frame data...")
    for frame_idx in range(num_frames):
        if frame_idx % 10 == 0:
            print(f"  Loading frame {frame_idx}/{num_frames}...")
        
        frame_data = json_data["frames"][frame_idx]
        if len(frame_data["persons"]) == 0:
            all_vertices.append(None)
            all_sphere_centers.append([])
            continue
        
        current_person_idx = person_idx if person_idx < len(frame_data["persons"]) else 0
        person_data = frame_data["persons"][current_person_idx]
        
        vertices = np.array(person_data["mesh"]["vertices"], dtype=np.float32)
        all_vertices.append(vertices)
        
        sphere_centers = []
        if show_skeleton and "skeleton" in person_data:
            keypoints_3d = np.array(person_data["skeleton"]["keypoints_3d"], dtype=np.float32)
            valid_keypoints = keypoints_3d[keypoints_3d[:, 2] > 0] if len(keypoints_3d) > 0 else keypoints_3d
            sphere_centers = valid_keypoints.tolist() if len(valid_keypoints) > 0 else []
        all_sphere_centers.append(sphere_centers)
    
    print("Frame data loaded. Rendering frames...")
    
    # Create temporary directory for frames
    temp_dir = Path(tempfile.mkdtemp())
    print(f"Temporary frames directory: {temp_dir}")
    
    try:
        # Create plotter (offscreen rendering)
        plotter = pv.Plotter(off_screen=True, window_size=resolution)
        plotter.background_color = 'white'
        
        # Calculate camera position from first valid frame
        camera_set = False
        camera_position = None
        camera_focal_point = None
        camera_view_up = None
        
        # Render each frame
        for frame_idx in range(num_frames):
            if frame_idx % 10 == 0:
                print(f"  Rendering frame {frame_idx}/{num_frames}...")
            
            # Clear previous frame
            plotter.clear()
            plotter.show_axes()
            
            if all_vertices[frame_idx] is not None:
                # Create mesh
                mesh = pv.PolyData(all_vertices[frame_idx], faces_pv)
                plotter.add_mesh(mesh, color='lightblue', opacity=1.0, show_edges=False)
                
                # Add skeleton
                if show_skeleton:
                    for center in all_sphere_centers[frame_idx]:
                        sphere = pv.Sphere(radius=0.02, center=center)
                        plotter.add_mesh(sphere, color='red', opacity=0.8)
                
                # Add frame text
                plotter.add_text(
                    f"Frame {frame_idx}/{num_frames-1} | Person {person_idx}",
                    font_size=20,
                    position='upper_left'
                )
                
                # Set camera position (chest-level front view) - only once on first valid frame
                if not camera_set:
                    # Calculate mesh bounds to position camera
                    bounds = mesh.bounds
                    center_x = (bounds[0] + bounds[1]) / 2
                    center_y = (bounds[2] + bounds[3]) / 2
                    center_z = (bounds[4] + bounds[5]) / 2
                    
                    # Calculate mesh size for distance
                    size_x = bounds[1] - bounds[0]
                    size_y = bounds[3] - bounds[2]
                    size_z = bounds[5] - bounds[4]
                    max_size = max(size_x, size_y, size_z)
                    
                    # Determine which axis is "up" (height) - usually the one with largest range
                    # For a standing person, height should be the largest dimension
                    axis_sizes = {'x': size_x, 'y': size_y, 'z': size_z}
                    height_axis = max(axis_sizes, key=axis_sizes.get)
                    
                    # For front view with feet at bottom:
                    # Camera should be perpendicular to the height axis, looking at the person
                    # Focal point at chest level (middle of height axis)
                    distance = max_size * 2.5
                    
                    if height_axis == 'y':
                        # Y is height (most common) - person stands along Y axis
                        # Camera in front along negative Z (opposite side), at chest height (center_y)
                        # View up is negative Y to flip orientation (feet at bottom)
                        camera_pos = [center_x, center_y, center_z - distance]  # Negative Z = front view
                        focal_point = [center_x, center_y, center_z]
                        view_up = [0, -1, 0]  # Negative Y = flip to get feet at bottom
                    elif height_axis == 'z':
                        # Z is height - person stands along Z axis
                        # Camera in front along negative Y, at chest height (center_z)
                        camera_pos = [center_x, center_y - distance, center_z]
                        focal_point = [center_x, center_y, center_z]
                        view_up = [0, 0, -1]  # Negative Z = flip
                    else:  # height_axis == 'x'
                        # X is height - person stands along X axis
                        # Camera in front along negative Z, at chest height (center_x)
                        camera_pos = [center_x, center_y - distance, center_z]
                        focal_point = [center_x, center_y, center_z]
                        view_up = [-1, 0, 0]  # Negative X = flip
                    
                    # Set camera position
                    plotter.camera_position = (camera_pos, focal_point, view_up)
                    plotter.camera.zoom(1.0)
                    
                    # Store for reuse
                    camera_position = (camera_pos, focal_point, view_up)
                    camera_set = True
                else:
                    # Maintain same camera position for all frames
                    plotter.camera_position = camera_position
            
            # Render to image
            frame_path = temp_dir / f"frame_{frame_idx:06d}.png"
            plotter.screenshot(str(frame_path))
        
        print("All frames rendered. Creating video...")
        
        # Create video from frames
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            resolution
        )
        
        # Read frames and write to video
        for frame_idx in range(num_frames):
            if frame_idx % 10 == 0:
                print(f"  Writing frame {frame_idx}/{num_frames} to video...")
            
            frame_path = temp_dir / f"frame_{frame_idx:06d}.png"
            if frame_path.exists():
                frame = cv2.imread(str(frame_path))
                if frame is not None:
                    # Resize if needed
                    if frame.shape[1] != resolution[0] or frame.shape[0] != resolution[1]:
                        frame = cv2.resize(frame, resolution)
                    video_writer.write(frame)
        
        video_writer.release()
        print(f"Video saved to: {output_path}")
        
    finally:
        # Clean up temporary directory
        print("Cleaning up temporary files...")
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("Video export complete!")


def visualize_all_persons(json_data, frame_idx=0, show_skeleton=True):
    """Visualize all persons in a frame at once."""
    if frame_idx >= len(json_data["frames"]):
        frame_idx = len(json_data["frames"]) - 1
        print(f"Warning: Frame index out of range, using frame {frame_idx}")
    
    frame_data = json_data["frames"][frame_idx]
    
    if len(frame_data["persons"]) == 0:
        print("Error: No persons in this frame")
        return
    
    if PYVISTA_AVAILABLE:
        plotter = pv.Plotter()
        colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightpink']
        
        for person_idx, person_data in enumerate(frame_data["persons"]):
            vertices = np.array(person_data["mesh"]["vertices"], dtype=np.float32)
            faces = np.array(json_data["mesh_info"]["faces"], dtype=np.int32)
            
            # Offset each person slightly to avoid overlap
            offset = np.array([person_idx * 0.5, 0, 0])
            vertices = vertices + offset
            
            n_faces = faces.shape[0]
            faces_pv = np.empty((n_faces, 4), dtype=np.int32)
            faces_pv[:, 0] = 3
            faces_pv[:, 1:] = faces
            faces_pv = faces_pv.flatten()
            
            mesh = pv.PolyData(vertices, faces_pv)
            color = colors[person_idx % len(colors)]
            plotter.add_mesh(mesh, color=color, opacity=1.0, show_edges=False)
            
            if show_skeleton and "skeleton" in person_data:
                keypoints_3d = np.array(person_data["skeleton"]["keypoints_3d"], dtype=np.float32)
                valid_keypoints = keypoints_3d[keypoints_3d[:, 2] > 0] if len(keypoints_3d) > 0 else keypoints_3d
                
                if len(valid_keypoints) > 0:
                    valid_keypoints = valid_keypoints + offset
                    for kp in valid_keypoints:
                        sphere = pv.Sphere(radius=0.02, center=kp)
                        plotter.add_mesh(sphere, color='red', opacity=0.8)
        
        plotter.add_text(
            f"Frame {frame_idx} | {len(frame_data['persons'])} person(s)",
            font_size=12,
            position='upper_left'
        )
        plotter.show_axes()
        plotter.background_color = 'white'
        plotter.show()
    elif OPEN3D_AVAILABLE:
        geometries = []
        colors = [[0.7, 0.8, 0.9], [0.7, 0.9, 0.8], [0.9, 0.9, 0.7], [0.9, 0.7, 0.7], [0.9, 0.7, 0.9]]
        
        for person_idx, person_data in enumerate(frame_data["persons"]):
            vertices = np.array(person_data["mesh"]["vertices"], dtype=np.float32)
            faces = np.array(json_data["mesh_info"]["faces"], dtype=np.int32)
            
            offset = np.array([person_idx * 0.5, 0, 0])
            vertices = vertices + offset
            
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color(colors[person_idx % len(colors)])
            geometries.append(mesh)
            
            if show_skeleton and "skeleton" in person_data:
                keypoints_3d = np.array(person_data["skeleton"]["keypoints_3d"], dtype=np.float32)
                valid_keypoints = keypoints_3d[keypoints_3d[:, 2] > 0] if len(keypoints_3d) > 0 else keypoints_3d
                
                if len(valid_keypoints) > 0:
                    valid_keypoints = valid_keypoints + offset
                    for kp in valid_keypoints:
                        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
                        sphere.translate(kp)
                        sphere.paint_uniform_color([1.0, 0.0, 0.0])
                        geometries.append(sphere)
        
        o3d.visualization.draw_geometries(
            geometries,
            window_name=f"Frame {frame_idx} | {len(frame_data['persons'])} person(s)",
            width=1024,
            height=768
        )


def main():
    parser = argparse.ArgumentParser(
        description='Interactive 3D renderer for SAM-3d-body JSON output',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View first frame, first person
  python render_3d.py --json outputs/video_data.json
  
  # View specific frame and person
  python render_3d.py --json outputs/video_data.json --frame 10 --person 0
  
  # View all persons in a frame
  python render_3d.py --json outputs/video_data.json --frame 5 --all-persons
  
  # Hide skeleton, only show mesh
  python render_3d.py --json outputs/video_data.json --no-skeleton
  
  # Animate through all frames
  python render_3d.py --json outputs/video_data.json --animate
  
  # Animate at custom FPS (default is 30 FPS)
  python render_3d.py --json outputs/video_data.json --animate --fps 15
  
  # Export all frames as MP4 video
  python render_3d.py --json outputs/video_data.json --export-video output.mp4
  
  # Export video with custom resolution
  python render_3d.py --json outputs/video_data.json --export-video output.mp4 --resolution 1280x720
        """
    )
    parser.add_argument(
        '--json',
        type=str,
        required=True,
        help='Path to JSON data file (output from process_video.py)'
    )
    parser.add_argument(
        '--frame',
        type=int,
        default=0,
        help='Frame index to visualize (default: 0)'
    )
    parser.add_argument(
        '--person',
        type=int,
        default=0,
        help='Person index to visualize (default: 0, ignored if --all-persons is used)'
    )
    parser.add_argument(
        '--all-persons',
        action='store_true',
        help='Show all persons in the frame (default: show only one person)'
    )
    parser.add_argument(
        '--no-skeleton',
        action='store_true',
        help='Hide skeleton keypoints (default: show skeleton)'
    )
    parser.add_argument(
        '--animate',
        action='store_true',
        help='Animate through all frames (default: show single frame)'
    )
    parser.add_argument(
        '--fps',
        type=float,
        default=30.0,
        help='Animation FPS when using --animate (default: 30.0)'
    )
    parser.add_argument(
        '--export-video',
        type=str,
        default=None,
        help='Export all frames as MP4 video to specified path (e.g., output.mp4)'
    )
    parser.add_argument(
        '--resolution',
        type=str,
        default='1920x1080',
        help='Video resolution for export (default: 1920x1080)'
    )
    
    args = parser.parse_args()
    
    # Load JSON data
    json_path = Path(args.json)
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}")
        sys.exit(1)
    
    print(f"Loading JSON data from: {json_path}")
    json_data = load_json_data(json_path)
    
    # Print info
    num_frames = len(json_data["frames"])
    print(f"Loaded {num_frames} frame(s)")
    
    if num_frames == 0:
        print("Error: No frames in JSON file")
        sys.exit(1)
    
    frame_data = json_data["frames"][args.frame]
    num_persons = len(frame_data["persons"])
    print(f"Frame {args.frame} has {num_persons} person(s)")
    
    if num_persons == 0:
        print("Error: No persons in this frame")
        sys.exit(1)
    
    show_skeleton = not args.no_skeleton
    
    # Export video if requested
    if args.export_video:
        try:
            # Parse resolution
            res_parts = args.resolution.split('x')
            if len(res_parts) != 2:
                print(f"Error: Invalid resolution format. Use WIDTHxHEIGHT (e.g., 1920x1080)")
                sys.exit(1)
            resolution = (int(res_parts[0]), int(res_parts[1]))
            
            if PYVISTA_AVAILABLE:
                export_video(json_data, args.export_video, args.person, show_skeleton, args.fps, resolution)
            else:
                print("Error: Video export requires PyVista. Please install: pip install pyvista")
                sys.exit(1)
            return
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error during video export: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # Visualize
    try:
        if args.animate:
            if PYVISTA_AVAILABLE:
                animate_all_frames(json_data, args.person, show_skeleton, args.fps)
            else:
                print("Error: Animation requires PyVista. Please install: pip install pyvista")
                sys.exit(1)
        elif args.all_persons:
            if PYVISTA_AVAILABLE:
                visualize_all_persons(json_data, args.frame, show_skeleton)
            elif OPEN3D_AVAILABLE:
                visualize_all_persons(json_data, args.frame, show_skeleton)
        else:
            if PYVISTA_AVAILABLE:
                visualize_with_pyvista(json_data, args.frame, args.person, show_skeleton)
            elif OPEN3D_AVAILABLE:
                visualize_with_open3d(json_data, args.frame, args.person, show_skeleton)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
