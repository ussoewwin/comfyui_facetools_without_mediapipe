"""
Common face mesh generation functions using InsightFace
"""

import warnings
import numpy as np
import cv2

# Simple cache for InsightFace app instances
_insightface_cache = {}

min_face_size_pixels: int = 64
f_thick = 2
f_rad = 1

# Color definitions (BGR format for OpenCV)
right_eye_color = (180, 200, 10)  # BGR: (B, G, R)
left_eye_color = (10, 200, 180)
right_eyebrow_color = (180, 220, 10)
left_eyebrow_color = (10, 220, 180)
mouth_color = (10, 180, 10)
head_color = (10, 200, 10)


def reverse_channels(image):
    """Given a numpy array in RGB form, convert to BGR.  Will also convert from BGR to RGB."""
    return image[:, :, ::-1]


def _draw_face_mesh_from_keypoints(image, kps, bbox, img_width, img_height):
    """
    Draw face mesh using InsightFace keypoints.
    InsightFace keypoints order: [right_eye, left_eye, nose, mouth_left, mouth_right]
    """
    # Convert keypoints to pixel coordinates
    right_eye = kps[0].astype(int)
    left_eye = kps[1].astype(int)
    nose = kps[2].astype(int)
    mouth_left = kps[3].astype(int)
    mouth_right = kps[4].astype(int)
    
    # Calculate face dimensions from bbox
    face_width = bbox[2] - bbox[0]
    face_height = bbox[3] - bbox[1]
    
    # Draw face oval (head outline)
    face_center_x = int((bbox[0] + bbox[2]) / 2)
    face_center_y = int((bbox[1] + bbox[3]) / 2)
    axes = (int(face_width * 0.5), int(face_height * 0.6))
    cv2.ellipse(image, (face_center_x, face_center_y), axes, 0, 0, 360, head_color, f_thick)
    
    # Draw left eye
    eye_radius = int(face_width * 0.03)
    cv2.circle(image, tuple(left_eye), eye_radius, left_eye_color, f_thick)
    
    # Draw right eye
    cv2.circle(image, tuple(right_eye), eye_radius, right_eye_color, f_thick)
    
    # Draw left eyebrow (above left eye)
    eyebrow_y = max(0, left_eye[1] - int(face_height * 0.05))
    eyebrow_center = (left_eye[0], eyebrow_y)
    eyebrow_axes = (int(face_width * 0.04), int(face_height * 0.01))
    cv2.ellipse(image, eyebrow_center, eyebrow_axes, 0, 0, 360, left_eyebrow_color, f_thick)
    
    # Draw right eyebrow (above right eye)
    eyebrow_y = max(0, right_eye[1] - int(face_height * 0.05))
    eyebrow_center = (right_eye[0], eyebrow_y)
    cv2.ellipse(image, eyebrow_center, eyebrow_axes, 0, 0, 360, right_eyebrow_color, f_thick)
    
    # Draw lips
    mouth_center = ((mouth_left[0] + mouth_right[0]) // 2, 
                   (mouth_left[1] + mouth_right[1]) // 2)
    mouth_width = int(np.linalg.norm(mouth_right - mouth_left) * 0.8)
    mouth_height = int(mouth_width * 0.4)
    mouth_axes = (mouth_width // 2, mouth_height // 2)
    cv2.ellipse(image, mouth_center, mouth_axes, 0, 0, 360, mouth_color, f_thick)
    
    # Draw connection lines
    # Eye to eye
    cv2.line(image, tuple(left_eye), tuple(right_eye), head_color, 1)
    # Nose to mouth center
    cv2.line(image, tuple(nose), mouth_center, head_color, 1)
    # Left eye to nose
    cv2.line(image, tuple(left_eye), tuple(nose), head_color, 1)
    # Right eye to nose
    cv2.line(image, tuple(right_eye), tuple(nose), head_color, 1)


def generate_annotation(
        img_rgb,
        max_faces: int,
        min_confidence: float
):
    """
    Find up to 'max_faces' inside the provided input image using InsightFace.
    If min_face_size_pixels is provided and nonzero it will be used to filter faces that occupy less than this many
    pixels in the image.
    """
    try:
        import insightface
    except ImportError:
        import subprocess
        import sys
        print("Installing insightface...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "insightface"])
        import insightface
    
    img_height, img_width, img_channels = img_rgb.shape
    assert(img_channels == 3)
    
    # Initialize InsightFace app
    cache_key = "facetools_face_mesh_insightface_app"
    if cache_key in _insightface_cache:
        app = _insightface_cache[cache_key]
    else:
        app = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        _insightface_cache[cache_key] = app
    
    # Convert RGB to BGR for OpenCV
    img_bgr = reverse_channels(img_rgb.copy())
    
    # Detect faces using InsightFace
    faces = app.get(img_bgr)
    
    if len(faces) == 0:
        print("No faces detected in image for InsightFace face mesh annotator.")
        return np.zeros_like(img_rgb)
    
    # Filter faces by confidence and size
    filtered_faces = []
    for face in faces:
        # Check detection confidence
        if hasattr(face, 'det_score') and face.det_score < min_confidence:
            continue
        
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        
        # Filter by face size
        if min_face_size_pixels > 0:
            face_width_pixels = x2 - x1
            face_height_pixels = y2 - y1
            face_size = min(face_width_pixels, face_height_pixels)
            if face_size < min_face_size_pixels:
                continue
        
        filtered_faces.append(face)
    
    # Limit to max_faces
    filtered_faces = filtered_faces[:max_faces]
    
    if len(filtered_faces) == 0:
        return np.zeros_like(img_rgb)
    
    # Create empty annotation image (BGR format)
    empty = np.zeros_like(img_bgr)
    
    # Draw detected faces
    for face in filtered_faces:
        bbox = face.bbox.astype(int)
        kps = face.kps  # Keypoints: [right_eye, left_eye, nose, mouth_left, mouth_right]
        
        _draw_face_mesh_from_keypoints(empty, kps, bbox, img_width, img_height)
    
    # Convert BGR back to RGB
    empty = reverse_channels(empty).copy()
    
    return empty

