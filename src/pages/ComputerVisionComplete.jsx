import { useState } from 'react'

function ComputerVisionComplete() {
  const [activeSection, setActiveSection] = useState(0)
  const [expandedCode, setExpandedCode] = useState({})

  const toggleCode = (sectionId, codeId) => {
    const key = `${sectionId}-${codeId}`
    setExpandedCode(prev => ({
      ...prev,
      [key]: !prev[key]
    }))
  }

  const sections = [
    {
      id: 'cv-fundamentals',
      title: 'Computer Vision Fundamentals',
      icon: '👁️',
      description: 'Master the foundations of computer vision and image processing',
      content: `
        Computer Vision enables computers to interpret and understand visual information from the world.
        Learn fundamental concepts, image processing techniques, and modern deep learning approaches.
      `,
      keyTopics: [
        'Image Processing Basics',
        'OpenCV Library Mastery',
        'Feature Detection and Extraction',
        'Object Detection and Recognition',
        'Image Segmentation Techniques',
        'Edge Detection and Contours',
        'Color Spaces and Transformations',
        'Morphological Operations'
      ],
      codeExamples: [
        {
          title: 'OpenCV Basics - Image Operations',
          description: 'Essential OpenCV operations for image processing',
          code: `import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

class ImageProcessor:
    def __init__(self):
        self.processed_images = []
    
    def load_image(self, image_path):
        """Load image using OpenCV"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        return img
    
    def display_images(self, images, titles, figsize=(15, 10)):
        """Display multiple images in a grid"""
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.ravel()
        
        for i, (img, title) in enumerate(zip(images, titles)):
            if len(img.shape) == 3:
                # Convert BGR to RGB for matplotlib
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[i].imshow(img_rgb)
            else:
                axes[i].imshow(img, cmap='gray')
            axes[i].set_title(title)
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def basic_operations(self, image_path):
        """Demonstrate basic image operations"""
        # Load original image
        original = self.load_image(image_path)
        height, width, channels = original.shape
        print(f"Image shape: {height}x{width}x{channels}")
        
        # 1. Convert to grayscale
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        
        # 2. Resize image
        resized = cv2.resize(original, (width//2, height//2))
        
        # 3. Apply Gaussian blur
        blurred = cv2.GaussianBlur(original, (15, 15), 0)
        
        # 4. Edge detection using Canny
        edges = cv2.Canny(gray, 100, 200)
        
        # 5. Histogram equalization
        gray_eq = cv2.equalizeHist(gray)
        
        # 6. Color space conversion (HSV)
        hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
        
        images = [original, gray, resized, blurred, edges, hsv]
        titles = ['Original', 'Grayscale', 'Resized', 'Blurred', 'Edges', 'HSV']
        
        self.display_images(images, titles)
        
        return {
            'original': original,
            'grayscale': gray,
            'edges': edges,
            'blurred': blurred
        }
    
    def color_filtering(self, image):
        """Advanced color filtering techniques"""
        # Convert to HSV for better color filtering
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges (example: blue color)
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        
        # Create mask for blue color
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Apply mask to original image
        blue_filtered = cv2.bitwise_and(image, image, mask=blue_mask)
        
        # Red color filtering
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = red_mask1 + red_mask2
        
        red_filtered = cv2.bitwise_and(image, image, mask=red_mask)
        
        images = [image, blue_mask, blue_filtered, red_mask, red_filtered, hsv]
        titles = ['Original', 'Blue Mask', 'Blue Filtered', 'Red Mask', 'Red Filtered', 'HSV']
        
        self.display_images(images, titles)
        
        return blue_filtered, red_filtered

# Usage example
processor = ImageProcessor()

# Example usage (replace with your image path)
# results = processor.basic_operations('sample_image.jpg')
# blue_filtered, red_filtered = processor.color_filtering(results['original'])

print("Computer Vision basics demonstration completed!")`
        },
        {
          title: 'Feature Detection and Matching',
          description: 'Advanced feature detection using SIFT, ORB, and feature matching',
          code: `import cv2
import numpy as np
import matplotlib.pyplot as plt

class FeatureDetector:
    def __init__(self):
        # Initialize different feature detectors
        self.sift = cv2.SIFT_create()
        self.orb = cv2.ORB_create()
        self.surf = cv2.xfeatures2d.SURF_create(400)  # Requires opencv-contrib-python
        
    def detect_sift_features(self, image):
        """Detect SIFT features in an image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect keypoints and descriptors
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        # Draw keypoints
        img_with_keypoints = cv2.drawKeypoints(
            image, keypoints, None, 
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        print(f"Found {len(keypoints)} SIFT keypoints")
        
        return keypoints, descriptors, img_with_keypoints
    
    def detect_orb_features(self, image):
        """Detect ORB features in an image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect keypoints and descriptors
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        # Draw keypoints
        img_with_keypoints = cv2.drawKeypoints(
            image, keypoints, None, color=(0, 255, 0)
        )
        
        print(f"Found {len(keypoints)} ORB keypoints")
        
        return keypoints, descriptors, img_with_keypoints
    
    def match_features(self, img1, img2, method='sift'):
        """Match features between two images"""
        if method == 'sift':
            detector = self.sift
            matcher = cv2.BFMatcher()
        elif method == 'orb':
            detector = self.orb
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Detect keypoints and descriptors
        kp1, des1 = detector.detectAndCompute(gray1, None)
        kp2, des2 = detector.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None:
            print("No descriptors found!")
            return None
        
        # Match descriptors
        if method == 'sift':
            matches = matcher.knnMatch(des1, des2, k=2)
            
            # Apply Lowe's ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append([m])
            
            # Draw matches
            img_matches = cv2.drawMatchesKnn(
                img1, kp1, img2, kp2, good_matches, None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            
        else:  # ORB
            matches = matcher.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Draw top matches
            img_matches = cv2.drawMatches(
                img1, kp1, img2, kp2, matches[:50], None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
        
        print(f"Found {len(good_matches if method == 'sift' else matches)} matches using {method.upper()}")
        
        return img_matches, kp1, kp2, good_matches if method == 'sift' else matches
    
    def find_homography(self, img1, img2):
        """Find homography between two images for object detection"""
        # Detect and match SIFT features
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        kp1, des1 = self.sift.detectAndCompute(gray1, None)
        kp2, des2 = self.sift.detectAndCompute(gray2, None)
        
        # Match features
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        if len(good_matches) < 10:
            print("Not enough good matches found!")
            return None
        
        # Extract matching points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        # Get corners of the first image
        h, w = gray1.shape
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        
        # Transform corners to second image
        transformed_corners = cv2.perspectiveTransform(corners, M)
        
        # Draw bounding box on second image
        result_img = img2.copy()
        cv2.polylines(result_img, [np.int32(transformed_corners)], True, (0, 255, 0), 3)
        
        return result_img, M, transformed_corners
    
    def corner_detection(self, image):
        """Detect corners using Harris and Shi-Tomasi methods"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        
        # Harris corner detection
        harris_corners = cv2.cornerHarris(gray, 2, 3, 0.04)
        harris_corners = cv2.dilate(harris_corners, None)
        
        # Threshold for corner detection
        img_harris = image.copy()
        img_harris[harris_corners > 0.01 * harris_corners.max()] = [0, 0, 255]
        
        # Shi-Tomasi corner detection
        corners = cv2.goodFeaturesToTrack(
            np.uint8(gray), maxCorners=100, qualityLevel=0.01, minDistance=10
        )
        
        img_shi_tomasi = image.copy()
        if corners is not None:
            corners = np.int0(corners)
            for corner in corners:
                x, y = corner.ravel()
                cv2.circle(img_shi_tomasi, (x, y), 3, (0, 255, 0), -1)
        
        return img_harris, img_shi_tomasi

# Comprehensive feature detection example
def demonstrate_feature_detection():
    detector = FeatureDetector()
    
    # Load sample images (replace with your own)
    # img1 = cv2.imread('object.jpg')
    # img2 = cv2.imread('scene.jpg')
    
    # Example with synthetic data
    # Create sample images for demonstration
    img1 = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.rectangle(img1, (100, 100), (300, 300), (255, 255, 255), -1)
    cv2.circle(img1, (200, 200), 50, (0, 0, 255), -1)
    
    img2 = np.zeros((500, 500, 3), dtype=np.uint8)
    cv2.rectangle(img2, (150, 150), (350, 350), (255, 255, 255), -1)
    cv2.circle(img2, (250, 250), 50, (0, 0, 255), -1)
    
    # Detect SIFT features
    kp1, des1, img1_sift = detector.detect_sift_features(img1)
    kp2, des2, img2_sift = detector.detect_orb_features(img2)
    
    # Match features
    matches_img, _, _, _ = detector.match_features(img1, img2, 'sift')
    
    # Corner detection
    harris_img, shi_tomasi_img = detector.corner_detection(img1)
    
    print("Feature detection demonstration completed!")
    
    return {
        'sift_features': img1_sift,
        'orb_features': img2_sift,
        'matches': matches_img,
        'harris_corners': harris_img,
        'shi_tomasi_corners': shi_tomasi_img
    }

# Run demonstration
# results = demonstrate_feature_detection()`
        }
      ]
    },
    {
      id: 'object-detection',
      title: 'Object Detection & Recognition',
      icon: '🎯',
      description: 'Implement modern object detection using YOLO, R-CNN, and deep learning',
      content: `
        Object detection combines classification and localization to identify and locate objects in images.
        Learn traditional methods and modern deep learning approaches for real-time object detection.
      `,
      keyTopics: [
        'YOLO (You Only Look Once)',
        'R-CNN and Fast R-CNN',
        'SSD (Single Shot Detector)',
        'Haar Cascades for Face Detection',
        'Template Matching',
        'Contour Detection and Analysis',
        'Real-time Object Tracking',
        'Custom Object Detection Models'
      ],
      codeExamples: [
        {
          title: 'YOLO Object Detection',
          description: 'Implement real-time object detection using YOLO',
          code: `import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class YOLODetector:
    def __init__(self, config_path, weights_path, classes_path):
        """
        Initialize YOLO detector
        
        Args:
            config_path: Path to YOLO config file (.cfg)
            weights_path: Path to YOLO weights file (.weights)
            classes_path: Path to classes file (.names)
        """
        # Load YOLO network
        self.net = cv2.dnn.readNet(weights_path, config_path)
        
        # Load class names
        with open(classes_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Get output layer names
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # Generate random colors for each class
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
    
    def detect_objects(self, image, confidence_threshold=0.5, nms_threshold=0.4):
        """
        Detect objects in an image
        
        Args:
            image: Input image
            confidence_threshold: Minimum confidence for detection
            nms_threshold: Non-maximum suppression threshold
        
        Returns:
            Annotated image with bounding boxes and labels
        """
        height, width, channels = image.shape
        
        # Prepare image for YOLO
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        
        # Run forward pass
        outputs = self.net.forward(self.output_layers)
        
        # Extract information from outputs
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > confidence_threshold:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
        
        # Draw bounding boxes and labels
        result_image = image.copy()
        
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                confidence = confidences[i]
                color = self.colors[class_ids[i]]
                
                # Draw bounding box
                cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
                
                # Draw label with confidence
                label_text = f"{label}: {confidence:.2f}"
                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                
                cv2.rectangle(result_image, (x, y - label_size[1] - 10), 
                            (x + label_size[0], y), color, -1)
                cv2.putText(result_image, label_text, (x, y - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return result_image, len(indexes) if len(indexes) > 0 else 0
    
    def detect_in_video(self, video_source=0, output_path=None):
        """
        Real-time object detection in video
        
        Args:
            video_source: Video file path or camera index (0 for webcam)
            output_path: Path to save output video (optional)
        """
        cap = cv2.VideoCapture(video_source)
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_fps = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = cv2.getTickCount()
            
            # Detect objects
            result_frame, num_objects = self.detect_objects(frame)
            
            # Calculate FPS
            end_time = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (end_time - start_time)
            total_fps += fps
            frame_count += 1
            
            # Draw FPS on frame
            cv2.putText(result_frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(result_frame, f"Objects: {num_objects}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('YOLO Object Detection', result_frame)
            
            if output_path:
                out.write(result_frame)
            
            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        avg_fps = total_fps / frame_count if frame_count > 0 else 0
        print(f"Average FPS: {avg_fps:.2f}")

class TraditionalObjectDetector:
    def __init__(self):
        # Initialize cascade classifiers for face and eye detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
    
    def detect_faces(self, image):
        """Detect faces using Haar cascades"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        result_image = image.copy()
        
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Region of interest for eyes
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = result_image[y:y+h, x:x+w]
            
            # Detect eyes within face region
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
        return result_image, len(faces)
    
    def template_matching(self, image, template):
        """Perform template matching"""
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        
        # Perform template matching
        result = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED)
        
        # Find locations where match quality is above threshold
        threshold = 0.8
        locations = np.where(result >= threshold)
        
        # Draw bounding boxes for matches
        result_image = image.copy()
        template_height, template_width = gray_template.shape
        
        for pt in zip(*locations[::-1]):
            bottom_right = (pt[0] + template_width, pt[1] + template_height)
            cv2.rectangle(result_image, pt, bottom_right, (0, 255, 0), 2)
        
        return result_image, len(locations[0])

# Usage example
def demonstrate_object_detection():
    # Traditional methods
    traditional_detector = TraditionalObjectDetector()
    
    # Create sample image for demonstration
    sample_image = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.rectangle(sample_image, (100, 100), (200, 200), (255, 255, 255), -1)
    cv2.circle(sample_image, (300, 150), 50, (0, 255, 0), -1)
    cv2.putText(sample_image, "Sample", (400, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Face detection (works better with real images containing faces)
    faces_detected, num_faces = traditional_detector.detect_faces(sample_image)
    
    print(f"Traditional object detection demonstrated!")
    print(f"Detected {num_faces} faces")
    
    # For YOLO detection, you would need to download the model files:
    # yolo_detector = YOLODetector('yolo.cfg', 'yolo.weights', 'coco.names')
    # objects_detected, num_objects = yolo_detector.detect_objects(sample_image)
    
    return faces_detected

# Run demonstration
# result = demonstrate_object_detection()`
        }
      ]
    },
    {
      id: 'image-segmentation',
      title: 'Image Segmentation & Analysis',
      icon: '🔍',
      description: 'Advanced segmentation techniques including semantic and instance segmentation',
      content: `
        Image segmentation divides an image into meaningful segments or regions.
        Learn watershed algorithms, region growing, clustering-based segmentation, and deep learning approaches.
      `,
      keyTopics: [
        'Semantic Segmentation',
        'Instance Segmentation',
        'Watershed Algorithm',
        'K-means Clustering Segmentation',
        'Region Growing Techniques',
        'GrabCut Algorithm',
        'Mask R-CNN Implementation',
        'Medical Image Segmentation'
      ],
      codeExamples: [
        {
          title: 'Advanced Segmentation Techniques',
          description: 'Implement various segmentation algorithms',
          code: `import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import segmentation, measure
from scipy import ndimage

class ImageSegmenter:
    def __init__(self):
        pass
    
    def watershed_segmentation(self, image):
        """Implement watershed segmentation"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Noise removal using morphological operations
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labeling
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1  # Add 1 to all labels so background is not 0, but 1
        markers[unknown == 255] = 0  # Mark unknown regions as 0
        
        # Apply watershed
        markers = cv2.watershed(image, markers)
        
        # Create result image
        result = image.copy()
        result[markers == -1] = [255, 0, 0]  # Mark boundaries in red
        
        return result, markers
    
    def kmeans_segmentation(self, image, k=3):
        """K-means clustering based segmentation"""
        # Reshape image to be a list of pixels
        pixel_values = image.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)
        
        # Define criteria and apply KMeans
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert centers to uint8
        centers = np.uint8(centers)
        
        # Map each pixel to its cluster center
        segmented_data = centers[labels.flatten()]
        segmented_image = segmented_data.reshape(image.shape)
        
        return segmented_image, centers, labels.reshape(image.shape[:2])
    
    def grabcut_segmentation(self, image, rectangle):
        """GrabCut algorithm for foreground extraction"""
        mask = np.zeros(image.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Apply GrabCut
        cv2.grabCut(image, mask, rectangle, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        
        # Modify mask to get final foreground
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        result = image * mask2[:, :, np.newaxis]
        
        return result, mask2
    
    def region_growing(self, image, seed_point, threshold=10):
        """Simple region growing algorithm"""
        h, w = image.shape[:2]
        segmented = np.zeros((h, w), dtype=np.uint8)
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Initialize
        seed_value = gray[seed_point[1], seed_point[0]]
        stack = [seed_point]
        segmented[seed_point[1], seed_point[0]] = 255
        
        # 8-connectivity directions
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        while stack:
            x, y = stack.pop()
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                if (0 <= nx < w and 0 <= ny < h and 
                    segmented[ny, nx] == 0 and
                    abs(int(gray[ny, nx]) - int(seed_value)) < threshold):
                    
                    segmented[ny, nx] = 255
                    stack.append((nx, ny))
        
        return segmented
    
    def superpixel_segmentation(self, image, n_segments=100):
        """SLIC superpixel segmentation"""
        # Convert BGR to RGB for skimage
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply SLIC
        segments = segmentation.slic(image_rgb, n_segments=n_segments, compactness=10, sigma=1)
        
        # Mark boundaries
        boundary_image = segmentation.mark_boundaries(image_rgb, segments)
        boundary_image = (boundary_image * 255).astype(np.uint8)
        
        # Convert back to BGR
        boundary_image = cv2.cvtColor(boundary_image, cv2.COLOR_RGB2BGR)
        
        return boundary_image, segments
    
    def semantic_segmentation_simple(self, image):
        """Simple color-based semantic segmentation"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for different objects
        color_ranges = {
            'sky': [(100, 50, 50), (130, 255, 255)],      # Blue
            'grass': [(40, 40, 40), (80, 255, 255)],     # Green
            'road': [(0, 0, 0), (180, 255, 80)],         # Dark colors
            'building': [(0, 0, 100), (180, 50, 255)]     # Light colors
        }
        
        # Create semantic mask
        semantic_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        color_map = np.zeros(image.shape, dtype=np.uint8)
        
        colors = {
            'sky': [255, 0, 0],      # Red
            'grass': [0, 255, 0],    # Green
            'road': [0, 0, 255],     # Blue
            'building': [255, 255, 0] # Yellow
        }
        
        for i, (label, (lower, upper)) in enumerate(color_ranges.items()):
            lower = np.array(lower)
            upper = np.array(upper)
            
            mask = cv2.inRange(hsv, lower, upper)
            semantic_mask[mask > 0] = i + 1
            color_map[mask > 0] = colors[label]
        
        return color_map, semantic_mask
    
    def analyze_segments(self, segmented_image, original_image):
        """Analyze segmented regions"""
        # Convert to labels
        gray_segments = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
        labels = measure.label(gray_segments)
        
        # Extract region properties
        regions = measure.regionprops(labels, original_image)
        
        analysis_results = []
        result_image = original_image.copy()
        
        for region in regions:
            # Calculate properties
            area = region.area
            perimeter = region.perimeter
            centroid = region.centroid
            bbox = region.bbox
            
            # Skip small regions
            if area < 100:
                continue
            
            analysis_results.append({
                'area': area,
                'perimeter': perimeter,
                'centroid': centroid,
                'bbox': bbox,
                'aspect_ratio': (bbox[2] - bbox[0]) / (bbox[3] - bbox[1]),
                'solidity': region.solidity,
                'eccentricity': region.eccentricity
            })
            
            # Draw bounding box and centroid
            cv2.rectangle(result_image, (bbox[1], bbox[0]), (bbox[3], bbox[2]), (0, 255, 0), 2)
            cv2.circle(result_image, (int(centroid[1]), int(centroid[0])), 5, (255, 0, 0), -1)
            
            # Add text with area
            cv2.putText(result_image, f"Area: {area}", 
                       (bbox[1], bbox[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result_image, analysis_results

# Comprehensive segmentation demonstration
def demonstrate_segmentation():
    segmenter = ImageSegmenter()
    
    # Create sample image
    sample_image = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # Add some shapes
    cv2.rectangle(sample_image, (50, 50), (150, 150), (255, 0, 0), -1)    # Blue rectangle
    cv2.circle(sample_image, (250, 100), 50, (0, 255, 0), -1)             # Green circle
    cv2.ellipse(sample_image, (200, 200), (80, 40), 45, 0, 360, (0, 0, 255), -1)  # Red ellipse
    
    # Add noise
    noise = np.random.randint(0, 50, sample_image.shape, dtype=np.uint8)
    sample_image = cv2.add(sample_image, noise)
    
    print("Running segmentation demonstrations...")
    
    # 1. Watershed segmentation
    watershed_result, watershed_markers = segmenter.watershed_segmentation(sample_image)
    
    # 2. K-means segmentation
    kmeans_result, centers, labels = segmenter.kmeans_segmentation(sample_image, k=4)
    
    # 3. Region growing (from center of image)
    seed_point = (200, 150)
    region_result = segmenter.region_growing(sample_image, seed_point, threshold=30)
    
    # 4. Superpixel segmentation
    superpixel_result, segments = segmenter.superpixel_segmentation(sample_image, n_segments=50)
    
    # 5. Semantic segmentation
    semantic_result, semantic_mask = segmenter.semantic_segmentation_simple(sample_image)
    
    # 6. Analyze segments
    analyzed_result, analysis = segmenter.analyze_segments(kmeans_result, sample_image)
    
    print(f"Segmentation completed!")
    print(f"Found {len(analysis)} significant regions")
    
    for i, region in enumerate(analysis):
        print(f"Region {i+1}: Area={region['area']}, Aspect Ratio={region['aspect_ratio']:.2f}")
    
    return {
        'original': sample_image,
        'watershed': watershed_result,
        'kmeans': kmeans_result,
        'region_growing': region_result,
        'superpixels': superpixel_result,
        'semantic': semantic_result,
        'analyzed': analyzed_result
    }

# Run demonstration
# results = demonstrate_segmentation()`
        }
      ]
    },
    {
      id: 'deep-learning-cv',
      title: 'Deep Learning for Computer Vision',
      icon: '🧠',
      description: 'Implement CNNs, transfer learning, and state-of-the-art vision models',
      content: `
        Deep learning has revolutionized computer vision with Convolutional Neural Networks (CNNs).
        Learn to build and train models for image classification, object detection, and segmentation.
      `,
      keyTopics: [
        'Convolutional Neural Networks (CNNs)',
        'Transfer Learning with Pre-trained Models',
        'Image Classification with ResNet, VGG',
        'Object Detection with YOLO, R-CNN',
        'Image Segmentation with U-Net',
        'Generative Adversarial Networks (GANs)',
        'Style Transfer and Image Enhancement',
        'Model Optimization and Deployment'
      ],
      codeExamples: [
        {
          title: 'CNN Implementation with TensorFlow/Keras',
          description: 'Build and train a CNN for image classification',
          code: `import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, VGG16
import numpy as np
import matplotlib.pyplot as plt
import cv2

class CNNImageClassifier:
    def __init__(self, input_shape=(224, 224, 3), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
    
    def build_custom_cnn(self):
        """Build a custom CNN architecture"""
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fully Connected Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def build_transfer_learning_model(self, base_model_name='resnet50', trainable_layers=0):
        """Build model using transfer learning"""
        # Load pre-trained base model
        if base_model_name.lower() == 'resnet50':
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=self.input_shape)
        elif base_model_name.lower() == 'vgg16':
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape)
        else:
            raise ValueError("Supported models: 'resnet50', 'vgg16'")
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Unfreeze top layers if specified
        if trainable_layers > 0:
            base_model.trainable = True
            for layer in base_model.layers[:-trainable_layers]:
                layer.trainable = False
        
        # Add custom classification head
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.001, optimizer='adam'):
        """Compile the model"""
        if self.model is None:
            raise ValueError("Model not built yet. Call build_custom_cnn() or build_transfer_learning_model() first.")
        
        if optimizer == 'adam':
            opt = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer == 'rmsprop':
            opt = optimizers.RMSprop(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
    
    def create_data_generators(self, train_dir, val_dir, batch_size=32, augment=True):
        """Create data generators with augmentation"""
        if augment:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                zoom_range=0.2,
                shear_range=0.2,
                fill_mode='nearest'
            )
        else:
            train_datagen = ImageDataGenerator(rescale=1./255)
        
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        return train_generator, val_generator
    
    def train_model(self, train_generator, val_generator, epochs=50):
        """Train the model with callbacks"""
        # Define callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks
        )
        
        return self.history
    
    def evaluate_model(self, test_generator):
        """Evaluate model performance"""
        # Evaluate on test set
        test_loss, test_accuracy, test_top_k = self.model.evaluate(test_generator)
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Top-K Accuracy: {test_top_k:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Generate predictions
        predictions = self.model.predict(test_generator)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_generator.classes
        
        # Classification report
        from sklearn.metrics import classification_report, confusion_matrix
        print("\\nClassification Report:")
        print(classification_report(true_classes, predicted_classes, 
                                   target_names=list(test_generator.class_indices.keys())))
        
        return {
            'accuracy': test_accuracy,
            'loss': test_loss,
            'predictions': predictions,
            'predicted_classes': predicted_classes,
            'true_classes': true_classes
        }
    
    def predict_single_image(self, image_path, class_names):
        """Predict single image"""
        # Load and preprocess image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.input_shape[:2])
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)
        
        # Make prediction
        predictions = self.model.predict(image)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        predicted_class = class_names[predicted_class_idx]
        
        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence: {confidence:.4f}")
        
        # Show top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        print("\\nTop 3 Predictions:")
        for i, idx in enumerate(top_3_indices):
            print(f"{i+1}. {class_names[idx]}: {predictions[0][idx]:.4f}")
        
        return predicted_class, confidence, predictions[0]
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_feature_maps(self, image_path, layer_name):
        """Visualize feature maps from a specific layer"""
        # Create model that outputs feature maps
        layer_output = self.model.get_layer(layer_name).output
        feature_model = models.Model(inputs=self.model.input, outputs=layer_output)
        
        # Load and preprocess image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.input_shape[:2])
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)
        
        # Get feature maps
        feature_maps = feature_model.predict(image)
        
        # Plot feature maps
        n_features = min(64, feature_maps.shape[-1])  # Show up to 64 feature maps
        size = int(np.sqrt(n_features))
        
        fig, axes = plt.subplots(size, size, figsize=(20, 20))
        axes = axes.ravel()
        
        for i in range(n_features):
            axes[i].imshow(feature_maps[0, :, :, i], cmap='viridis')
            axes[i].set_title(f'Feature Map {i+1}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()

# Example usage
def demonstrate_cnn():
    # Initialize classifier
    classifier = CNNImageClassifier(input_shape=(224, 224, 3), num_classes=10)
    
    # Build model using transfer learning
    model = classifier.build_transfer_learning_model('resnet50', trainable_layers=10)
    
    # Compile model
    classifier.compile_model(learning_rate=0.001, optimizer='adam')
    
    # Print model summary
    print("Model Architecture:")
    model.summary()
    
    print("\\nCNN demonstration completed!")
    print("To train the model, provide training and validation data directories")
    
    return classifier

# Run demonstration
# classifier = demonstrate_cnn()`
        }
      ]
    }
  ]

  return (
    <div className="page">
      <div className="content">
        <div className="page-header">
          <h1>👁️ Complete Computer Vision & Image Processing</h1>
          <p className="page-description">
            Master computer vision from fundamentals to advanced deep learning. Learn OpenCV, 
            feature detection, object recognition, and modern CNN architectures.
          </p>
        </div>

        <div className="learning-path">
          <h2>🗺️ Computer Vision Learning Path</h2>
          <div className="path-steps">
            <div className="path-step">
              <div className="step-number">1</div>
              <h3>CV Fundamentals</h3>
              <p>Master OpenCV, image processing, and basic computer vision operations</p>
            </div>
            <div className="path-step">
              <div className="step-number">2</div>
              <h3>Object Detection</h3>
              <p>Implement YOLO, R-CNN, and traditional detection methods</p>
            </div>
            <div className="path-step">
              <div className="step-number">3</div>
              <h3>Image Segmentation</h3>
              <p>Learn watershed, clustering, and advanced segmentation techniques</p>
            </div>
            <div className="path-step">
              <div className="step-number">4</div>
              <h3>Deep Learning CV</h3>
              <p>Build CNNs, implement transfer learning, and deploy vision models</p>
            </div>
          </div>
        </div>

        <div className="section-tabs">
          {sections.map((section, index) => (
            <button
              key={section.id}
              className={`tab-button ${activeSection === index ? 'active' : ''}`}
              onClick={() => setActiveSection(index)}
            >
              <span className="tab-icon">{section.icon}</span>
              {section.title}
            </button>
          ))}
        </div>

        <div className="section-content">
          {sections.map((section, index) => (
            <div
              key={section.id}
              className={`section ${activeSection === index ? 'active' : ''}`}
            >
              <div className="section-header">
                <h2>
                  <span className="section-icon">{section.icon}</span>
                  {section.title}
                </h2>
                <p className="section-description">{section.description}</p>
              </div>

              <div className="section-overview">
                <p>{section.content}</p>
              </div>

              <div className="key-topics">
                <h3>🎯 Key Topics Covered</h3>
                <div className="topics-grid">
                  {section.keyTopics.map((topic, idx) => (
                    <div key={idx} className="topic-item">
                      <span className="topic-bullet">▶</span>
                      {topic}
                    </div>
                  ))}
                </div>
              </div>

              <div className="code-examples">
                <h3>💻 Code Examples & Implementation</h3>
                {section.codeExamples.map((example, idx) => (
                  <div key={idx} className="code-example">
                    <div className="example-header">
                      <h4>{example.title}</h4>
                      <p>{example.description}</p>
                      <button
                        className="toggle-code"
                        onClick={() => toggleCode(section.id, idx)}
                      >
                        {expandedCode[`${section.id}-${idx}`] ? 'Hide Code' : 'Show Code'}
                      </button>
                    </div>
                    
                    {expandedCode[`${section.id}-${idx}`] && (
                      <div className="code-block">
                        <pre><code>{example.code}</code></pre>
                      </div>
                    )}
                  </div>
                ))}
              </div>

              <div className="practice-exercises">
                <h3>🏋️ Practice Exercises</h3>
                <div className="exercises">
                  <div className="exercise">
                    <h4>Beginner Exercise</h4>
                    <p>Build a basic image classifier using OpenCV and traditional machine learning.</p>
                  </div>
                  <div className="exercise">
                    <h4>Intermediate Exercise</h4>
                    <p>Implement real-time object detection using YOLO and optimize for performance.</p>
                  </div>
                  <div className="exercise">
                    <h4>Advanced Exercise</h4>
                    <p>Create a custom CNN architecture for a specific computer vision task with transfer learning.</p>
                  </div>
                </div>
              </div>

              <div className="real-world-projects">
                <h3>🚀 Real-World Project Ideas</h3>
                <div className="projects-grid">
                  <div className="project-card">
                    <h4>Smart Security System</h4>
                    <p>Build a facial recognition system with real-time alerts and visitor logging.</p>
                  </div>
                  <div className="project-card">
                    <h4>Medical Image Analysis</h4>
                    <p>Create tools for detecting anomalies in X-rays, MRIs, or other medical imagery.</p>
                  </div>
                  <div className="project-card">
                    <h4>Autonomous Vehicle Vision</h4>
                    <p>Implement lane detection, traffic sign recognition, and obstacle avoidance systems.</p>
                  </div>
                  <div className="project-card">
                    <h4>Industrial Quality Control</h4>
                    <p>Develop automated inspection systems for manufacturing defect detection.</p>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        <div className="next-steps">
          <h2>🎯 Next Steps in Computer Vision</h2>
          <div className="next-steps-grid">
            <div className="next-step">
              <h3>🤖 Advanced AI</h3>
              <p>Explore GANs, style transfer, and generative computer vision models</p>
            </div>
            <div className="next-step">
              <h3>📱 Mobile CV</h3>
              <p>Deploy computer vision models on mobile devices and edge computing</p>
            </div>
            <div className="next-step">
              <h3>🏭 Production Systems</h3>
              <p>Scale computer vision applications for real-world deployment</p>
            </div>
            <div className="next-step">
              <h3>🔬 Research Areas</h3>
              <p>Dive into cutting-edge CV research: 3D vision, multimodal AI, and more</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ComputerVisionComplete