import cv2
import numpy as np
import pandas as pd
import ast
from ultralytics import YOLO
from collections import deque

class VehicleDetectionSystem:
    def __init__(self, video_path, results_csv, scaling_factor=0.0001):
        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Initialize YOLO model for vehicle detection
        self.model = YOLO("yolov8n.pt")
        
        # Load license plate results
        self.plate_results = pd.read_csv(results_csv)
        
        # Speed detection parameters
        self.line_y = self.height // 2
        self.scaling_factor = scaling_factor
        self.speed_smoothing_window = 10  # Increased window size
        self.speed_history_window = 30  # Frames to keep speed history
        self.min_speed_threshold = 5  # Minimum speed to display (km/h)
        self.max_speed_threshold = 150  # Maximum realistic speed (km/h)
        self.speed_change_threshold = 5  # Maximum allowed speed change between frames
        
        # Tracking dictionaries
        self.previous_positions = {}
        self.previous_timestamps = {}
        self.speed_histories = {}  # Store speed history for each vehicle
        self.position_histories = {}  # Store position history for each vehicle
        self.license_plates = self._process_license_plates()
        
        # Real-world calibration (adjust these based on your video)
        self.pixels_per_meter = 30  # Approximate pixels per meter in your video
        self.frame_interval = 1.0 / self.fps  # Time between frames
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter('output_combined.mp4', fourcc, self.fps, 
                                 (self.width, self.height))

    def _process_license_plates(self):
        """Process and store license plate information"""
        license_plates = {}
        for car_id in np.unique(self.plate_results['car_id']):
            max_score = np.amax(self.plate_results[self.plate_results['car_id'] == car_id]['license_number_score'])
            plate_info = self.plate_results[
                (self.plate_results['car_id'] == car_id) & 
                (self.plate_results['license_number_score'] == max_score)
            ].iloc[0]
            
            license_plates[car_id] = {
                'license_number': plate_info['license_number'],
                'frame_number': plate_info['frame_nmr'],
                'bbox': self._parse_bbox(plate_info['license_plate_bbox']),
                'car_bbox': self._parse_bbox(plate_info['car_bbox'])
            }
            
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, plate_info['frame_nmr'])
            ret, frame = self.cap.read()
            if ret:
                x1, y1, x2, y2 = license_plates[car_id]['bbox']
                license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                license_crop = cv2.resize(license_crop, 
                                       (int((x2 - x1) * 400 / (y2 - y1)), 400))
                license_plates[car_id]['license_crop'] = license_crop
        
        return license_plates

    def _parse_bbox(self, bbox_str):
        """Parse bbox string to coordinates"""
        return list(map(int, ast.literal_eval(
            bbox_str.replace('[ ', '[')
            .replace('   ', ' ')
            .replace('  ', ' ')
            .replace(' ', ','))))

    def _initialize_tracking(self, class_id):
        """Initialize tracking for a new vehicle"""
        self.speed_histories[class_id] = deque(maxlen=self.speed_history_window)
        self.position_histories[class_id] = deque(maxlen=self.speed_history_window)

    def _calculate_speed_robust(self, class_id, curr_pos, curr_time):
        """Calculate speed with multiple improvements for stability"""
        if class_id not in self.position_histories:
            self._initialize_tracking(class_id)
            return None

        self.position_histories[class_id].append((curr_pos, curr_time))
        
        if len(self.position_histories[class_id]) < 5:  # Need minimum history
            return None

        # Calculate speed using multiple previous positions
        speeds = []
        positions = list(self.position_histories[class_id])
        
        # Use different time intervals for better accuracy
        for i in range(1, min(5, len(positions))):
            prev_pos, prev_time = positions[-i-1]
            curr_pos, curr_time = positions[-1]
            
            time_delta = curr_time - prev_time
            if time_delta <= 0:
                continue

            # Calculate displacement in pixels
            displacement = np.sqrt(
                (curr_pos[0] - prev_pos[0]) ** 2 + 
                (curr_pos[1] - prev_pos[1]) ** 2
            )
            
            # Convert to real-world units
            distance_meters = displacement / self.pixels_per_meter
            speed_ms = distance_meters / time_delta
            speed_kmh = speed_ms * 3.6
            
            if self.min_speed_threshold <= speed_kmh <= self.max_speed_threshold:
                speeds.append(speed_kmh)

        if not speeds:
            return None

        # Calculate median speed (more robust than mean)
        current_speed = np.median(speeds)

        # Apply additional filtering
        if self.speed_histories[class_id]:
            last_speed = self.speed_histories[class_id][-1]
            if abs(current_speed - last_speed) > self.speed_change_threshold:
                current_speed = last_speed + np.sign(current_speed - last_speed) * self.speed_change_threshold

        self.speed_histories[class_id].append(current_speed)
        
        # Return rolling median of recent speeds
        return np.median(list(self.speed_histories[class_id]))

    def _draw_border(self, img, top_left, bottom_right, color=(0, 255, 0), 
                    thickness=10, line_length_x=200, line_length_y=200):
        """Draw border around detected vehicle"""
        x1, y1 = top_left
        x2, y2 = bottom_right
        
        lines = [
            ((x1, y1), (x1, y1 + line_length_y)),
            ((x1, y1), (x1 + line_length_x, y1)),
            ((x1, y2), (x1, y2 - line_length_y)),
            ((x1, y2), (x1 + line_length_x, y2)),
            ((x2, y1), (x2 - line_length_x, y1)),
            ((x2, y1), (x2, y1 + line_length_y)),
            ((x2, y2), (x2, y2 - line_length_y)),
            ((x2, y2), (x2 - line_length_x, y2))
        ]
        
        for start, end in lines:
            cv2.line(img, start, end, color, thickness)

    def process_video(self):
        """Process the video and combine license plate and speed detection"""
        frame_nmr = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            current_time = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            
            # Draw reference line for speed detection
            cv2.line(frame, (0, self.line_y), (self.width, self.line_y), 
                    (0, 255, 255), 2)
            
            # Get vehicle detections
            results = self.model(frame)
            
            # Process each detected vehicle
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    
                    # Only process if we have license plate info
                    matching_plate = None
                    for car_id, plate_info in self.license_plates.items():
                        plate_box = plate_info['car_bbox']
                        if (abs(x1 - plate_box[0]) < 50 and 
                            abs(y1 - plate_box[1]) < 50):
                            matching_plate = plate_info
                            break
                    
                    if matching_plate is not None:
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        
                        # Calculate speed
                        speed = self._calculate_speed_robust(
                            class_id, 
                            (center_x, center_y), 
                            current_time
                        )
                        
                        if speed is not None:
                            # Draw vehicle box and information
                            self._draw_border(frame, (x1, y1), (x2, y2))
                            
                            # Draw license plate and speed
                            license_crop = matching_plate['license_crop']
                            H, W, _ = license_crop.shape
                            
                            try:
                                # Draw license plate image
                                frame[int(y1) - H - 100:int(y1) - 100,
                                      int((x2 + x1 - W) / 2):int((x2 + x1 + W) / 2)] = license_crop
                                
                                # Draw white background for text
                                frame[int(y1) - H - 400:int(y1) - H - 100,
                                      int((x2 + x1 - W) / 2):int((x2 + x1 + W) / 2)] = (255, 255, 255)
                                
                                # Draw license plate number
                                plate_number = matching_plate['license_number']
                                (text_width, text_height), _ = cv2.getTextSize(
                                    plate_number, cv2.FONT_HERSHEY_SIMPLEX, 4.3, 17)
                                
                                cv2.putText(frame, plate_number,
                                          (int((x2 + x1 - text_width) / 2), 
                                           int(y1 - H - 250 + (text_height / 2))),
                                          cv2.FONT_HERSHEY_SIMPLEX, 4.3, (0, 0, 0), 17)
                                
                                # Draw speed
                                speed_text = f"Speed: {speed:.1f} km/h"
                                cv2.putText(frame, speed_text,
                                          (int((x2 + x1 - text_width) / 2), 
                                           int(y1 - H - 150)),
                                          cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 8)
                            except:
                                pass
            
            # Write frame
            self.out.write(frame)
            frame_nmr += 1
            
        # Cleanup
        self.cap.release()
        self.out.release()

# Usage example
if __name__ == "__main__":
    detector = VehicleDetectionSystem(
        video_path="sample.mp4",
        results_csv="test_interpolated.csv",
        scaling_factor=0.0001
    )
    detector.process_video()