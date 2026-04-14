import cv2
import numpy as np
from ultralytics import YOLO

class UltimateHybridTracker:
    def __init__(self, model_path="models/plate_keypoints.pt"):
        # Load your custom trained YOLOv8-Pose model
        self.model = YOLO(model_path)
        
        # Mathematical Grid Setup (600x400 mapping)
        self.max_width = 600
        self.max_height = 400
        self.cell_w = self.max_width // 12
        self.cell_h = self.max_height // 8

    def order_points(self, pts):
        """Sorts the 4 YOLO plate corners into Top-Left, Top-Right, Bottom-Right, Bottom-Left"""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def predict_well(self, frame):
        """Runs the hybrid perception and mapping pipeline on a single frame."""
        
        # --- 1. DEEP LEARNING (Perception) ---
        # Run YOLO inference (verbose=False keeps your terminal clean)
        results = self.model(frame, verbose=False)

        # Safety check: Did YOLO find anything?
        if not results or not results[0].keypoints or not results[0].keypoints.has_visible:
            return None, frame 

        # Extract the X/Y coordinates of the keypoints.
        # Assuming your annotation ordered them: 0-3 are plate corners, 4 is pipette tip
        keypoints = results[0].keypoints.xy[0].cpu().numpy()

        if len(keypoints) < 5:
            return None, frame # Didn't find all 5 required points

        plate_corners = keypoints[0:4]
        pipette_tip = keypoints[4]

        # --- 2. CLASSICAL CV (Geometry & Math) ---
        # Order the corners for the perspective transform
        src_pts = self.order_points(plate_corners)
        
        dst_pts = np.array([
            [0, 0],
            [self.max_width - 1, 0],
            [self.max_width - 1, self.max_height - 1],
            [0, self.max_height - 1]
        ], dtype="float32")

        # Calculate the Homography Matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # Format the tip for matrix multiplication
        pts = np.array([[[float(pipette_tip[0]), float(pipette_tip[1])]]])
        
        # Multiply the physical camera pixel by 'M'
        warped_pt = cv2.perspectiveTransform(pts, M)[0][0]
        
        col_idx = int(warped_pt[0] // self.cell_w) + 1
        row_idx = int(warped_pt[1] // self.cell_h)
        row_letter = chr(65 + row_idx)

        # Validate the prediction
        if 1 <= col_idx <= 12 and 0 <= row_idx <= 7:
            prediction = {"well_row": row_letter, "well_column": str(col_idx)}
        else:
            prediction = None # Hovering outside the grid

        # --- 3. Draw Visuals for the output video ---
        # Draw the plate boundary in green
        cv2.polylines(frame, [np.int32(src_pts)], isClosed=True, color=(0, 255, 0), thickness=3)
        # Draw the pipette tip in red
        cv2.circle(frame, (int(pipette_tip[0]), int(pipette_tip[1])), 8, (0, 0, 255), -1)
        
        if prediction:
            text = f"Target: {prediction['well_row']}{prediction['well_column']}"
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)

        return prediction, frame