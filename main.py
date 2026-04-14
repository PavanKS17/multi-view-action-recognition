import argparse
import json
import cv2
import os
from src.hybrid_pipeline import UltimateHybridTracker

def process_video(topview_path):
    cap = cv2.VideoCapture(topview_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening {topview_path}")

    # Initialize your new YOLO/Homography pipeline
    tracker = UltimateHybridTracker(model_path="models/plate_keypoints.pt")
    
    predictions = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run the frame through the pipeline
        pred, processed_frame = tracker.predict_well(frame)
        
        if pred:
            predictions.append(pred)
            
        # Optional: Show the video live as it processes (Great for debugging!)
        cv2.imshow("Inference", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # In a real scenario, you'd filter this list to only grab the prediction 
    # at the exact moment of dispense (Z-axis drop). For now, we grab the most common one.
    if predictions:
        # Simplistic logic: return the most frequent prediction in the clip
        final_prediction = max(predictions, key=lambda x: predictions.count(x))
        return [final_prediction]
    return []

def main():
    parser = argparse.ArgumentParser(description="YOLO Hybrid Inference")
    parser.add_argument("--fpv", required=True, help="Path to FPV video")
    parser.add_argument("--topview", required=True, help="Path to Topview video")
    args = parser.parse_args()

    clip_id_fpv = os.path.splitext(os.path.basename(args.fpv))[0]
    clip_id_top = os.path.splitext(os.path.basename(args.topview))[0]

    # Note: We are ignoring the FPV video for this simplified spatial tracking test
    predicted_wells = process_video(args.topview)

    output = {
        "clip_id_FPV": clip_id_fpv,
        "clip_id_Topview": clip_id_top,
        "wells_prediction": predicted_wells
    }

    print(json.dumps(output, indent=4))

if __name__ == "__main__":
    main()