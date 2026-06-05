import cv2
import numpy as np
import tensorflow as tf

# ==========================================
# 1. INITIALIZE THE TFLITE MODEL
# ==========================================
MODEL_PATH = "face_detection_front.tflite"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Extract expected input shape (128x128 for BlazeFace Front)
_, img_height, img_width, _ = input_details[0]['shape']

# ==========================================
# 2. NUMERICALLY STABLE POST-PROCESSING
# ==========================================
def stable_sigmoid(x):
    """Prevents overflow warnings by splitting positive and negative values."""
    return np.where(x >= 0, 
                    1.0 / (1.0 + np.exp(-x)), 
                    np.exp(x) / (1.0 + np.exp(x)))

def decode_boxes(raw_boxes, raw_scores, score_threshold=0.70):
    """
    Decodes raw model outputs. 
    Applies stable sigmoid and clips values to prevent mathematical overflows.
    """
    boxes = np.squeeze(raw_boxes)   # Shape: (896, 16) or similar
    scores = np.squeeze(raw_scores) # Shape: (896, 1) or (896,)
    
    # Clip values to prevent extreme data anomalies from crashing exp()
    scores = np.clip(scores, -20, 20)
    probabilities = stable_sigmoid(scores)
    
    # Handle both 1D and 2D score arrays
    if len(probabilities.shape) > 1:
        probabilities = probabilities[:, 0]

    candidate_indices = np.where(probabilities > score_threshold)[0]
    
    detections = []
    for idx in candidate_indices:
        # BlazeFace coordinates are typically ordered as [ymin, xmin, ymax, xmax]
        ymin, xmin, ymax, xmax = boxes[idx][:4]
        
        # Clamp coordinates between 0.0 and 1.0 boundary limits
        xmin = max(0.0, min(float(xmin), 1.0))
        ymin = max(0.0, min(float(ymin), 1.0))
        xmax = max(0.0, min(float(xmax), 1.0))
        ymax = max(0.0, min(float(ymax), 1.0))
        
        # Fix inverse/flipped box dimensions if the raw output is mirrored
        if xmin > xmax: xmin, xmax = xmax, xmin
        if ymin > ymax: ymin, ymax = ymax, ymin

        detections.append({
            'box': [xmin, ymin, xmax, ymax],
            'score': float(probabilities[idx])
        })
        
    return detections

# ==========================================
# 3. LIVE WEBCAM VIDEO CAPTURE LOOP
# ==========================================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam hardware.")
    exit()

print("\n" + "="*50)
print("Webcam successfully started!")
print("-> Show your face to the camera loop.")
print("-> Click on the video window and press 'q' to quit safely.")
print("="*50 + "\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab camera frame.")
        break
        
    # Flip frame horizontally for a natural 'mirror' selfie view
    frame = cv2.flip(frame, 1)
    orig_h, orig_w, _ = frame.shape

    # Pre-process frame: Convert to RGB -> Resize -> Normalize to [-1.0, 1.0]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_frame, (img_width, img_height))
    normalized = (resized.astype(np.float32) - 127.5) / 127.5
    input_tensor = np.expand_dims(normalized, axis=0)

    # Bind tensor data and run hardware inference
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.interpreter.invoke() if hasattr(interpreter, 'interpreter') else interpreter.invoke()

    # Extract response layer arrays
    raw_regressors = interpreter.get_tensor(output_details[0]['index'])
    raw_classifiers = interpreter.get_tensor(output_details[1]['index'])
    
    # Process outputs
    detected_faces = decode_boxes(raw_regressors, raw_classifiers)

    # Render bounding boxes over the original feed image frame
    for face in detected_faces:
        xmin, ymin, xmax, ymax = face['box']
        
        # Scale coordinates back up to native monitor size
        start_point = (int(xmin * orig_w), int(ymin * orig_h))
        end_point = (int(xmax * orig_w), int(ymax * orig_h))
        
        # Draw target details
        cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)
        cv2.putText(frame, f"Face: {face['score']:.2f}", (start_point[0], max(15, start_point[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show video output window on screen
    cv2.imshow("Real-Time Face Tracking (BlazeFace)", frame)

    # Check for 'q' key to quit cleanly
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Webcam pipeline shut down cleanly.")