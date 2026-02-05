import cv2                      # OpenCV library for video capture and image processing
import mediapipe as mp          # MediaPipe library for face detection

def make_odd(n: int) -> int:
    # GaussianBlur kernel size must be an odd number
    return n if n % 2 == 1 else n + 1

def clamp(v, lo, hi):
    # Limits value v between lo and hi
    return max(lo, min(hi, v))

def main():
    # Open default camera (index 0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera could not be opened.")

    # Set camera resolution to Full HD (1920x1080)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Access MediaPipe face detection module
    mp_fd = mp.solutions.face_detection

    # Create face detector
    # model_selection=0 -> optimized for faces close to the camera
    detector = mp_fd.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5
    )

    # --- Configuration parameters ---
    detect_every = 5        # Run face detection every N frames (performance optimization)
    blur_strength = 55      # Blur intensity (kernel size, will be made odd)
    padding = 0.30          # Extra padding around face to avoid cutting blur on side views

    frame_idx = 0           # Frame counter
    tracker = None          # OpenCV tracker object
    have_track = True     # Indicates whether tracking is active
    last_box = None         # Last known face bounding box (x, y, w, h)

    # Function to create a tracker instance
    def create_tracker():
        # KCF tracker: fast and reasonably stable
        return cv2.TrackerKCF_create()

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        H, W = frame.shape[:2]   # Frame height and width

        # --- Face detection step ---
        # Run detection every N frames or if tracking is lost
        if frame_idx % detect_every == 0 or not have_track:
            # Convert BGR (OpenCV format) to RGB (MediaPipe requirement)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Perform face detection
            res = detector.process(rgb)

            best = None
            best_score = 0.0

            # Select the face with the highest confidence score
            if res.detections:
                for d in res.detections:
                    score = float(d.score[0])
                    if score > best_score:
                        best_score = score
                        best = d

            if best is not None:
                # Get relative bounding box (values between 0 and 1)
                bb = best.location_data.relative_bounding_box

                # Convert relative coordinates to pixel coordinates
                x = int(bb.xmin * W)
                y = int(bb.ymin * H)
                w = int(bb.width * W)
                h = int(bb.height * H)

                # Add padding to the bounding box
                px = int(w * padding)
                py = int(h * padding)

                # Ensure bounding box stays within frame boundaries
                x = clamp(x - px, 0, W - 1)
                y = clamp(y - py, 0, H - 1)
                w = clamp(w + 2 * px, 1, W - x)
                h = clamp(h + 2 * py, 1, H - y)

                # Save bounding box
                last_box = (x, y, w, h)

                # Initialize tracker with detected face
                tracker = create_tracker()
                tracker.init(frame, last_box)
                have_track = True

        # --- Tracking step ---
        # Update face position using tracker for every frame
        if have_track and tracker is not None:
            ok, box = tracker.update(frame)
            if ok:
                # Update bounding box with tracked position
                x, y, w, h = map(int, box)
                x = clamp(x, 0, W - 1)
                y = clamp(y, 0, H - 1)
                w = clamp(w, 1, W - x)
                h = clamp(h, 1, H - y)
                last_box = (x, y, w, h)
            else:
                # Tracking failed, detection will be used again
                have_track = False

        # --- Blur application ---
        if last_box is not None:
            x, y, w, h = last_box
            roi = frame[y:y + h, x:x + w]  # Region of interest (face area)

            if roi.size > 0:
                k = make_odd(blur_strength)

                # Limit kernel size for performance and visual correctness
                max_k = make_odd(
                    min(99, max(9, min(roi.shape[0], roi.shape[1]) // 2))
                )
                kk = min(k, max_k)

                # Apply Gaussian blur to face region
                frame[y:y + h, x:x + w] = cv2.GaussianBlur(
                    roi, (kk, kk), 0
                )

        # Display information on screen
        cv2.putText(
            frame,
            f"detect_every={detect_every} blur={blur_strength} pad={padding}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # Show the processed frame
        cv2.imshow("Fast Stable Face Blur (q quit, +/- blur, [/] detect rate)", frame)

        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key in (ord('+'), ord('=')):
            blur_strength = min(199, blur_strength + 10)
        if key in (ord('-'), ord('_')):
            blur_strength = max(3, blur_strength - 10)
        if key == ord('*'):
            detect_every = min(15, detect_every + 1)
        if key == ord('/'):
            detect_every = max(1, detect_every - 1)
        if key == ord('p'):
            padding = min(0.60, padding + 0.05)
        if key == ord('o'):
            padding = max(0.00, padding - 0.05)

    # Release camera and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
