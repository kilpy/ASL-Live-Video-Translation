"""Webcam capture: smoke-test the camera feed. Run with `uv run python -m asl_translate.capture`."""
import cv2


def main(camera_index: int = 0) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {camera_index}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imshow("ASL capture (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
