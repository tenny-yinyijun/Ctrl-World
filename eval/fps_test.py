import cv2

path = "/n/fs/worldmodeliw/ctrlworld/dataset_example/debug/videos/train/12/0.mp4"

# path = "/n/fs/worldmodeliw/ctrlworld/evaluation_inf_results/debug/base_model/12/view0.mp4"

cap = cv2.VideoCapture(path)

if not cap.isOpened():
    raise RuntimeError("Failed to open video")

num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Number of frames:", num_frames)

fps = cap.get(cv2.CAP_PROP_FPS)
print("FPS:", fps)

cap.release()