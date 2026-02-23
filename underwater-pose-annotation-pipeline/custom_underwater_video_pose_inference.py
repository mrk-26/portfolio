import cv2
import json
import numpy as np
import sys
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmengine.registry import init_default_scope

from mmdet.apis import inference_detector, init_detector
from mmpose.apis import inference_topdown
from mmpose.models import build_pose_estimator
from mmpose.visualization import PoseLocalVisualizer

from collections import defaultdict

# -------- CONFIGURATION --------
# video_path = '../mmpose/swimmer.avi'   # 'test_video.avi'
video_path = sys.argv[1] if len(sys.argv) > 1 else 'swimmer.avi'
output_video_path = '../mmpose/output/test_pose_output.avi'
output_json_path = '../mmpose/output/test_pose_output.json'

det_config = '../mmdetection/work_dirs/rtmdet_tiny_1class_underwater/rtmdet_tiny_1class_underwater.py'
det_checkpoint = '../mmdetection/work_dirs/rtmdet_tiny_1class_underwater/best_coco_bbox_mAP_epoch_17.pth'

pose_config = '../mmpose/configs/underwater/rtmpose-l_underwater.py'
pose_checkpoint = '../mmpose/work_dirs/rtmpose-l_underwater/best_coco_AP_epoch_30.pth'

device = 'cuda:0'

keypoint_thresholds = [0.05, 0.1, 0.15, 0.2, 0.3]

# -------- INITIALIZE DETECTOR --------
init_default_scope('mmdet')
detector = init_detector(det_config, det_checkpoint, device=device)

# -------- INITIALIZE POSE MODEL --------
init_default_scope('mmpose')
pose_cfg = Config.fromfile(pose_config)
pose_model = build_pose_estimator(pose_cfg.model)
load_checkpoint(pose_model, pose_checkpoint, map_location='cpu')
pose_model.cfg = pose_cfg
pose_model.eval()

# -------- GET & SET METADATA --------
dataset_meta = pose_cfg.dataset_info
# Convert skeleton_info to skeleton format
if 'skeleton_info' in dataset_meta:
    print("Converting skeleton_info to skeleton format...")
    # Create mapping from keypoint name to index
    keypoint_name_to_id = {v['name']: int(k) for k, v in dataset_meta['keypoint_info'].items()}
    
    dataset_meta['skeleton'] = [
    [keypoint_name_to_id[conn['link'][0]], keypoint_name_to_id[conn['link'][1]]]
    for conn in dataset_meta['skeleton_info'].values()
]

# Ensure required visualization parameters
dataset_meta['pose_kpt_color'] = [v['color'] for v in dataset_meta['keypoint_info'].values()]
dataset_meta['pose_link_color'] = [conn['color'] for conn in dataset_meta['skeleton_info'].values()]
dataset_meta['num_keypoints'] = len(dataset_meta['keypoint_info'])

print("✅ Final skeleton connections:", dataset_meta['skeleton'])
print("✅ Keypoint colors:", dataset_meta['pose_kpt_color'])
print("✅ Link colors:", dataset_meta['pose_link_color'])


# -------- INITIALIZE VISUALIZER --------
visualizer = PoseLocalVisualizer(line_width=3, radius=4)
# visualizer.set_dataset_meta(dataset_meta)
visualizer.set_dataset_meta(dataset_meta)
pose_model.dataset_meta = dataset_meta

# -------- SETUP VIDEO --------
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

frame_id = 0
results_json = []

# -------- TEMPORAL SMOOTHING SETUP --------
smoothed_kpts = defaultdict(lambda: None)
alpha = 0.4  # EMA factor

def expand_bbox(bbox, image_shape, scale=0.2):
    """
    Expand a bounding box by a given scale factor while staying within image bounds.

    Parameters:
        bbox (list or array): [x1, y1, x2, y2]
        image_shape (tuple): shape of the image (height, width, channels)
        scale (float): scale factor (e.g., 1.2 for 20% expansion)

    Returns:
        list: [new_x1, new_y1, new_x2, new_y2]
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2

    new_w = w * scale
    new_h = h * scale

    new_x1 = max(0, int(cx - new_w / 2))
    new_y1 = max(0, int(cy - new_h / 2))
    new_x2 = min(image_shape[1] - 1, int(cx + new_w / 2))
    new_y2 = min(image_shape[0] - 1, int(cy + new_h / 2))

    return [new_x1, new_y1, new_x2, new_y2]




while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_id += 1
    # print(f"Processing frame {frame_id}...")

    # ---- Person Detection ----
    init_default_scope('mmdet')
    det_result = inference_detector(detector, frame)
    bboxes = det_result.pred_instances.bboxes.cpu().numpy()
    scores = det_result.pred_instances.scores.cpu().numpy()
    labels = det_result.pred_instances.labels.cpu().numpy()

    # Only keep 'person' class (label == 0) and high confidence
    person_bboxes = [
        bbox for bbox, score, label in zip(bboxes, scores, labels)
        if label == 0 and score > 0.5
    ]

    if not person_bboxes:
        out_video.write(frame)
        continue

    expanded_bboxes = []
    for bbox in person_bboxes:
        expanded = expand_bbox(bbox, frame.shape, scale=1.2)
        expanded_bboxes.append(expanded)

    bboxes_np = np.array(expanded_bboxes, dtype=np.float32)

    # ---- Pose Estimation ----
    init_default_scope('mmpose')
    pose_results = inference_topdown(pose_model, frame, bboxes_np)

    # -------- APPLY TEMPORAL SMOOTHING --------
    smoothed_results = []
    for i, pose_result in enumerate(pose_results):
        kpts = pose_result.pred_instances.keypoints
        if smoothed_kpts[i] is None:
            smoothed_kpts[i] = kpts
        else:
            smoothed_kpts[i] = alpha * kpts + (1 - alpha) * smoothed_kpts[i]
        pose_result.pred_instances.keypoints = smoothed_kpts[i]
        smoothed_results.append(pose_result)

    # ---- Visualization ----
    vis_frame = frame.copy()
    try:
        visualizer.set_image(vis_frame)
        for pose_result in smoothed_results:
            if frame_id ==300:
                kpts = pose_result.pred_instances.keypoints
                kpt_scores = pose_result.pred_instances.keypoint_scores
                print(f"Keypoint scores for frame {frame_id}: {kpt_scores}")
                print(f"Min score: {np.min(kpt_scores)}, Max score: {np.max(kpt_scores)}")
                print("Keypoints:", pose_result.pred_instances.keypoints.shape)
                print("Pose skeleton length:", len(dataset_meta['skeleton']))

            visualizer.add_datasample(
                name='result',
                image=vis_frame,
                data_sample=pose_result,
                draw_gt=False,
                draw_pred=True,
                kpt_thr=0.25,
                show=False
            )
        vis_frame = visualizer.get_image()
    except Exception as e:
        print(f"Error during visualization: {e}")
        
    for bbox in expanded_bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    out_video.write(vis_frame)


   #  print(f"Video saved for kpt_thr {kpt_thr} to: {out_path}")
    # ---- Save Keypoints to JSON ----
    instances = []
    for pose_result, bbox in zip(pose_results, person_bboxes):
        keypoints = pose_result.pred_instances.keypoints.tolist()
        instances.append({
            'keypoints': keypoints,
            'bbox': bbox.tolist()
        })
    results_json.append({
        'frame_id': frame_id,
        'instances': instances
    })
    

cap.release()
out_video.release()


# -------- SAVE JSON --------
with open(output_json_path, 'w') as f:
    json.dump(results_json, f, indent=2)

print(f"\n✅ Done. Video saved to: {output_video_path}")
print(f"✅ Keypoints saved to: {output_json_path}")
