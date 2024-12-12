import cv2
import numpy as np
from rknn.api import RKNN
import time
import os
import argparse
import itertools
from collections import defaultdict
from typing import Dict, List
import torch
from scipy.optimize import linear_sum_assignment

class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3

def linear_assignment(tracks, detections, threshold):
    if len(tracks) == 0 or len(detections) == 0:
        return [], range(len(tracks)), range(len(detections))

    cost_matrix = np.zeros((len(tracks), len(detections)))
    for i, track in enumerate(tracks):
        cost_matrix[i] = 1 - iou(track.tlbr, detections)

    matches_idx = []
    unmatched_tracks = []
    unmatched_detections = []

    track_indices, detection_indices = linear_sum_assignment(cost_matrix)

    for track_idx, detection_idx in zip(track_indices, detection_indices):
        if cost_matrix[track_idx, detection_idx] > threshold:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches_idx.append([track_idx, detection_idx])

    return matches_idx, unmatched_tracks, unmatched_detections

def iou(box1, box2):
    """计算IOU"""
    box1 = np.asarray(box1).reshape(-1, 4)
    box2 = np.asarray(box2).reshape(-1, 4)
    
    x1 = np.maximum(box1[:, 0].reshape(-1, 1), box2[:, 0])
    y1 = np.maximum(box1[:, 1].reshape(-1, 1), box2[:, 1])
    x2 = np.minimum(box1[:, 2].reshape(-1, 1), box2[:, 2])
    y2 = np.minimum(box1[:, 3].reshape(-1, 1), box2[:, 3])
    
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    union = area1.reshape(-1, 1) + area2 - intersection
    
    return intersection / np.maximum(union, np.finfo(float).eps)

def join_tracks(tracked_tracks, lost_tracks):
    """合并跟踪列表"""
    return tracked_tracks + lost_tracks

class STrack(object):
    def __init__(self, tlbr, cls, score=None):
        """初始化跟踪对象"""
        self._tlbr = np.asarray(tlbr, dtype=np.float32).reshape(4)  # 确保是4维数组
        self.cls = cls
        self.score = score
        self.track_id = None
        self.is_activated = False
        self.tracklet_len = 0
        self.state = TrackState.New
        self.start_frame = 0
        self.end_frame = 0
        
    @property
    def tlbr(self):
        """获取边界框坐标"""
        return self._tlbr.copy()
        
    def activate(self, frame_id, new_id=True):
        if new_id:
            self.track_id = self.next_id()
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.start_frame = frame_id
        
    def re_activate(self, new_track, cls, frame_id, new_id=False):
        self._tlbr = np.asarray(new_track, dtype=np.float32).reshape(4)
        self.cls = cls
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.start_frame = frame_id
        if new_id:
            self.track_id = self.next_id()
            
    def update(self, new_track, cls, frame_id):
        self._tlbr = np.asarray(new_track, dtype=np.float32).reshape(4)
        self.cls = cls
        self.tracklet_len += 1
        self.state = TrackState.Tracked
        self.is_activated = True
        self.end_frame = frame_id
        
    def mark_lost(self):
        self.state = TrackState.Lost
        
    def mark_removed(self):
        self.state = TrackState.Removed
    
    next_id = itertools.count().__next__

class BYTETracker(object):
    def __init__(self, track_thresh=0.25, track_buffer=30, match_thresh=0.8):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.frame_id = 0
        self.tracks = []
        self.lost_tracks = []

    def update(self, boxes, scores, classes):
        self.frame_id += 1
        activated_tracks = []
        refind_tracks = []
        lost_tracks = []
        removed_tracks = []

        if boxes is not None and len(boxes) > 0:
            boxes = boxes.reshape(-1, 4)
        else:
            return []

        # 根据分数进行筛选
        remain_inds = scores > self.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        
        dets_second = boxes[inds_second]
        dets = boxes[remain_inds]
        
        classes_1 = classes[remain_inds]
        classes_2 = classes[inds_second]
        scores_1 = scores[remain_inds]
        scores_2 = scores[inds_second]

        # Initialize lists
        unconfirmed = []
        tracked_tracks = []
        for track in self.tracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_tracks.append(track)

        # 第一阶段关联：高分检测框
        track_pool = join_tracks(tracked_tracks, self.lost_tracks)
        
        if len(dets) > 0:
            matches, u_track, u_detection = linear_assignment(
                track_pool, dets, self.match_thresh)

            for itracked, idet in matches:
                track = track_pool[itracked]
                det = dets[idet]
                cls = classes_1[idet]
                score = scores_1[idet]
                
                if track.state == TrackState.Tracked:
                    track.update(det, cls, self.frame_id)
                    activated_tracks.append(track)
                else:
                    track.re_activate(det, cls, self.frame_id, new_id=False)
                    refind_tracks.append(track)

            for it in u_track:
                track = track_pool[it]
                if not track.state == TrackState.Lost:
                    track.mark_lost()
                    lost_tracks.append(track)

            for idet in u_detection:
                det = dets[idet]
                cls = classes_1[idet]
                score = scores_1[idet]
                track = STrack(det, cls, score)
                track.activate(self.frame_id)
                activated_tracks.append(track)

        # 第二阶段关联：低分检测框
        if len(dets_second) > 0:
            matches, u_track, u_detection = linear_assignment(
                unconfirmed + lost_tracks, dets_second, self.match_thresh)

            for itracked, idet in matches:
                track = unconfirmed[itracked] if itracked < len(unconfirmed) else lost_tracks[itracked - len(unconfirmed)]
                det = dets_second[idet]
                cls = classes_2[idet]
                score = scores_2[idet]
                
                if track.state == TrackState.Tracked:
                    track.update(det, cls, self.frame_id)
                    activated_tracks.append(track)
                else:
                    track.re_activate(det, cls, self.frame_id, new_id=False)
                    refind_tracks.append(track)

        # 更新所有轨迹状态
        for track in self.tracks:
            if track not in activated_tracks:
                track.mark_removed()
                removed_tracks.append(track)

        # 更新lost tracks
        self.lost_tracks.extend(lost_tracks)
        self.lost_tracks = [t for t in self.lost_tracks if t.end_frame >= self.frame_id - self.track_buffer]
                
        # 更新跟踪器状态
        self.tracks = [t for t in self.tracks if t not in removed_tracks]
        self.tracks.extend(activated_tracks)
        self.tracks.extend(refind_tracks)

        # 输出活跃的轨迹
        output_tracks = []
        for track in self.tracks:
            if track.is_activated:
                output_tracks.append(track)
                
        return output_tracks

class YOLO_RKNN(object):
    def __init__(self, model_path, target='rk3588', enable_tracker=False):
        self.CLASSES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
                       "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                       "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                       "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                       "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                       "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                       "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", 
                       "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                       "teddy bear", "hair drier", "toothbrush"]
        
        self.OBJ_THRESH = 0.25
        self.NMS_THRESH = 0.45
        self.IMG_SIZE = (640, 640)
        
        self.rknn = RKNN(verbose=True)
        print(f'--> Loading RKNN model: {model_path}')
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            print('Load RKNN model failed!')
            exit(ret)
        print('Load RKNN model done')

        print('--> Init runtime environment')
        ret = self.rknn.init_runtime(target=target)
        if ret != 0:
            print('Init runtime environment failed!')
            exit(ret)
        print('Init runtime environment done')

        self.enable_tracker = enable_tracker
        if enable_tracker:
            self.tracker = BYTETracker()

    def letter_box(self, img, new_shape=(640, 640), pad_color=(0, 0, 0)):
        shape = img.shape[:2]  # 当前形状 [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # 缩放比例
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # 计算填充
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        dw, dh = dw / 2, dh / 2  # 分配到两侧

        # 调整大小
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        # 添加边框
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)
        
        return img, r, (dw, dh)

    def dfl(self, position):
        x = torch.tensor(position)
        n, c, h, w = x.shape
        p_num = 4
        mc = c//p_num
        y = x.reshape(n, p_num, mc, h, w)
        y = y.softmax(2)
        acc_metrix = torch.tensor(range(mc)).float().reshape(1,1,mc,1,1)
        y = (y*acc_metrix).sum(2)
        return y.numpy()

    def box_process(self, position):
        grid_h, grid_w = position.shape[2:4]
        col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
        col = col.reshape(1, 1, grid_h, grid_w)
        row = row.reshape(1, 1, grid_h, grid_w)
        grid = np.concatenate((col, row), axis=1)
        stride = np.array([self.IMG_SIZE[1]//grid_h, self.IMG_SIZE[0]//grid_w]).reshape(1,2,1,1)

        position = self.dfl(position)
        box_xy = grid + 0.5 - position[:,0:2,:,:]
        box_xy2 = grid + 0.5 + position[:,2:4,:,:]
        xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)

        return xyxy

    def post_process(self, input_data):
        boxes, scores, classes_conf = [], [], []
        default_branch = 3
        pair_per_branch = len(input_data)//default_branch

        for i in range(default_branch):
            boxes.append(self.box_process(input_data[pair_per_branch*i]))
            classes_conf.append(input_data[pair_per_branch*i+1])
            scores.append(np.ones_like(input_data[pair_per_branch*i+1][:,:1,:,:], dtype=np.float32))

        def sp_flatten(_in):
            ch = _in.shape[1]
            _in = _in.transpose(0,2,3,1)
            return _in.reshape(-1, ch)

        boxes = [sp_flatten(_v) for _v in boxes]
        classes_conf = [sp_flatten(_v) for _v in classes_conf]
        scores = [sp_flatten(_v) for _v in scores]

        boxes = np.concatenate(boxes)
        classes_conf = np.concatenate(classes_conf)
        scores = np.concatenate(scores)

        # 过滤
        box_confidences = scores.reshape(-1)
        candidate, class_num = classes_conf.shape

        class_max_score = np.max(classes_conf, axis=-1)
        classes = np.argmax(classes_conf, axis=-1)

        _class_pos = np.where(class_max_score * box_confidences >= self.OBJ_THRESH)
        scores = (class_max_score * box_confidences)[_class_pos]
        boxes = boxes[_class_pos]
        classes = classes[_class_pos]

        # NMS
        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]

            x = b[:, 0]
            y = b[:, 1]
            w = b[:, 2] - b[:, 0]
            h = b[:, 3] - b[:, 1]
            areas = w * h
            order = s.argsort()[::-1]
            keep = []

            while order.size > 0:
                i = order[0]
                keep.append(i)

                xx1 = np.maximum(x[i], x[order[1:]])
                yy1 = np.maximum(y[i], y[order[1:]])
                xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
                yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

                w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
                h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
                inter = w1 * h1

                ovr = inter / (areas[i] + areas[order[1:]] - inter)
                inds = np.where(ovr <= self.NMS_THRESH)[0]
                order = order[inds + 1]

            if len(keep) > 0:
                nboxes.append(b[keep])
                nclasses.append(c[keep])
                nscores.append(s[keep])

        if not nboxes:
            return None, None, None

        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)

        return boxes, classes, scores

    def draw(self, image, boxes, scores, classes, track_ids=None):
        """增强的绘制函数，支持跟踪ID显示"""
        for i, (box, score, cl) in enumerate(zip(boxes, scores, classes)):
            top, left, right, bottom = [int(_b) for _b in box]
            
            # 添加跟踪ID显示
            if track_ids is not None:
                label = f'{self.CLASSES[cl]} #{track_ids[i]}'
            else:
                label = f'{self.CLASSES[cl]}'
                
            print(f"{label} @ (%.3f %.3f %.3f %.3f) %.3f" % (top, left, right, bottom, score))
            cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
            cv2.putText(image, f'{label} {score:.2f}',
                        (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return image

    def inference(self, frame):
        """增强的推理函数，支持目标跟踪"""
        # 执行常规推理
        img, ratio, (dw, dh) = self.letter_box(frame, self.IMG_SIZE, pad_color=(0,0,0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        outputs = self.rknn.inference(inputs=[img])
        boxes, classes, scores = self.post_process(outputs)
        
        if boxes is None:
            return None, None, None, ratio, (dw, dh), None
            
        # 如果启用了跟踪器，执行跟踪
        track_ids = None
        if self.enable_tracker and boxes is not None:
            tracks = self.tracker.update(boxes, scores, classes)
            if tracks:
                boxes = np.array([track._tlbr for track in tracks])
                scores = np.array([track.score for track in tracks])
                classes = np.array([track.cls for track in tracks])
                track_ids = np.array([track.track_id for track in tracks])
        
        return boxes, classes, scores, ratio, (dw, dh), track_ids

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='../model/yolov8.rknn', help='model path')
    parser.add_argument('--target', type=str, default='rk3588', help='target platform')
    parser.add_argument('--camera_id', type=int, default=21, help='camera device id')
    parser.add_argument('--enable_tracker', action='store_true', help='enable ByteTrack object tracking')
    args = parser.parse_args()

    # 初始化检测器
    detector = YOLO_RKNN(args.model_path, args.target, args.enable_tracker)
    
    # 打开摄像头
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    
    print("Press 'q' to quit")
    
    fps = 0
    frame_count = 0
    prev_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        # 执行检测和跟踪
        boxes, classes, scores, ratio, (dw, dh), track_ids = detector.inference(frame)
        
        # 绘制结果
        if boxes is not None:
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / ratio
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / ratio
            frame = detector.draw(frame, boxes, scores, classes, track_ids)
        
        # 显示FPS和跟踪状态
        status = "Tracking ON" if args.enable_tracker else "Tracking OFF"
        cv2.putText(frame, f'FPS: {fps:.1f} | {status}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('YOLOv8 Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    detector.rknn.release()

if __name__ == "__main__":
    main()