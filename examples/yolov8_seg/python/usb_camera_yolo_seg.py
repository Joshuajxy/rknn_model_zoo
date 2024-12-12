import cv2
import numpy as np
from rknn.api import RKNN
import time
import os
import argparse
import torch
import torch.nn.functional as F

class Colors:
    def __init__(self):
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

class YOLO_SEG_RKNN(object):
    def __init__(self, model_path, target='rk3588'):
        self.CLASSES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
                       "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                       "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                       "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                       "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                       "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                       "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", 
                       "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                       "teddy bear", "hair drier", "toothbrush"]
        
        # Initialize parameters
        self.OBJ_THRESH = 0.25
        self.NMS_THRESH = 0.45
        self.MAX_DETECT = 300
        self.IMG_SIZE = (640, 640)
        
        # Initialize RKNN
        self.rknn = RKNN(verbose=True)
        print(f'--> Loading RKNN model: {model_path}')
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            print('Load RKNN model failed!')
            exit(ret)
        print('Load RKNN model done')

        # Initialize runtime environment
        print('--> Init runtime environment')
        ret = self.rknn.init_runtime(target=target)
        if ret != 0:
            print('Init runtime environment failed!')
            exit(ret)
        print('Init runtime environment done')

        self.color = Colors()

    def letter_box(self, img, new_shape=(640, 640), pad_color=(114, 114, 114)):
        """Resize and pad image while meeting stride-multiple constraints."""
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        dw, dh = dw / 2, dh / 2  # divide padding into 2 sides

        # Resize
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        # Add border
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)
        
        return img, r, (dw, dh)

    def dfl(self, position):
        """Distribution Focal Loss."""
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
        """Process bounding boxes."""
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

    def _crop_mask(self, masks, boxes):
        """Crop predicted masks by zeroing out everything not in the predicted bbox."""
        n, h, w = masks.shape
        x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
        r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
        c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)
        
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    def filter_boxes(self, boxes, box_confidences, box_class_probs, seg_part):
        """Filter boxes with object threshold."""
        box_confidences = box_confidences.reshape(-1)
        candidate, class_num = box_class_probs.shape

        class_max_score = np.max(box_class_probs, axis=-1)
        classes = np.argmax(box_class_probs, axis=-1)

        _class_pos = np.where(class_max_score * box_confidences >= self.OBJ_THRESH)
        scores = (class_max_score * box_confidences)[_class_pos]

        boxes = boxes[_class_pos]
        classes = classes[_class_pos]
        seg_part = (seg_part * box_confidences.reshape(-1, 1))[_class_pos]

        return boxes, classes, scores, seg_part

    def post_process(self, input_data):
        """Post process for segmentation."""
        proto = input_data[-1]
        boxes, scores, classes_conf, seg_part = [], [], [], []
        default_branch = 3
        pair_per_branch = len(input_data)//default_branch

        for i in range(default_branch):
            boxes.append(self.box_process(input_data[pair_per_branch*i]))
            classes_conf.append(input_data[pair_per_branch*i+1])
            scores.append(np.ones_like(input_data[pair_per_branch*i+1][:,:1,:,:], dtype=np.float32))
            seg_part.append(input_data[pair_per_branch*i+3])

        def sp_flatten(_in):
            ch = _in.shape[1]
            _in = _in.transpose(0,2,3,1)
            return _in.reshape(-1, ch)

        boxes = [sp_flatten(_v) for _v in boxes]
        classes_conf = [sp_flatten(_v) for _v in classes_conf]
        scores = [sp_flatten(_v) for _v in scores]
        seg_part = [sp_flatten(_v) for _v in seg_part]

        boxes = np.concatenate(boxes)
        classes_conf = np.concatenate(classes_conf)
        scores = np.concatenate(scores)
        seg_part = np.concatenate(seg_part)

        # Filter boxes
        boxes, classes, scores, seg_part = self.filter_boxes(boxes, scores, classes_conf, seg_part)

        if boxes is None or len(boxes) == 0:
            return None, None, None, None

        # NMS
        nboxes, nclasses, nscores, nseg_part = [], [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]
            sp = seg_part[inds]

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

                if len(keep) > self.MAX_DETECT:
                    break

            if len(keep) > 0:
                nboxes.append(b[keep])
                nclasses.append(c[keep])
                nscores.append(s[keep])
                nseg_part.append(sp[keep])

        if not nboxes:
            return None, None, None, None

        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)
        seg_part = np.concatenate(nseg_part)

        # Process segmentation
        ph, pw = proto.shape[-2:]
        proto = proto.reshape(seg_part.shape[-1], -1)
        seg_img = np.matmul(seg_part, proto)
        seg_img = 1 / (1 + np.exp(-seg_img))  # sigmoid
        seg_img = seg_img.reshape(-1, ph, pw)

        # Interpolate masks to original image size
        seg_img = torch.tensor(seg_img)
        seg_img = F.interpolate(seg_img[None], size=(self.IMG_SIZE[1], self.IMG_SIZE[0]), mode='bilinear', align_corners=False)[0]
        seg_img = self._crop_mask(seg_img, torch.tensor(boxes))
        seg_img = seg_img.numpy()
        seg_img = seg_img > 0.5

        return boxes, classes, scores, seg_img

    def get_true_mask(self, seg_mask, img_shape, pad_info):
        """Get mask for original image considering padding and scaling.
        
        Args:
            seg_mask: Original segmentation mask
            img_shape: Original image shape (h, w)
            pad_info: Padding information (ratio, (dw, dh))
        Returns:
            Mask aligned with original image
        """
        ratio, (dw, dh) = pad_info
        ori_h, ori_w = img_shape
        
        # 计算letterbox后的实际图像尺寸（不包含padding）
        unpad_h = int(round(ori_h * ratio))
        unpad_w = int(round(ori_w * ratio))
        
        # 从640x640的mask中裁剪出实际图像区域（去除padding）
        mask_h, mask_w = seg_mask.shape
        dh = int(dh)
        dw = int(dw)
        seg_mask = seg_mask[dh:mask_h-dh, dw:mask_w-dw]
        
        # 缩放到原始图像大小
        seg_mask = cv2.resize(seg_mask.astype(np.uint8), (ori_w, ori_h), 
                             interpolation=cv2.INTER_NEAREST)
        
        return seg_mask

    def merge_seg(self, image, seg_img, classes, pad_info):
        """Merge segmentation masks with the original image."""
        h, w = image.shape[:2]
        for i in range(len(seg_img)):
            # Get properly aligned mask for original image
            seg = self.get_true_mask(seg_img[i], (h, w), pad_info)
            seg = cv2.cvtColor(seg, cv2.COLOR_GRAY2BGR)
            seg = seg * self.color(classes[i])
            seg = seg.astype(np.uint8)
            image = cv2.add(image, seg)
        return image

    def inference(self, frame):
        """Inference function."""
        orig_h, orig_w = frame.shape[:2]
        
        # Preprocess
        img, ratio, (dw, dh) = self.letter_box(frame, self.IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Inference
        outputs = self.rknn.inference(inputs=[img])
        
        # Post process
        boxes, classes, scores, seg_img = self.post_process(outputs)
        
        # Pack padding info
        pad_info = (ratio, (dw, dh))
        
        return boxes, classes, scores, seg_img, pad_info

    def draw(self, image, boxes, scores, classes):
        """Draw detection results."""
        for box, score, cl in zip(boxes, scores, classes):
            top, left, right, bottom = [int(_b) for _b in box]
            print("%s @ (%d %d %d %d) %.3f" % (self.CLASSES[cl], top, left, right, bottom, score))
            cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
            cv2.putText(image, '{0} {1:.2f}'.format(self.CLASSES[cl], score),
                        (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='../model/yolov8_seg.rknn', help='model path')
    parser.add_argument('--target', type=str, default='rk3588', help='target platform')
    parser.add_argument('--camera_id', type=int, default=21, help='camera device id')
    args = parser.parse_args()

    # Initialize model
    detector = YOLO_SEG_RKNN(args.model_path, args.target)
    
    # Open camera
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    
    print("Press 'q' to quit")
    
    # Initialize FPS calculation
    fps = 0
    prev_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        # Perform detection and segmentation
        boxes, classes, scores, seg_img, pad_info = detector.inference(frame)
        
        # Draw results
        if boxes is not None:
            ratio, (dw, dh) = pad_info
            # Transform boxes to original image coordinates
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / ratio
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / ratio
            
            # Draw boxes
            frame = detector.draw(frame, boxes, scores, classes)
            
            # Draw segmentation with proper alignment
            frame = detector.merge_seg(frame, seg_img, classes, pad_info)
        
        # Display FPS
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('YOLOv8 Detection & Segmentation', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    detector.rknn.release()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='../model/yolov8_seg.rknn', help='model path')
    parser.add_argument('--target', type=str, default='rk3588', help='target platform')
    parser.add_argument('--camera_id', type=int, default=21, help='camera device id')
    args = parser.parse_args()

    # Initialize model
    detector = YOLO_SEG_RKNN(args.model_path, args.target)
    
    # Open camera
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    
    print("Press 'q' to quit")
    
    # Initialize FPS calculation
    fps = 0
    frame_count = 0
    prev_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        # Perform detection and segmentation
        boxes, classes, scores, seg_img, ratio, (dw, dh), (orig_h, orig_w) = detector.inference(frame)
        
        # Draw results
        if boxes is not None:
            # Transform boxes to original image coordinates
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / ratio
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / ratio
            
            # Draw boxes
            frame = detector.draw(frame, boxes, scores, classes)
            
            # Draw segmentation
            frame = detector.merge_seg(frame, seg_img, classes)
        
        # Display FPS
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('YOLOv8 Detection & Segmentation', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    detector.rknn.release()

if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()