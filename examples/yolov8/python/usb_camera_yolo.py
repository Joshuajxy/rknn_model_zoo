import cv2
import numpy as np
from rknn.api import RKNN
import time
import os
import argparse

class YOLO_RKNN(object):
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
        
        # 初始化参数
        self.OBJ_THRESH = 0.25
        self.NMS_THRESH = 0.45
        self.IMG_SIZE = (640, 640)
        
        # 初始化RKNN
        self.rknn = RKNN(verbose=True)
        print(f'--> Loading RKNN model: {model_path}')
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            print('Load RKNN model failed!')
            exit(ret)
        print('Load RKNN model done')

        # 初始化运行时环境
        print('--> Init runtime environment')
        ret = self.rknn.init_runtime(target=target)
        if ret != 0:
            print('Init runtime environment failed!')
            exit(ret)
        print('Init runtime environment done')

    def letter_box(self, img, new_shape=(640, 640), pad_color=(0, 0, 0)):
        """信封式缩放"""
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
        """Distribution Focal Loss"""
        import torch
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
        """处理边界框"""
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
        """后处理"""
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

    def draw(self, image, boxes, scores, classes):
        """绘制检测结果"""
        for box, score, cl in zip(boxes, scores, classes):
            top, left, right, bottom = [int(_b) for _b in box]
            print("%s @ (%.3f %.3f %.3f %.3f) %.3f" % (self.CLASSES[cl], top, left, right, bottom, score))
            cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
            cv2.putText(image, '{0} {1:.2f}'.format(self.CLASSES[cl], score),
                        (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return image

    def inference(self, frame):
        """推理函数"""
        # 预处理
        img, ratio, (dw, dh) = self.letter_box(frame, self.IMG_SIZE, pad_color=(0,0,0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 推理
        outputs = self.rknn.inference(inputs=[img])
        
        # 后处理
        boxes, classes, scores = self.post_process(outputs)
        
        return boxes, classes, scores, ratio, (dw, dh)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='../model/yolov8.rknn', help='model path')
    parser.add_argument('--target', type=str, default='rk3588', help='target platform')
    parser.add_argument('--camera_id', type=int, default=21, help='camera device id')
    args = parser.parse_args()


    # 初始化模型
    detector = YOLO_RKNN(args.model_path, args.target)
    
    # 打开摄像头
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    
    print("Press 'q' to quit")
    
    # 初始化FPS计算
    fps = 0
    frame_count = 0
    prev_time = time.time()  # 添加这行初始化
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # 计算FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        # 执行检测
        boxes, classes, scores, ratio, (dw, dh) = detector.inference(frame)
        
        # 绘制结果
        if boxes is not None:
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / ratio
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / ratio
            frame = detector.draw(frame, boxes, scores, classes)
        
        # 显示FPS
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('YOLOv8 Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    detector.rknn.release()

if __name__ == "__main__":
    main()