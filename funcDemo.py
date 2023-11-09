import argparse
import time
from pathlib import Path
import sys
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
import time
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox, reloadFrame
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


class YoloCv():
    def __init__(self):
        self.source = opt.source
        self.weights = opt.weights
        self.view_img = opt.view_img
        self.save_txt = opt.save_txt
        self.imgsz = opt.img_size
        self.trace = not opt.no_trace
        self.stride = None
        self.model = None
        #self.dataset = reloadFrame(self.source, img_size=self.imgsz, stride=None)
    
    def detect(self):
        save_img = not opt.nosave and not self.source.endswith('.txt')  # save inference images
        webcam = self.source.isnumeric() or self.source.endswith('.txt') or self.source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
        # Directories
        self.save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        print("save_dir: {}".format(self.save_dir))

        # Initialize
        set_logging()
        device = select_device(opt.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(self.weights, map_location=device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        imgsz = check_img_size(self.imgsz, s=self.stride)  # check img_size

        if self.trace:
            self.model = TracedModel(self.model, device, opt.img_size)
        
        if half:
            self.model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Set Dataloader
        vid_path, vid_writer = None, None
        blindImg = False
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = reloadFrame(self.source, img_size=imgsz, stride=self.stride)
        else:
            view_img = check_imshow()
            dataset = reloadFrame(self.source, img_size=imgsz, stride=self.stride)
        
        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        # Run inference
        if device.type != 'cpu':
            self.model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(self.model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1

        print("start of for-loop")
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            if img[0][0][0] == 0:
                blindImg = True
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    self.model(img, augment=opt.augment)[0]

            # Inference
            t1 = time_synchronized()
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = self.model(img, augment=opt.augment)[0]
            t2 = time_synchronized()
            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t3 = time_synchronized()
            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                save_path = str(self.save_dir / p.name)  # img.jpg
                item = 0
                trackers = []
                txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                x0,x1,y0,y1=[list() for _ in range(4)] # 4 empty list for to store items
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if self.save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or view_img:  # Add bbox to image
                            label = f'{self.names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=1)
                            x0.insert(item,int(xyxy[0]))
                            y0.insert(item,int(xyxy[1]))
                            x1.insert(item,int(xyxy[2]))
                            y1.insert(item,int(xyxy[3]))
                            print("box: {}".format(xyxy))
                            #x0, x1, y0, y1 = int(xyxy[0]),int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                            item += 1

                # Print time (inference + NMS)
                print(f'{s}Done. 1({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

                # Stream results
                # if view_img:
                #     cv2.imshow(str(p), im0)
                #     cv2.waitKey(2000)  # 1 millisecond
                #     print("image window close")
                # print("end-loop")
                # cv2.destroyAllWindows()
                break
            else:
                print("bling")
                break
        else:
            print("not classify")
        
        return x0, y0, x1, y1
    
    def reloadFunc(self):
        imgsz = check_img_size(self.imgsz, s=self.stride)  # check img_size
        dataset = reloadFrame(self.source, img_size=imgsz, stride=self.stride)
        device = select_device(opt.device)
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img =  img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # Warmup
            if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    self.model(img, augment=opt.augment)[0]
            # Inference 
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = self.model(img, augment=opt.augment)[0]
            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            # Apply Classifier
            for i, det in enumerate(pred):  # detections per image
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                save_path = str(self.save_dir / p.name)  # img.jpg
                item = 0
                trackers = []
                txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                x0,x1,y0,y1=[list() for _ in range(4)] # 4 empty list for to store items
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if self.save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        #if save_img or view_img:  # Add bbox to image
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=1)
                        x0.insert(item,int(xyxy[0]))
                        y0.insert(item,int(xyxy[1]))
                        x1.insert(item,int(xyxy[2]))
                        y1.insert(item,int(xyxy[3]))
                        print("box: {}".format(xyxy))
                        #x0, x1, y0, y1 = int(xyxy[0]),int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                        item += 1

                # Stream results
                #if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(2000)  # 1 millisecond
                print("image window close")
                print("end-loop")
                # cv2.destroyAllWindows()
                break
            else:
                print("bling")
                break
        else:
            print("not classify")   
        return x0, y0, x1, y1

    def tracking(self):
        Firstload = False
        dect = False
        multiTracker = cv2.legacy.MultiTracker_create()
        window = cv2.VideoCapture(0)
        ret, frame = window.read()
        if not ret:
            print('Failed to read webcam')
            sys.exit(1)
        while True:
            ret, frame = window.read()
            if dect:
                success, boxes = multiTracker.update(frame)
                if success:
                    for i, box in enumerate(boxes):
                        x,y,w,h = [int(v) for v in box] 
                        #print("x: {} y: {} w: {} h: {} ".format(x,y,w,h))   
                        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                else:
                    dect = False
                    Firstload = False
                    print("lost contact")
            cv2.imshow("img",frame)
            ret_value = cv2.waitKey(1)
            if ret_value==ord('q') or ret_value==ord('Q'):
                break
            if Firstload:
                for target in range(len(x0)):
                    bbox = (x0[target], y0[target],x1[target],y1[target])  # (x, y, x1, y1)
                    #print("x0:{} y0: {} x1: {} y1: {}".format(x0[target], y0[target],x1[target],y1[target]))
                    width = x1[target] - x0[target]
                    height = y1[target] - y0[target]
                    blx = (x0[target], y0[target],width,height)
                    x, y, w, h = [int(v) for v in bbox]
                    tracker = cv2.legacy.TrackerKCF_create()
                    try:
                        multiTracker.add(tracker,frame,blx)
                        print("added")
                    except Exception as e:
                        print('Failed to add tracker', e)
                    Firstload = False
                    dect = True
            elif not Firstload and not dect:
                print("not Firstload")
                if ret_value==ord('r') or ret_value==ord('R'):
                    window.release()
                    dect = False
                    x0, y0, x1, y1 = self.reloadFunc()
                    Firstload = True if len(x0)>0 else False
                    while not Firstload:
                        x0, y0, x1, y1 = self.reloadFunc()
                        if len(x0)>0:
                            Firstload = True
                            break
                    multiTracker = cv2.legacy.MultiTracker_create()
                    window = cv2.VideoCapture(0)
            else:
                print("scanning")
        window.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1) # To prevent freezing after closing the window
        print("end")
            


            

    
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    #print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                yolo = YoloCv()
                x0, y0, x1, y1 = yolo.detect()
                yolo.tracking()
                strip_optimizer(opt.weights)
        else:
            yolo = YoloCv()
            x0, y0, x1, y1 = yolo.detect()
            yolo.tracking()
            
            
