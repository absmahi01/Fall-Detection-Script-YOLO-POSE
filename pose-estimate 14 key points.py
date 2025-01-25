import cv2
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import csv  # Import csv for saving keypoints
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt, strip_optimizer, xyxy2xywh
from utils.plots import output_to_keypoint, plot_skeleton_kpts, colors, plot_one_box_kpt

@torch.no_grad()
def run(poseweights="yolov7-w6-pose.pt", source="football1.mp4", device='cpu', view_img=False,
        save_conf=False, line_thickness=3, hide_labels=False, hide_conf=True):

    frame_count = 0
    total_fps = 0
    time_list = []
    fps_list = []
    
    device = select_device(opt.device)
    half = device.type != 'cpu'

    model = attempt_load(poseweights, map_location=device)
    _ = model.eval()
    names = model.module.names if hasattr(model, 'module') else model.names

    if source.isnumeric():    
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print('Error while trying to read video. Please check path again')
        raise SystemExit()

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    vid_write_image = letterbox(cap.read()[1], (frame_width), stride=64, auto=True)[0]
    resize_height, resize_width = vid_write_image.shape[:2]
    out_video_name = f"{source.split('/')[-1].split('.')[0]}"
    out = cv2.VideoWriter(f"{source}_keypoint.mp4",
                          cv2.VideoWriter_fourcc(*'mp4v'), 30,
                          (resize_width, resize_height))

    # Create a CSV file to save keypoints
    csv_file = open(f"{source.split('/')[-1].split('.')[0]}_keypoints.csv", mode='w', newline='')
    csv_writer = csv.writer(csv_file)

    # Write header for CSV
    header = [
        'frame', 'nose_x', 'nose_y', 'nose_conf',
        'left_eye_x', 'left_eye_y', 'left_eye_conf',
        'right_eye_x', 'right_eye_y', 'right_eye_conf',
        'left_ear_x', 'left_ear_y', 'left_ear_conf',
        'right_ear_x', 'right_ear_y', 'right_ear_conf',
        'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_conf',
        'right_shoulder_x', 'right_shoulder_y', 'right_shoulder_conf',
        'left_elbow_x', 'left_elbow_y', 'left_elbow_conf',
        'right_elbow_x', 'right_elbow_y', 'right_elbow_conf',
        'left_wrist_x', 'left_wrist_y', 'left_wrist_conf',
        'right_wrist_x', 'right_wrist_y', 'right_wrist_conf',
        'left_hip_x', 'left_hip_y', 'left_hip_conf',
        'right_hip_x', 'right_hip_y', 'right_hip_conf',
        'left_knee_x', 'left_knee_y', 'left_knee_conf',
        'right_knee_x', 'right_knee_y', 'right_knee_conf',
        'left_ankle_x', 'left_ankle_y', 'left_ankle_conf',
        'right_ankle_x', 'right_ankle_y', 'right_ankle_conf'
    ]
    csv_writer.writerow(header)

    while cap.isOpened():
        print("Frame {} Processing".format(frame_count + 1))

        ret, frame = cap.read()
        
        if ret:
            orig_image = frame
            image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
            image = letterbox(image, (frame_width), stride=64, auto=True)[0]
            image_ = image.copy()
            image = transforms.ToTensor()(image)
            image = torch.tensor(np.array([image.numpy()]))
            
            image = image.to(device)
            image = image.float()
            start_time = time.time()
            
            with torch.no_grad():
                output_data, _ = model(image)

            output_data = non_max_suppression_kpt(output_data,
                                                    0.25,
                                                    0.65,
                                                    nc=model.yaml['nc'],
                                                    nkpt=model.yaml['nkpt'],
                                                    kpt_label=True)

            output = output_to_keypoint(output_data)
            im0 = image[0].permute(1, 2, 0) * 255
            im0 = im0.cpu().numpy().astype(np.uint8)
            im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

            # Initialize a row for the current frame with default values (e.g., -1 for missing points)
            keypoints_row = [frame_count + 1] + [-1] * 42  # 42 = 14 keypoints * 3 (x, y, confidence)

            for i, pose in enumerate(output_data):
                if len(output_data):
                    for c in pose[:, 5].unique():
                        n = (pose[:, 5] == c).sum()
                        print("No of Objects in Current Frame : {}".format(n))
                        
                    for det_index, (*xyxy, conf, cls) in enumerate(reversed(pose[:, :6])):
                        c = int(cls)
                        kpts = pose[det_index, 6:]

                        # Fill in the keypoints into the row
                        for j in range(len(kpts) // 3):  # Assuming kpts contains x, y, confidence for each keypoint
                            if j < 14:  # Ensure we don't exceed the number of defined keypoints
                                keypoints_row[j * 3 + 1] = kpts[j * 3].item()      # x
                                keypoints_row[j * 3 + 2] = kpts[j * 3 + 1].item()  # y
                                keypoints_row[j * 3 + 3] = kpts[j * 3 + 2].item()  # confidence
                        
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        plot_one_box_kpt(xyxy, im0, label=label, color=colors(c, True),
                                         line_thickness=line_thickness, kpt_label=True, kpts=kpts, steps=3, 
                                         orig_shape=im0.shape[:2])

            csv_writer.writerow(keypoints_row)  # Write the filled row to CSV

            end_time = time.time()
            fps = 1 / (end_time - start_time)
            total_fps += fps
            frame_count += 1
            
            fps_list.append(total_fps)
            time_list.append(end_time - start_time)

            if view_img:
                cv2.imshow("YOLOv7 Pose Estimation Demo", im0)
                cv2.waitKey(1)

            out.write(im0)

        else:
            break

    cap.release()
    csv_file.close()  # Close the CSV file
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")

    # Plot the comparison graph
    plot_fps_time_comparision(time_list=time_list, fps_list=fps_list)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='football1.mp4', help='video/0 for webcam')
    parser.add_argument('--device', type=str, default='cpu', help='cpu/0,1,2,3(gpu)')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    opt = parser.parse_args()
    return opt

def plot_fps_time_comparision(time_list, fps_list):
    plt.figure()
    plt.xlabel('Time (s)')
    plt.ylabel('FPS')
    plt.title('FPS and Time Comparison Graph')
    plt.plot(time_list, fps_list, 'b', label="FPS & Time")
    plt.savefig("FPS_and_Time_Comparison_pose_estimate.png")

def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    strip_optimizer(opt.device, opt.poseweights)
    main(opt)