import cv2
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import csv  # Import csv for saving keypoints
import os
import glob
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt, strip_optimizer, xyxy2xywh
from utils.plots import output_to_keypoint, plot_one_box_kpt, colors

@torch.no_grad()
def run(poseweights="yolov7-w6-pose.pt", source_folder="videos/", output_folder="outputs/", device='cpu', view_img=False,
        save_conf=False, line_thickness=3, hide_labels=False, hide_conf=True):

    # Get all video files in the source folder
    video_files = glob.glob(os.path.join(source_folder, '*.mp4'))  # Adjust the extension if needed

    for video_path in video_files:
        frame_count = 0
        total_fps = 0
        time_list = []
        fps_list = []

        device = select_device(opt.device)  # Select GPU if available
        half = device.type != 'cpu'

        # Load the model
        model = attempt_load(poseweights, map_location=device)
        _ = model.eval()
        names = model.module.names if hasattr(model, 'module') else model.names

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f'Error while trying to read video {video_path}. Please check path again')
            continue

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        # Create a VideoWriter object
        out_video_name = os.path.basename(video_path).split('.')[0]
        out = cv2.VideoWriter(os.path.join(output_folder, f"{out_video_name}_keypoint.mp4"),
                              cv2.VideoWriter_fourcc(*'mp4v'), 30,
                              (frame_width, frame_height))

        # Create a CSV file to save keypoints
        csv_file_path = os.path.join(output_folder, f"{out_video_name}_keypoints.csv")
        csv_file = open(csv_file_path, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        # Write header for CSV
        header = ['Frame', 'Keypoint', 'X', 'Y', 'Confidence']
        csv_writer.writerow(header)

        while cap.isOpened():
            print(f"Processing Frame {frame_count + 1} of {out_video_name}")

            ret, frame = cap.read()

            if ret:
                orig_image = frame
                # Resize image to the desired size for the model
                image = letterbox(orig_image, (640, 640), stride=64, auto=True)[0]  # Adjust size as needed
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = transforms.ToTensor()(image)
                image = image.unsqueeze(0)  # Add batch dimension
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

                for i, pose in enumerate(output_data):
                    if len(pose):
                        for c in pose[:, 5].unique():
                            n = (pose[:, 5] == c).sum()
                            print("Number of Objects in Current Frame: {}".format(n))

                        for det_index, (*xyxy, conf, cls) in enumerate(reversed(pose[:, :6])):
                            c = int(cls)
                            kpts = pose[det_index, 6:]

                            # Write each keypoint to CSV
                            for j in range(len(kpts) // 3):  # Assuming kpts contains x, y, confidence for each keypoint
                                x = kpts[j * 3].item()
                                y = kpts[j * 3 + 1].item()
                                confidence = kpts[j * 3 + 2].item()
                                csv_writer.writerow([frame_count + 1, j + 1, x, y, confidence])

                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            plot_one_box_kpt(xyxy, im0, label=label, color=colors(c, True),
                                             line_thickness=line_thickness, kpt_label=True, kpts=kpts, steps=3, 
                                             orig_shape=im0.shape[:2])

                end_time = time.time()
                elapsed_time = end_time - start_time

                # Check to avoid division by zero
                if elapsed_time > 0:
                    fps = 1 / elapsed_time
                else:
                    fps = float('inf')  # or set to 0, depending on your needs

                total_fps += fps
                frame_count += 1

                fps_list.append(total_fps)
                time_list.append(elapsed_time)

                if view_img:
                    cv2.imshow("YOLOv7 Pose Estimation Demo", im0)
                    cv2.waitKey(1)

                out.write(im0)

            else:
                break

        cap.release()
        csv_file.close()  # Close the CSV file
        avg_fps = total_fps / frame_count if frame_count > 0 else 0
        print(f"Average FPS for {out_video_name}: {avg_fps:.3f}")

        # Plot the comparison graph
        plot_fps_time_comparision(time_list=time_list, fps_list=fps_list)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source-folder', type=str, default='videos/', help='folder containing videos')
    parser.add_argument('--output-folder', type=str, default='outputs/', help='folder to save output CSV and video')
    parser.add_argument('--device', type=str, default='0', help='gpu id (0 for first GPU, etc.) or cpu')
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