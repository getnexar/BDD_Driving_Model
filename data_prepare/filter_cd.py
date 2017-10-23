import argparse as _argparse
import os
import sys
import subprocess
from ffprobe import FFProbe

parser = _argparse.ArgumentParser()
parser.add_argument('--video_duration',
                    default=2,
                    help='expected video duration',
                    type=int)

parser.add_argument('--dataset_path',
                    default='/opt/nexar/projects/BDD_Driving_Model/23_10',
                    help='dataset path',
                    type=str)

def probe_file(filename):
    cmnd = ['ffprobe', '-show_format', '-pretty', filename]
    p = subprocess.Popen(cmnd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #print filename
    out, err =  p.communicate()
    duration = out.split('\n')
    whole_time = 0
    if len(duration)>8:
        time = duration[7].split(':')
        hour =  time[-3].split('=')[1]
        minute = time[-2]
        second = time[-1]
    #print hour, minute, second
        whole_time = float(hour)*3600 + float(minute)*60 + float(second)
    return whole_time

def filter_dataset(dataset_path):

    rides_path = os.path.join(dataset_path, 'ride')

    if not os.path.exists(rides_path):
        print dataset_path + " doesn't exist"
        return

    output_index = os.path.join(dataset_path, "video_filtered_" + str(args.video_duration) + ".txt")

    count = 0

    for item in os.listdir(rides_path):
        incident_dir = os.path.join(rides_path, item)
        if not os.path.isdir(incident_dir):
            continue

        for video in os.listdir(incident_dir):
            if video.endswith(".mov"):
                # now we get a video file
                this_video = os.path.join(incident_dir, video)

                count = count + 1
                if count % 1000 == 0:
                    print(count)
                duration = probe_file(this_video.strip())

                if duration >= args.video_duration:
                    with open(output_index, "a") as myfile:
                        myfile.write(os.path.abspath(this_video) + "\n")
                else:
                    print(
                    this_video, "video duration(" + str(duration) + "s) is less than " + int(args.video_duration) + "s")


if __name__ == '__main__':
    args = parser.parse_args()

    dataset_path = args.dataset_path

    filter_dataset(os.path.join(dataset_path, '0/test'))
    filter_dataset(os.path.join(dataset_path, '0/train'))
    filter_dataset(os.path.join(dataset_path, '0/validation'))

    filter_dataset(os.path.join(dataset_path, '1/test'))
    filter_dataset(os.path.join(dataset_path, '1/train'))
    filter_dataset(os.path.join(dataset_path, '1/validation'))