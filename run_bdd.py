import sys
import os
from subprocess import call
import argparse as _argparse

parser = _argparse.ArgumentParser()

parser.add_argument('--data_root',
                    help='data root directory',
                    type=str)

parser.add_argument('--bdd_root',
                    help='bdd project root directory',
                    type=str)

parser.add_argument('--model_name',
                    help='model name',
                    type=str)

def filter():
    call(["python", "data_prepare/filter_cd.py",
          "--dataset_path="+args.data_root,
          "--video_duration=2"])

def prepare_tfrecords(label, data_type):
    video_index = os.path.join(args.data_root, label+'/' + data_type + '/video_filtered_2.txt')
    if not os.path.exists(video_index):
        print "no video index for label(" + label + ") and data_type(" + data_type + ")"
        return

    call(["python", "data_prepare/prepare_tfrecords_cd.py",
          "--video_index=" + video_index,
          "--output_directory=" + os.path.join(args.data_root, label+'/' + data_type + '/tfrecords')])

def run_pretrained(label, data_type):
    input_dir = os.path.join(args.data_root, label + '/' + data_type + '/tfrecords')
    if not os.path.exists(input_dir):
        print "no tfrecords dir for label(" + label + ") and data type(" + data_type + ") "
        return

    model_path = os.path.join(args.bdd_root, 'data' + '/' + args.model_name + '/model.ckpt-126001.bestmodel')

    if not os.path.exists(model_path):
        print "no model data for " + args.model_name
        return

    call(["python", "run_pretrained.py",
          "--input_directory=" + input_dir,
          "--model_name=" + args.model_name,
          "--model_path=" + model_path,
          "--output_dir="+os.path.join(args.data_root, label + '/' + data_type + '/output')])

if __name__ == '__main__':
    args = parser.parse_args()

    filter()

    prepare_tfrecords('0', 'train')
    prepare_tfrecords('1', 'train')
    prepare_tfrecords('0', 'validation')
    prepare_tfrecords('1', 'validation')
    prepare_tfrecords('0', 'test')
    prepare_tfrecords('1', 'test')

    run_pretrained('0', 'train')
    run_pretrained('1', 'train')
    run_pretrained('0', 'validation')
    run_pretrained('1', 'validation')
    run_pretrained('0', 'test')
    run_pretrained('1', 'test')





