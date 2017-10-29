import tensorflow as tf
from tensorflow.core.example import example_pb2
from cStringIO import StringIO
from PIL import Image
from matplotlib.pyplot import imshow
import numpy as np
import argparse as _argparse
import os
import wrapper
import itertools

parser = _argparse.ArgumentParser()
parser.add_argument('--input_directory',
                    default='/opt/nexar/projects/BDD_Driving_Model/dataset/train/tfrecords',
                    help='tfrecords directory',
                    type=str)
parser.add_argument('--model_name',
                    default='discrete_tcnn1',
                    help='pre-trained model',
                    type=str)
parser.add_argument('--model_path',
                    default='/opt/nexar/projects/BDD_Driving_Model/data/discrete_tcnn1/model.ckpt-126001.bestmodel',
                    help='pre-trained model file',
                    type=str)
parser.add_argument('--output_dir',
                    default='/opt/nexar/projects/BDD_Driving_Model/dataset/train/output-20171023',
                    help='model output directory',
                    type=str)

if __name__ == '__main__':

    args = parser.parse_args()

    dataset_path = args.input_directory

    model_out_name = os.path.join(args.output_dir , "model_vectors.bdd")

    a = wrapper.Wrapper(args.model_name,
                        args.model_path,
                        20)

    example = example_pb2.Example()

    for item in os.listdir(dataset_path):
        if item.endswith(".tfrecords"):
            a.reset()
            in_tfrecord = os.path.join(dataset_path, item)
            fprefix = item.split(".")[0]
            out_name = os.path.join(args.output_dir + "/videos", fprefix + ".bdd")

            if not os.path.exists(os.path.dirname(out_name)):
                os.makedirs(os.path.dirname(out_name))

            count = 0

            for example_serialized in tf.python_io.tf_record_iterator(in_tfrecord):
                example.ParseFromString(example_serialized)
                feature_map = example.features.feature
                encoded = feature_map['image/encoded'].bytes_list.value
                print count
                count += 1

            model_out = []

            for i in range(len(encoded)):
                if i % 5 == 0:
                    file_jpgdata = StringIO(encoded[i])
                    dt = Image.open(file_jpgdata)
                    arr = np.asarray(dt)
                    out = a.observe_a_frame(arr)

                    model_out.append(out[0][0].tolist())
                    #Hanna - debug file (add a flag)
                    with open(out_name, "a") as myfile:
                        np.savetxt(myfile, out[0], delimiter=',', newline='\n')

                    print out
                    print i / 5

            out_line = ""
            res = list(itertools.chain.from_iterable(model_out))
            out_line += ','.join(str(e) for e in res)
            with open(model_out_name, "a") as myfile:
                myfile.write( fprefix + ","+ out_line + '\n')
