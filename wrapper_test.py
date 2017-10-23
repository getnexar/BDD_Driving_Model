import wrapper
import tensorflow as tf
from tensorflow.core.example import example_pb2
from cStringIO import StringIO
from PIL import Image
from matplotlib.pyplot import imshow
import numpy as np

def main():
    a = wrapper.Wrapper("discrete_tcnn1",
                "/opt/nexar/projects/BDD_Driving_Model/data/discrete_tcnn1/model.ckpt-126001.bestmodel",
                20)

    example = example_pb2.Example()
    in_file = "/opt/nexar/projects/BDD_Driving_Model/input-example/tfrecords/validation/5ce3cf52-f5ab-4a8c-89c8-a6b698132064.tfrecords"

    count = 0
    for example_serialized in tf.python_io.tf_record_iterator(in_file):
        example.ParseFromString(example_serialized)
        feature_map = example.features.feature
        encoded = feature_map['image/encoded'].bytes_list.value
        print count
        count += 1

    file_jpgdata = StringIO(encoded[0])
    dt = Image.open(file_jpgdata)
    #imshow(np.asarray(dt))
    print a.observe_a_frame(np.asarray(dt))

    for i in range(len(encoded)):
        if i % 5 == 0:
            file_jpgdata = StringIO(encoded[0])
            dt = Image.open(file_jpgdata)
            arr = np.asarray(dt)
            out = a.observe_a_frame(arr)
            print out
            print i / 5



if __name__ == '__main__':
  main()