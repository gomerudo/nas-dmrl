"""Export a TFRecords set of files to a csv, for the meta-dataset."""

import glob
import argparse
import pandas as pd
import tensorflow as tf


def get_sorted_tfrecords(src_dir):
    """Sort the tfrecords that we obtain."""
    return sorted(
        glob.glob("{dir}/*.tfrecords".format(dir=src_dir))
    )


def export_to_csv(tfrecords_list, img_size=84, export_path="./exported"):
    # 1. Read the TFRecord data from the files
    tfrecord_data = tf.data.TFRecordDataset(tfrecords_list)

    # 2. Get the dataset as tf.Dataset object
    dataset = _input_fn(
        tfrecord_data, _n_elements(tfrecords_list), img_size=img_size
    )

    # 3. Initialize the main csv file
    imgs_df = pd.DataFrame(
        columns=["f{i}".format(i=x) for x in range(img_size*img_size*3)]
        ["label"]
    )
    outfile = open(export_path, 'w')
    imgs_df.to_csv(outfile, index=False)
    outfile.close()

    # Iterate over all images. For each batch, we append to the csv file to
    # avoid memory issues if the dataset is too big.
    for idx, batch in enumerate(dataset):
        print("Processing batch #{i}".format(i=idx+1))
        imgs_df = pd.DataFrame()
        for img, label in zip(batch[0]['x'], batch[1]):
            img_np = img.numpy()
            label_np = int(label.numpy())
            imgs_df = imgs_df.append([list(img_np.flatten()) + [label_np]])
        outfile = open(export_path, 'a')
        imgs_df.to_csv(outfile, index=False, header=False)
        outfile.close()


def _n_elements(tfrecords_list):
    c = 0
    for fn in tfrecords_list:
        for _ in tf.python_io.tf_record_iterator(fn):
            c += 1
    return c


def _parser(record, image_size):
    # the 'features' here include your normal data feats along
    # with the label for that data
    features = {
        'image': tf.FixedLenFeature([], dtype=tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    }

    parsed = tf.parse_single_example(record, features)

    # The image (features)
    image_decoded = tf.image.decode_jpeg(parsed['image'], channels=3)
    image_resized = tf.image.resize_images(
        image_decoded,
        [image_size, image_size],
        method=tf.image.ResizeMethod.BICUBIC,
        align_corners=True
    )
    image_normalized = image_resized / 255.0

    # The label
    label = tf.cast(parsed['label'], tf.int32)

    return {'x': image_normalized}, label


def _input_fn(tfrecord_data, length, batch_size=128, img_size=84):
    dataset = tfrecord_data 

    dataset = dataset.map(lambda record: _parser(record, img_size))
    dataset = dataset.batch(batch_size)
    # iterator = dataset.make_one_shot_iterator()

    return dataset


if __name__ == '__main__':
    # Define the arguments

    parser = argparse.ArgumentParser(
        description='TFRecords explorer for meta-dataset'
    )
    parser.add_argument('--src_dir', action="store", dest="tfrecords_dir")
    parser.add_argument('--imgsize', action="store", dest="imgsize")
    parser.add_argument('--target_file', action="store", dest="export_file")

    # Obtain the arguments of interest
    cmd_args = parser.parse_args()
    tfrecords_dir = cmd_args.tfrecords_dir
    imgsize = int(cmd_args.imgsize)
    export_file = cmd_args.export_file

    print(
        "Using {tfd} as the source TFRecords directory".format(
            tfd=tfrecords_dir
        )
    )
    print(
        "Image shape will be {w}x{h}x3".format(w=imgsize, h=imgsize)
    )
    print(
        "Export file is {ef}".format(ef=export_file)
    )

    tf.enable_eager_execution()

    tfrecords_list = get_sorted_tfrecords(src_dir=tfrecords_dir)
    export_to_csv(tfrecords_list, imgsize, export_file)
