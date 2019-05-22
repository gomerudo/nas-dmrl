"""This is a simple python script to explore a set of TFRecords.

The main goal is to manually verify the correctness of the TFRecords, in terms
of the number of classes available and the elements per class.
"""

import glob
import argparse
import tensorflow as tf


def get_sorted_tfrecords(src_dir):
    """Sort the tfrecords that we obtain."""
    return sorted(
        glob.glob("{dir}/*.tfrecords".format(dir=src_dir))
    )


def make_summary_dict(tfrecords_list):
    summary = dict()

    # Iterate over all files in the directory
    for tf_record in tfrecords_list:
        # Obtain the record_iterator
        record_iterator = tf.python_io.tf_record_iterator(path=tf_record)
        
        # Iterate over the elements in the TFRecord, i.e. over the observations
        for string_record in record_iterator:
            # Parse the record
            example = tf.train.Example()
            example.ParseFromString(string_record)

            # Obtain the label value
            label = example.features.feature.get("label").int64_list.value[0]

            # Update the summary
            try:
                summary[str(label)] += 1
            except Exception as ex:
                summary[str(label)] = 1

    return summary


def print_summary(summary_dict):

    n_classes = len(summary_dict.keys())

    print("{l} SUMMARY {r}".format(l="="*30, r="="*30))
    print("\n- Total number of classes: {tc}".format(tc=n_classes))
    # print("\nClasses are: \n{c}".format(c=classes))
    print("- Summary of elements per class:\n")

    print("\t {l} ".format(l="-"*16))
    print("\t| Class\t | N\t |")
    print("\t {l} ".format(l="-"*16))
    count = 0
    for key, value in summary_dict.items():
        count += value
        print("\t| {cl}\t | {co}\t |".format(cl=key, co=value))
    print("\t {l} ".format(l="-"*16))
    print("\t| Total\t | {t}\t |".format(t=count))
    print("\t {l} ".format(l="-"*16))


if __name__ == '__main__':
    # Define the arguments

    parser = argparse.ArgumentParser(
        description='TFRecords explorer for meta-dataset'
    )
    parser.add_argument('--path', action="store", dest="tfrecords_dir")

    # Obtain the arguments of interest
    cmd_args = parser.parse_args()
    tfrecords_dir = cmd_args.tfrecords_dir
    print(
        "Using {tfd} as the source TFRecords directory".format(
            tfd=tfrecords_dir
        )
    )

    tfrecords_list = get_sorted_tfrecords(src_dir=tfrecords_dir)
    summary_dict = make_summary_dict(tfrecords_list)

    print_summary(summary_dict)
