# TODO()

import glob
import os
import tensorflow as tf
from itertools import groupby
from collections import defaultdict

def split_data_train_test(file_path_regex, label_position, test_size=0.2):
    
    '''
    Split data in to train and test
    
    Args
    file_paths_regex: lit of tuples(filename, file path)
    label_position: in the path which position is the lable at 
                    e.g. ./imagenet-dogs/Images/n02085620-Chihuahua/n02085620_10074.jpg'
                    n02085620-Chihuahua is the label so we would put 3 ''.split('.')
    test_size: deffault to 20% 
    
    '''
    image_filenames = glob.glob(file_path_regex)
    image_filename_path = map(lambda x: (x.split("/")[label_position], x), image_filenames)
    
              
    train = defaultdict(list)
    test = defaultdict(list)
    test_size = int(1 / 0.2)
    
    for cat, sub in groupby(image_filename_path, lambda x: x[0]):
        for i, s in enumerate(sub):
            if i % test_size == 0:
                test[cat].append(s)
            else:
                train[cat].append(s)
      
    return train, test

    
def write_records_file(dataset, record_location):

    sess = tf.Session()

    writer = None

    current_index = 0
    
    for folder, images_filenames in dataset.items():
        for image_filename in images_filenames:
            if current_index % 100 == 0:
                if writer:
                    writer.close()

                record_filename = "{record_location}-{current_index}.tfrecords".format(
                    record_location=record_location,
                    current_index=current_index)

                writer = tf.python_io.TFRecordWriter(record_filename)

            current_index += 1

            image_file = tf.read_file(image_filename[1])

            try:
                image = tf.image.decode_jpeg(image_file)
            except:
                print(image_filename)
                continue

            # convert to gray scale and resize images
            grayscale_image = tf.image.rgb_to_grayscale(image)        
            resized_image = tf.image.resize_images(grayscale_image, size = [250, 151])

            image_bytes = sess.run(tf.cast(resized_image, tf.uint8)).tobytes()            
            image_label = folder.encode('utf-8')

            example = tf.train.Example(features=tf.train.Features(feature={
                        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_label])),
                        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
                    }))

            writer.write(example.SerializeToString())

        writer.close()

# example         
train_data, test_data = split_data_train_test(file_path_regex = './imagenet-dogs/Images/n02*/*.jpg', label_position=3)
write_records_file(test_data, "./output/testing-images/testing-image")
write_records_file(train_data, "./output/training-images/train-image")

        
#os.path.expanduser('~/Downloads/')

image_filenames = glob.glob('./imagenet-dogs/Images/n02*/*.jpg')
image_filenames[0:2]


# create empty dict > split data here for training and testing (file names and file path)
train_data = defaultdict(list)
test_data = defaultdict(list)

# image_filename_with_breed = map(lambda x: (x.split("/")[3], x.split("/")[4]), image_filenames)
image_filename_with_breed = map(lambda x: (x.split("/")[3], x), image_filenames)
image_filename_with_breed[0:3]

for dog_breed, breed_images in groupby(image_filename_with_breed, lambda x: x[0]):
    # if dog_breed == 'n02085620-Chihuahua':
        for i, breed_image in enumerate(breed_images):
            if i % 5 == 0:
                test_data[dog_breed].append(breed_image)
            else:
                train_data[dog_breed].append(breed_image)


    
train_data, test_data = split_data_train_test(file_path_regex = './imagenet-dogs/Images/n02*/*.jpg', label_position=3)

# test_data.keys()
test_data['n02086910-papillon'][0:2]

# train_data.keys()

train_data, test_data = split_data_train_test(file_path_regex = './imagenet-dogs/Images/n02*/*.jpg', label_position=3)
write_records_file(test_data, "./output/testing-images/testing-image")
write_records_file(train_data, "./output/training-images/train-image")

k = train_data.keys()[0]
train_data[k][0][1]

# t1 = (1,2)
# t2 = (3,4)
# from itertools import product
# for a, b in product(t1,t2):
#     print a,b