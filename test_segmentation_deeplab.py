import os
import tarfile
from pathlib import Path

import argparse
import cv2
import glob
import numpy as np
import tensorflow as tf
import tqdm
from PIL import Image
from six.moves import urllib


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        # """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()
        graph_def = None
        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.GraphDef.FromString(file_handle.read())
                break

        tar_file.close()

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    def run(self, image):
        """Runs inference on a single image.

        Args:
          image: A PIL.Image object, raw input image.

        Returns:
          resized_image: RGB image resized from original input image.
          seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map


def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.

    Returns:
    A Colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap


def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
    label: A 2D array with integer type, storing the segmentation label.

    Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

    Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deeplab Segmentation')
    parser.add_argument('-i', '--image_dir', default=None, type=str,
                        help='Directory to search for images (*_img.png)')
    parser.add_argument('-v', '--video_dir', default=None, type=str,
                        help='Directory to search for videos (*.mp4)')
    args = parser.parse_args()

## setup ####################

    LABEL_NAMES = np.asarray([
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
    ])

    FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
    FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

    MODEL_NAME = 'xception_coco_voctrainval'  # @param ['mobilenetv2_coco_voctrainaug', 'mobilenetv2_coco_voctrainval', 'xception_coco_voctrainaug', 'xception_coco_voctrainval']

    _DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
    _MODEL_URLS = {
        'mobilenetv2_coco_voctrainaug':
            'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
        'mobilenetv2_coco_voctrainval':
            'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
        'xception_coco_voctrainaug':
            'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
        'xception_coco_voctrainval':
            'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
    }
    _TARBALL_NAME = _MODEL_URLS[MODEL_NAME]

    model_dir = 'deeplab_model'
    if not os.path.exists(model_dir):
        tf.io.gfile.makedirs(model_dir)

    download_path = os.path.join(model_dir, _TARBALL_NAME)
    if not os.path.exists(download_path):
        print('downloading model to %s, this might take a while...' % download_path)
        urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME],
                                   download_path)
        print('download completed! loading DeepLab model...')

    MODEL = DeepLabModel(download_path)
    print('model loaded successfully!')

    #######################################################################################
    # Images

    image_dir = args.image_dir
    if image_dir:
        image_paths = glob.glob(image_dir + '/*_img.png')
        image_paths.sort()
        print("Found {} images in {}".format(len(image_paths), image_dir))

        for image_path in tqdm.tqdm(image_paths):
            image = Image.open(image_path)

            res_im, seg = MODEL.run(image)

            seg = cv2.resize(seg.astype(np.uint8), image.size)

            mask_sel = (seg == 15).astype(np.float32)

            name = image_path.replace('img', 'masksDL')
            cv2.imwrite(name, (255 * mask_sel).astype(np.uint8))

        print('\nDone: ' + image_dir)
    else:
        print("No image dir specified!")

    #######################################################################################
    # Videos
    video_dir = args.video_dir

    if video_dir:
        video_paths = glob.glob(video_dir + '/*_raw.mp4')
        video_paths.sort()
        print("Found {} videos in {}".format(len(video_paths), video_dir))

        for video_path in tqdm.tqdm(video_paths):
            video = cv2.VideoCapture(video_path)
            num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(video.get(cv2.CAP_PROP_FPS))
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

            output_path = video_path.replace("raw", "masksDL").replace("mp4", "avi")

            video_writer = cv2.VideoWriter(output_path,
                                           cv2.VideoWriter_fourcc(*'png '),
                                           fps,
                                           (width, height))

            for i_frame in tqdm.trange(num_frames):
                ret, frame = video.read()

                if not ret:
                    print("Could not read video frame {}!".format(i_frame))

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)

                res_im, seg = MODEL.run(image)

                seg = cv2.resize(seg.astype(np.uint8), image.size)

                mask_sel = (seg == 15).astype(np.float32)

                # Make 3 channel image
                mask_sel = np.repeat(np.expand_dims(mask_sel, -1), 3, axis=2)

                video_writer.write((255 * mask_sel).astype(np.uint8))

            video.release()
            video_writer.release()
    else:
        print("No video dir specified!")
