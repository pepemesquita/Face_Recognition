import argparse
import cv2
import dlib
# contruct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='input/image_1.jpg',
                    help='path to the input image')
parser.add_argument('-u', '--upsample', default=None, type=int,
                    help='factor by which to upsample the image, default None, ' + \
                          'pass 1, 2, 3, ...')
args = vars(parser.parse_args())

# read the image and convert to RGB color format
image = cv2.imread(args['input'])
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# path for saving the result image
save_name = f"outputs/{args['input'].split('/')[-1].split('.')[0]}_u{args['upsample']}.jpg"
# initilaize the Dlib face detector
detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')

