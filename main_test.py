from keras import backend as K
from models.keras_ssd7 import build_model
from keras.models import load_model
from keras.optimizers import Adam
from scipy.misc import imread
import numpy as np
from matplotlib import pyplot as plt

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization
from data_generator.object_detection_2d_data_generator import DataGenerator
from eval_utils.average_precision_evaluator import Evaluator

img_height = 300
img_width = 300
n_classes = 1
model_mode = 'inference'

scales = [0.08, 0.16, 0.32, 0.64, 0.96] # An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.
aspect_ratios = [0.5, 1.0, 2.0] # The list of aspect ratios for the anchor boxes
two_boxes_for_ar1 = True # Whether or not you want to generate two anchor boxes for aspect ratio 1
steps = None # In case you'd like to set the step sizes for the anchor box grids manually; not recommended
offsets = None # In case you'd like to set the offsets for the anchor box grids manually; not recommended
clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [1.0, 1.0, 1.0, 1.0] # The list of variances by which the encoded target coordinates are scaled
normalize_coords = True # Whether or not the model is supposed to use coordinates relative to the image size <--- tu było true
mean_color = [123, 117, 104]
intensity_mean = 127.5 # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.
intensity_range = 127.5 # Set this to your preference (maybe `None`). The current settings transform the input pixel values to the interval `[-1,1]`.

K.clear_session() # Clear previous models from memory.

model = build_model(image_size=(img_height, img_width, 3),
                    n_classes=n_classes,
                    mode=model_mode,
                    l2_regularization=0.0005,
                    scales=scales,
                    aspect_ratios_global=aspect_ratios,
                    aspect_ratios_per_layer=None,
                    two_boxes_for_ar1=two_boxes_for_ar1,
                    steps=steps,
                    offsets=offsets,
                    clip_boxes=clip_boxes,
                    variances=variances,
                    normalize_coords=normalize_coords,
                    subtract_mean=intensity_mean,
                    divide_by_stddev=intensity_range)

weights_path = 'borowki_1_klasa/ssd7_pascal_07+12_epoch-02_loss-0.8794_val_loss-0.5052.h5'

model.load_weights(weights_path, by_name=True)
model.summary()
# 3: Compile the model so that Keras won't complain the next time you load it.

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

dataset = DataGenerator()

# The XML parser needs to now what object class names to look for and in which order to map them to integers.
classes = ['background', 'borowka_ok']

anotations_borowki_ok_test = 'datasets/Oznaczone_borowki_rgb/borowka_rgb_ok_test'
anotations_borowki_nok_test = 'datasets/Oznaczone_borowki_rgb/borowka_rgb_nok_test'
anotations_borowki_ok_train = 'datasets/Oznaczone_borowki_rgb/borowka_rgb_ok'
anotations_borowki_nok_train = 'datasets/Oznaczone_borowki_rgb/borowka_rgb_nok'

set_borowki_ok_test = 'datasets/Oznaczone_borowki_rgb/borowka_rgb_ok_test/borowka_ok_test.txt'
set_borowki_nok_test = 'datasets/Oznaczone_borowki_rgb/borowka_rgb_nok_test/borowka_nok_test.txt'
set_borowki_ok_train = 'datasets/Oznaczone_borowki_rgb/borowka_rgb_ok/borowka_ok_train.txt'
set_borowki_nok_train = 'datasets/Oznaczone_borowki_rgb/borowka_rgb_nok/borowka_nok_train.txt'


dataset.parse_xml(images_dirs=[anotations_borowki_ok_test],
                      image_set_filenames=[set_borowki_ok_test],
                      annotations_dirs=[anotations_borowki_ok_test],
                      classes=classes,
                      include_classes='all',
                      exclude_truncated=False,
                      exclude_difficult=True,
                      ret=False)

evaluator = Evaluator(model=model,
                      n_classes=n_classes,
                      data_generator=dataset,
                      model_mode=model_mode)

results = evaluator(img_height=img_height,
                    img_width=img_width,
                    batch_size=8,
                    data_generator_mode='resize',
                    round_confidences=False,
                    matching_iou_threshold=0.5,
                    border_pixels='include',
                    sorting_algorithm='quicksort',
                    average_precision_mode='sample',
                    num_recall_points=11,
                    ignore_neutral_boxes=True,
                    return_precisions=True,
                    return_recalls=True,
                    return_average_precisions=True,
                    verbose=True)

mean_average_precision, average_precisions, precisions, recalls = results

for i in range(1, len(average_precisions)):
    print("{:<14}{:<6}{}".format(classes[i], 'AP', round(average_precisions[i], 3)))
print()
print("{:<14}{:<6}{}".format('','mAP', round(mean_average_precision, 3)))
