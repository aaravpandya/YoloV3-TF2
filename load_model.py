from utils import augment_img, save_model_to_json
import cv2 as cv
import tensorflow as tf
from tensorflow import keras


# import read_tfrecord

import numpy as np
import json
config_path = "config"
with open(config_path, 'r') as fp:
	config = json.load(fp)
TF_FORCE_GPU_ALLOW_GROWTH = True

nb_class = 1


def get_dataset(paths=config['generated_parameters']['paths']):

  raw_image_dataset = tf.data.TFRecordDataset(paths)

  feature = {
      'height': tf.io.FixedLenFeature([], tf.int64),
      'width': tf.io.FixedLenFeature([], tf.int64),
      'depth': tf.io.FixedLenFeature([], tf.int64),
      'name':  tf.io.VarLenFeature(tf.string),
      'xmin':  tf.io.VarLenFeature(tf.int64),
      'xmax':  tf.io.VarLenFeature(tf.int64),
      'ymin':  tf.io.VarLenFeature(tf.int64),
      'ymax':  tf.io.VarLenFeature(tf.int64),
      'image_raw': tf.io.FixedLenFeature([], tf.string),
  }

  def _parse_image_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, feature)

  parsed_image_dataset = raw_image_dataset.map(_parse_image_function)

  return parsed_image_dataset


def get_model(anchors=config['generated_parameters']['anchors'], labels=config['generated_parameters']['labels']):

    model_path = "/home/wisenet/Downloads/keras-yolo3/backend.h5"
    model_arch_path = "model.json"
    with open(model_arch_path, 'r') as fp:
        model_arch = json.load(fp)
    # print(model_arch)
    output_size = int((len(anchors)/6)*(5+len(labels)))
    out_1 = next(item for i, item in enumerate(
    	model_arch['config']['layers']) if item["config"]["name"] == "conv_coco1")
    out_2 = next(item for i, item in enumerate(
    	model_arch['config']['layers']) if item["config"]["name"] == "conv_coco2")
    out_3 = next(item for i, item in enumerate(
    	model_arch['config']['layers']) if item["config"]["name"] == "conv_coco3")
    out_1['config']['filters'] = output_size
    out_2['config']['filters'] = output_size
    out_3['config']['filters'] = output_size
    model_arch_s = json.dumps(model_arch)
    model = tf.keras.models.model_from_json(model_arch_s)
    model_trained = keras.models.load_model(model_path)
    # print(model_trained.layers[1].get_weights())
    # print(len(model_trained.get_weights()))
    for i, layer in enumerate(model_trained.layers[0:-3]):
        model.layers[i].set_weights(layer.get_weights())
    del model_trained
    return model


def aug_image(image, height, width, depth,bbox,input_sizes, jitter=config['input_parameters']['jitter']):
  seed = tf.random.experimental.get_global_generator().make_seeds(7)  # dont place on gpu

  index = tf.reduce_mean(tf.random.stateless_uniform([1], minval = 0, maxval=input_sizes.shape[0],seed = seed[:,0], dtype=tf.int32)) #bug : cannot make maxval equal to the len of input sizes
  net_h = input_sizes[index]
  net_w = input_sizes[index]
  dw = jitter * width
  dh = jitter * height
  
  new_aspect_ratio = (width + tf.random.stateless_uniform([1], minval=-dw, maxval=dw, seed=seed[:,1],dtype=tf.float32, name='Random_aspect_ratio_tensor_op'))/(
      height + tf.random.stateless_uniform([1], minval=-dh, maxval=dh, seed=seed[:,2],dtype=tf.float32, name='Random_aspect_ratio_tensor_op_2'))
  scale = tf.random.stateless_uniform([1], minval = 0.25, maxval = 2, seed = seed[:,3], dtype=tf.float32)

  #autograph 
  if(tf.math.less(new_aspect_ratio,1)):
    new_h = tf.cast(tf.math.floor(scale * net_h),tf.int32)
    new_w = tf.cast(tf.math.floor(net_h * new_aspect_ratio),tf.int32)
  else:
    new_w = tf.cast(tf.math.floor(scale * net_w),tf.int32)
    new_h = tf.cast(tf.math.floor(net_w / new_aspect_ratio),tf.int32)
  # print(new_w.shape)
  size = tf.stack([new_h,new_w])

  # scaling
  image = tf.image.resize(image, size[:,0])
  # cropping/padding
  s = tf.cast(width,tf.int32)-new_w
  # print(s.shape)
  dx_ = tf.reduce_mean(tf.cast(net_w,tf.int32)-new_w)
  if(dx_ < 0):
    dx = tf.random.stateless_uniform([1], minval = dx_, maxval=0,seed = seed[:,4], dtype=tf.int32)
  elif(dx_ > 0):
    dx = tf.random.stateless_uniform([1], minval = 0, maxval=dx_,seed = seed[:,4], dtype=tf.int32)
  else:
    dx = 0
  dy_ = tf.reduce_mean(tf.cast(net_h,tf.int32)-new_h)
  if(dy_ < 0):
    dy = tf.random.stateless_uniform([1], minval = dy_, maxval=0,seed = seed[:,5], dtype=tf.int32)
  elif(dy_ > 0):
    dy = tf.random.stateless_uniform([1], minval = 0, maxval=dy_,seed = seed[:,5], dtype=tf.int32)
  else:
    dy = 0
  # new_size = tf.stack([new_h + dy, new_w + dx])

  w_ = tf.reduce_mean(new_w + dx)
  h_ = tf.reduce_mean(new_h + dy)
  # print(w_.shape, h_.shape)
  image = tf.image.resize_with_crop_or_pad(image, h_, w_)
  print(image.shape)
  image = tf.image.resize_with_crop_or_pad(image, tf.cast(net_h,tf.int32), tf.cast(net_w,tf.int32))
  if(depth > 1):
    image = tf.image.random_hue(image, config['input_parameters']['hue_delta'])
    image = tf.image.random_saturation(image, config['input_parameters']['saturation_delta'][0],config['input_parameters']['saturation_delta'][1])
    image = tf.image.random_contrast(image ,config['input_parameters']['contrast_delta'][0],config['input_parameters']['contrast_delta'][1])
  flip = tf.random.stateless_uniform([1], minval = 0, maxval=2,seed = seed[:,6], dtype=tf.int32)
  if(flip == 1):
    image = tf.image.flip_left_right(image)
  
  # Scaling bounding boxes
  sx = tf.cast(new_w, tf.float32)/width
  sy = tf.cast(new_h, tf.float32)/height
  # dx = tf.cast(tf.reduce_mean(net_w - tf.cast(new_w,tf.float32)),tf.int32)
  # dy = tf.cast(tf.reduce_mean(net_h - tf.cast(new_h,tf.float32)),tf.int32)
  bbox_x = tf.cast((tf.cast(bbox[:,0:2],tf.float32)*sx), tf.int32) + tf.cast(dx/2, tf.int32) + tf.cast((net_w-tf.cast(w_,tf.float32))/2, tf.int32)
  bbox_x = tf.clip_by_value(bbox_x, clip_value_min = 0, clip_value_max = tf.cast(net_w,tf.int32))
  if(flip == 1):
    bbox_xmin = tf.cast(net_w,tf.int32) - bbox_x[:,1]
    bbox_xmax = tf.cast(net_w,tf.int32) - bbox_x[:,0]
    bbox_x = tf.concat([bbox_xmin,bbox_xmax],axis=0,name="flip_concat")
    bbox_x = tf.expand_dims(bbox_x, axis=0)
  print("bbox x shape " + str(bbox_x.shape))
  bbox_y = tf.cast((tf.cast(bbox[:,2:4],tf.float32)*sy), tf.int32) + tf.cast(dy/2, tf.int32) + tf.cast((net_h-tf.cast(h_,tf.float32))/2, tf.int32)
  bbox_y = tf.clip_by_value(bbox_y, clip_value_min = 0, clip_value_max = tf.cast(net_h,tf.int32))
  print(bbox_y.shape)
  bbox = tf.concat([bbox_x,bbox_y], axis=1,name="final_concat")

  return image, net_h, net_w, bbox



def data_augmentation(elem):
  image = tf.image.decode_image(
      elem['image_raw'], expand_animations=False, dtype=tf.dtypes.float32)
  height = tf.cast(elem['height'], tf.float32)
  width = tf.cast(elem['width'], tf.float32)
  depth = elem['depth']
  input_sizes = tf.range(288,480,32,dtype=tf.float32)
  names = elem['name'].values
  xmin = elem['xmin'].values
  xmax = elem['xmax'].values
  ymin = elem['ymin'].values
  ymax = elem['ymax'].values
  bbox = tf.transpose([xmin,xmax,ymin,ymax])
  image_1 , net_h, net_w ,bbox= aug_image(image,height,width, depth, bbox ,input_sizes)
  yolo_train = generate_train_output(bbox)
  return image, image_1,bbox, height, width


model = get_model()
keras.utils.plot_model(model, to_file='model_input_288.png', show_shapes=True, dpi=192)
# model.save
# dataset = get_dataset()
# # utils.save_model_to_json(model)
# for epoch in range(config['input_parameters']['num_epochs']):
#   dataset = dataset.map(data_augmentation)
#   for image_features in dataset:
#     print("**** IMAGE *****")
#     image = image_features[0]
#     image_1 = image_features[1]
#     bbox = image_features[2]
#     bbox_1 = image_features[3]
#     new_h = image_features[4]
#     new_w = image_features[5]
#     height  = image_features[6]
#     width = image_features[7]
#     dx = image_features[8]

    
#     dy = image_features[9]
#     bbox_y = image_features[10]
#     print("Original image shape" + str(image.shape))
#     print("Original image shape acc to dataset " + str(height.numpy())+ "  " +str(width.numpy()))
#     print("image shape after aug" + str(image_1.shape))
#     print("newh new w" + str(new_h.numpy()) + str(new_w.numpy()))
#     print("dx dy " + str(dx.numpy()) + " " + str(dy.numpy()))
#     # print(bbox_x.numpy().shape, bbox_y.numpy().shape)
#     print(bbox.numpy())
#     # print("flip " + str(flip.numpy()))

#     # print(type(image_raw))
#     # image = tf.image.decode_image(image_raw, dtype=tf.dtypes.float32).numpy()

#   #   image = image.astype(np.uint8)
#     # image = tf.expand_dims(image, 0)
#     # print(image.shape)
#     # bbox = bbox.numpy()
#     bbox_1 = bbox_1.numpy()
#     # image = image.numpy()
#     image_1 = image_1.numpy()
#     # for i in range(bbox.shape[0]):
#     #   cv.rectangle(image, (bbox[i,0], bbox[i,2]), (bbox[i,1], bbox[i,3]), (255,0,0), 2)
#     for i in range(bbox_1.shape[0]):
#       cv.rectangle(image_1, (bbox_1[i,0], bbox_1[i,2]), (bbox_1[i,1], bbox_1[i,3]), (255,0,0), 2)
#     # cv.imshow("f",image)
#     # cv.waitKey()
#     cv.imshow("f", image_1)
#     cv.waitKey()
   
#     # x = model(image)
#     # print(len(x))
#     # print(x[0].shape)
#     # print(x[1].shape)
#     # print(x[2].shape)
#     # break
#   break
