import tensorflow as tf
import json

with open("config",'r') as fp:
    config = json.load(fp)
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
#   print(image.shape)
  #   Note the following net_h, net_w are different from the ones used above. The ones used above are used to scale randomly.
  #   The below ones are used to make image size same. It doesnt affect augmentation but makes batching possible.
  net_h = input_sizes[-1]
  net_w = input_sizes[-1]
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
  #   bbox_xmin = tf.cast(net_w,tf.int32) - bbox_x[:,1]
  #   bbox_xmax = tf.cast(net_w,tf.int32) - bbox_x[:,0]
  #   bbox_x = tf.concat([bbox_xmin,bbox_xmax],axis=0,name="flip_concat")
  #   bbox_x = tf.expand_dims(bbox_x, axis=0)
      bbox_x = tf.reverse(bbox_x, axis=[1])
      bbox_x = tf.math.abs(bbox_x - tf.cast(net_w,tf.int32))
#   print("bbox x shape " + str(bbox_x.shape))
  bbox_y = tf.cast((tf.cast(bbox[:,2:4],tf.float32)*sy), tf.int32) + tf.cast(dy/2, tf.int32) + tf.cast((net_h-tf.cast(h_,tf.float32))/2, tf.int32)
  bbox_y = tf.clip_by_value(bbox_y, clip_value_min = 0, clip_value_max = tf.cast(net_h,tf.int32))
#   print(bbox_y.shape)
  bbox = tf.concat([bbox_x,bbox_y], axis=1,name="final_concat")

  return image, net_h, net_w, bbox


def find_iou_train(anchor,bbox):
  '''
  Only used in data augmentation. Won't work with any other boxes.
  '''
  tf_abs = tf.math.abs
  tf_min = tf.math.minimum

  bbox_w = tf.expand_dims(tf_abs(bbox[:,0] - bbox[:,1]), axis = -1)
  bbox_h = tf.expand_dims(tf_abs(bbox[:,2] - bbox[:,3]), axis = -1)
  intersection = tf_min(bbox_w,anchor[0])*tf_min(bbox_h,anchor[1])
  union = (bbox_h*bbox_w) + (anchor[0]*anchor[1]) - intersection
  iou = tf.squeeze(tf.math.divide(intersection,union))
  return iou


def generate_train_output(bbox, net_h, net_w, anchors=config['generated_parameters']['anchors'], labels=config['generated_parameters']['labels'], max_boxes = config['generated_parameters']['max_objs_per_image']):
  
  grid_size = net_h/32 # downsample value, shift it to config
  anchor_size = tf.cast(len(anchors)/3, tf.float32)
  output_1 = tf.zeros((grid_size, grid_size, anchor_size , 5+len(labels))) # fix acnhors / 6
  output_2 = tf.zeros((grid_size*2, grid_size*2, anchor_size, 5+len(labels)))
  output_3 = tf.zeros((grid_size*4, grid_size*4, anchor_size, 5+len(labels)))
  t_boxes = tf.zeros((max_boxes,4))

  iou = []


  for anchor in anchors:
    iou.append(find_iou_train(anchor,bbox))
#   iou = 
  max_anchors = tf.math.argmax(iou,axis=0)
  if(tf.size(max_anchors)==1):
      max_anchors=tf.expand_dims(max_anchors,axis=0)
  selected_anchors = tf.cast(tf.gather(anchors, max_anchors),tf.float32)
  
  indexes = tf.math.floor(max_anchors/3)
  for i in range(len(max_anchors)):
      if(indexes[i] == 0):
          cx = tf.cast((bbox[i,0] + bbox[i,1]),tf.float32)*0.5
          cx = cx / float(net_w) * grid_size * 4 # sigma(t_x) + c_x
          cy = tf.cast((bbox[i,2] + bbox[i,3]),tf.float32)*0.5
          cy = cy / float(net_h) * grid_size * 4 # sigma(t_y) + c_y
          w = tf.math.log(tf.cast(tf.math.abs(bbox[i,0] - bbox[i,1]),tf.float32)) / selected_anchors[i,0] # t_w
          h = tf.math.log(tf.cast(tf.math.abs(bbox[i,2] - bbox[i,3]),tf.float32)) / selected_anchors[i,1] # t_h
          grid_x = tf.cast(tf.math.floor(cx),tf.int32)
          grid_y = tf.cast(tf.math.floor(cy),tf.int32)
          # assign ground truth x, y, w, h, confidence and class probs to y_batch
          indices = [ [grid_y, grid_x, tf.cast(max_anchors[i]%3,tf.int32), 0],
          [grid_y, grid_x, tf.cast(max_anchors[i]%3,tf.int32), 1],
          [grid_y, grid_x, tf.cast(max_anchors[i]%3,tf.int32), 2],
          [grid_y, grid_x, tf.cast(max_anchors[i]%3,tf.int32), 3],
          [grid_y, grid_x, tf.cast(max_anchors[i]%3,tf.int32), 4],
          [grid_y, grid_x, tf.cast(max_anchors[i]%3,tf.int32), 5]]
          updates = [cx,cy,w,h,1.,1.]
          output_3 = tf.tensor_scatter_nd_update(output_3,indices=indices, updates=updates)
        #   t_boxes.append([cx,cy,tf.cast(tf.math.abs(bbox[i,0] - bbox[i,1]),tf.float32),tf.cast(tf.math.abs(bbox[i,2] - bbox[i,3]),tf.float32)])
          t_indices = [[i]]
          t_updates = [[cx,cy,tf.cast(tf.math.abs(bbox[i,0] - bbox[i,1]),tf.float32),tf.cast(tf.math.abs(bbox[i,2] - bbox[i,3]),tf.float32)]]
          t_boxes = tf.tensor_scatter_nd_update(t_boxes,indices=t_indices,updates=t_updates)
      elif(indexes[i]==1):
          cx = tf.cast((bbox[i,0] + bbox[i,1]),tf.float32)*0.5
          cx = cx / float(net_w) * grid_size * 2 # sigma(t_x) + c_x
          cy = tf.cast((bbox[i,2] + bbox[i,3]),tf.float32)*0.5
          cy = cy / float(net_h) * grid_size * 2 # sigma(t_y) + c_y
          w = tf.math.log(tf.cast(tf.math.abs(bbox[i,0] - bbox[i,1]),tf.float32)) / selected_anchors[i,0] # t_w
          h = tf.math.log(tf.cast(tf.math.abs(bbox[i,2] - bbox[i,3]),tf.float32)) / selected_anchors[i,1] # t_h
          grid_x = tf.cast(tf.math.floor(cx),tf.int32)
          grid_y = tf.cast(tf.math.floor(cy),tf.int32)
          # assign ground truth x, y, w, h, confidence and class probs to y_batch
          indices = [ [grid_y, grid_x, tf.cast(max_anchors[i]%3,tf.int32), 0],
          [grid_y, grid_x, tf.cast(max_anchors[i]%3,tf.int32), 1],
          [grid_y, grid_x, tf.cast(max_anchors[i]%3,tf.int32), 2],
          [grid_y, grid_x, tf.cast(max_anchors[i]%3,tf.int32), 3],
          [grid_y, grid_x, tf.cast(max_anchors[i]%3,tf.int32), 4],
          [grid_y, grid_x, tf.cast(max_anchors[i]%3,tf.int32), 5]]
          updates = [cx,cy,w,h,1.,1.]
          output_2 = tf.tensor_scatter_nd_update(output_2,indices=indices, updates=updates)
        #   t_boxes.append([cx,cy,tf.cast(tf.math.abs(bbox[i,0] - bbox[i,1]),tf.float32),tf.cast(tf.math.abs(bbox[i,2] - bbox[i,3]),tf.float32)])
          t_indices = [[i]]
          t_updates = [[cx,cy,tf.cast(tf.math.abs(bbox[i,0] - bbox[i,1]),tf.float32),tf.cast(tf.math.abs(bbox[i,2] - bbox[i,3]),tf.float32)]]
          t_boxes = tf.tensor_scatter_nd_update(t_boxes,indices=t_indices,updates=t_updates)
      elif(indexes[i]==2):
          cx = tf.cast((bbox[i,0] + bbox[i,1]),tf.float32)*0.5
          cx = cx / float(net_w) * grid_size # sigma(t_x) + c_x
          cy = tf.cast((bbox[i,2] + bbox[i,3]),tf.float32)*0.5
          cy = cy / float(net_h) * grid_size # sigma(t_y) + c_y
          w = tf.math.log(tf.cast(tf.math.abs(bbox[i,0] - bbox[i,1]),tf.float32)) / selected_anchors[i,0] # t_w
          h = tf.math.log(tf.cast(tf.math.abs(bbox[i,2] - bbox[i,3]),tf.float32)) / selected_anchors[i,1] # t_h
          grid_x = tf.cast(tf.math.floor(cx),tf.int32)
          grid_y = tf.cast(tf.math.floor(cy),tf.int32)
          # assign ground truth x, y, w, h, confidence and class probs to y_batch
          indices = [ [grid_y, grid_x, tf.cast(max_anchors[i]%3,tf.int32), 0],
          [grid_y, grid_x, tf.cast(max_anchors[i]%3,tf.int32), 1],
          [grid_y, grid_x, tf.cast(max_anchors[i]%3,tf.int32), 2],
          [grid_y, grid_x, tf.cast(max_anchors[i]%3,tf.int32), 3],
          [grid_y, grid_x, tf.cast(max_anchors[i]%3,tf.int32), 4],
          [grid_y, grid_x, tf.cast(max_anchors[i]%3,tf.int32), 5]]
          updates = [cx,cy,w,h,1.,1.]
          output_1 = tf.tensor_scatter_nd_update(output_1,indices=indices, updates=updates)
        #   t_boxes.append([cx,cy,tf.cast(tf.math.abs(bbox[i,0] - bbox[i,1]),tf.float32),tf.cast(tf.math.abs(bbox[i,2] - bbox[i,3]),tf.float32)])
          t_indices = [[i]]
          t_updates = [[cx,cy,tf.cast(tf.math.abs(bbox[i,0] - bbox[i,1]),tf.float32),tf.cast(tf.math.abs(bbox[i,2] - bbox[i,3]),tf.float32)]]
          t_boxes = tf.tensor_scatter_nd_update(t_boxes,indices=t_indices,updates=t_updates)
        #   updates = tf.convert_to_tensor(updates)
        #   output_3[grid_y, grid_x, tf.cast(max_anchors[i]%3,tf.int32)]      += 0
        #   output_3[grid_y, grid_x, tf.cast(max_anchors[i]%3,tf.int32), 0:4] += [cx,cy,w,h]
        #   output_3[grid_y, grid_x, tf.cast(max_anchors[i]%3,tf.int32), 4  ] += 1.
        #   output_3[grid_y, grid_x, tf.cast(max_anchors[i]%3,tf.int32), 5+0] += 1.
#   t_boxes = tf.convert_to_tensor(t_boxes)
#   paddings = [[0,4-tf.shape(t_boxes)[0]],[0,0]]
  t_boxes = tf.reshape(t_boxes,[1,1,1,max_boxes,4])
  return output_1,output_2,output_3, t_boxes


def data_augmentation(elem):
  image = tf.image.decode_image(
      elem['image_raw'], expand_animations=False, dtype=tf.dtypes.float32)
  if(tf.shape(image)[2]==1):
    image = tf.image.grayscale_to_rgb(image)
  height = tf.cast(elem['height'], tf.float32)
  width = tf.cast(elem['width'], tf.float32)
  depth = elem['depth']
  input_sizes = tf.range(288,480,32,dtype=tf.float32)
  names = elem['name'].values # TODO: Taking from config file atm. 
  xmin = elem['xmin'].values
  xmax = elem['xmax'].values
  ymin = elem['ymin'].values
  ymax = elem['ymax'].values
  bbox = tf.transpose([xmin,xmax,ymin,ymax])
  image_1 , net_h, net_w ,bbox_1= aug_image(image,height,width, depth, bbox ,input_sizes)
  output_1,output_2,output_3, t_boxes = generate_train_output(bbox_1,net_h,net_w)
  
#   pad = tf.shape(t_boxes)
#   return bbox_1,output_1,output_2,output_3, net_h,net_w, width
  return image_1, output_1, output_2, output_3, t_boxes, net_h, net_w

