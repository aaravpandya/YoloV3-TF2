import tensorflow as tf
import numpy as np
import json

with open("config",'r') as fp:
    config = json.load(fp)



def aug_image_pred(image, height, width, depth,bbox,input_sizes, jitter=config['input_parameters']['jitter']):

  net_h = input_sizes[-1]
  net_w = input_sizes[-1]

  image = tf.image.resize(image, [net_h,net_w])

  sx = tf.cast(net_w, tf.float32)/width
  sy = tf.cast(net_h, tf.float32)/height

  bbox_x = tf.cast((tf.cast(bbox[:,0:2],tf.float32)*sx), tf.int32)
  bbox_x = tf.clip_by_value(bbox_x, clip_value_min = 0, clip_value_max = tf.cast(net_w,tf.int32))

  bbox_y = tf.cast((tf.cast(bbox[:,2:4],tf.float32)*sy), tf.int32)
  bbox_y = tf.clip_by_value(bbox_y, clip_value_min = 0, clip_value_max = tf.cast(net_h,tf.int32))

  bbox = tf.concat([bbox_x,bbox_y], axis=1,name="final_concat")

  return image, net_h, net_w, bbox

def data_augmentation_pred(elem):
  image = tf.image.decode_image(
      elem['image_raw'], expand_animations=False, dtype=tf.dtypes.float32)
  if(tf.shape(image)[2]==1):
    image = tf.image.grayscale_to_rgb(image)
  height = tf.cast(elem['height'], tf.float32)
  width = tf.cast(elem['width'], tf.float32)
  depth = elem['depth']
  input_sizes = tf.range(288,480,32,dtype=tf.float32)
  #   names = elem['name'].values TODO: Using from config atm.
  xmin = elem['xmin'].values
  xmax = elem['xmax'].values
  ymin = elem['ymin'].values
  ymax = elem['ymax'].values
  bbox = tf.transpose([xmin,xmax,ymin,ymax])
  image_1 , net_h, net_w ,bbox_1= aug_image_pred(image,height,width, depth, bbox ,input_sizes)

  return image_1, height,width, bbox_1, net_h, net_w,image

def get_boxes(y,height,width,net_h,net_w,anchors=config['generated_parameters']['anchors'], nms_threshold =  config["input_parameters"]["nms_threshold"]):
    max_grid_h, max_grid_w = 56, 56 #fix later
    cell_x = tf.cast(tf.reshape(tf.tile(tf.range(max_grid_w), [max_grid_h]), ( max_grid_h, max_grid_w, 1, 1)),tf.float32)
    cell_y = tf.transpose(cell_x, (1,0,2,3))
    cell_grid = tf.tile(tf.concat([cell_x,cell_y],-1), [1, 1, 3, 1])
    anchors = tf.convert_to_tensor(anchors,dtype=tf.float32, name="anchor_conversion")
    all_boxes = []
    for idx,y_pred in enumerate(y):
        y_pred = tf.squeeze(tf.reshape(y_pred, tf.concat([tf.shape(y_pred)[:3], tf.constant([3, -1])], axis=0)))
        grid_w, grid_h = tf.shape(y_pred)[0:2]
        x,y,w,h = y_pred[...,0],y_pred[...,1],y_pred[...,2],y_pred[...,3]
        objectness = tf.sigmoid(y_pred[...,4])
        class_pred = tf.nn.softmax(y_pred[...,5:])
        x = (tf.sigmoid(x) + cell_grid[:grid_h,:grid_w,:,0])/tf.cast(grid_w,tf.float32)
        y = (tf.sigmoid(y) + cell_grid[:grid_h,:grid_w,:,1])/tf.cast(grid_h,tf.float32)
        anchor_i = tf.cast(6-3*idx,tf.int32)
        anchor_j = tf.cast(9-3*idx,tf.int32)
        w = (tf.exp(w) * anchors[anchor_i:anchor_j,0])/net_w
        h = (tf.exp(h) * anchors[anchor_i:anchor_j,0])/net_h

        mask = tf.where(objectness > nms_threshold)
        x,y,w,h,classes, objectness = tf.gather_nd(x,mask),tf.gather_nd(y,mask),tf.gather_nd(w,mask),tf.gather_nd(h,mask),tf.gather_nd(class_pred,mask), tf.gather_nd(objectness,mask)

        x_min = x - (w/2)
        x_max = x + (w/2)
        y_min = y - (h/2)
        y_max = y + (h/2)

        if(tf.size(x) <1):
            continue

        if (net_w/width < net_h/height):
            new_w = net_w
            new_h = (height*net_w)/width
        else:
            new_h = net_w
            new_w = (width*net_h)/height
        x_offset, x_scale = (net_w - new_w)/2./net_w, new_w/net_w
        y_offset, y_scale = (net_h - new_h)/2./net_h, new_h/net_h
        x_min = tf.clip_by_value((x_min-x_offset)/x_scale * width,clip_value_min=0,clip_value_max=width)
        x_max = tf.clip_by_value((x_max-x_offset)/x_scale * width,clip_value_min=0,clip_value_max=width)
        y_min = tf.clip_by_value((y_min-y_offset)/y_scale * height,clip_value_min=0,clip_value_max=height)
        y_max = tf.clip_by_value((y_max-y_offset)/y_scale * height,clip_value_min=0,clip_value_max=height)
        if(tf.shape(classes)[0] == 1):
            classes = tf.squeeze(classes)
            classes = tf.expand_dims(classes, axis=0)
        else:
            classes = tf.squeeze(classes)
        boxes = tf.stack([x_min,x_max,y_min,y_max,classes,objectness],axis=1)
        all_boxes.append(boxes.numpy())
    return all_boxes

def compute_overlap(a, b):
    """
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes

    """
    area = (b[:, 1] - b[:, 0]) * (b[:, 3] - b[:, 2])
    iw = np.minimum(np.expand_dims(a[:, 1], axis=1), b[:, 1]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 2], 1), b[:, 2])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 1] - a[:, 0]) * (a[:, 3] - a[:, 2]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua

def compute_ap(recall, precision):
    """
    Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.

    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def compute_map(model ,test_data):
    test_data = test_data.map(data_augmentation_pred).batch(1).cache()
    scores = []
    num_anns = 0
    true_pos = []
    false_pos = []
    for feature in test_data:
        true_box = feature[3].numpy()
        true_box = np.reshape(true_box,(true_box.shape[1],true_box.shape[2]))
        y_ = model(feature[0])
        boxes_pred = get_boxes(y_, feature[1][0],feature[2][0] ,feature[4][0], feature[5][0])
        num_anns += true_box.shape[0]
        try:
            boxes_pred_coord = np.concatenate(boxes_pred)[:,0:4]
            objectness = np.concatenate(boxes_pred)[:,5]
            anns = []

            for idx,pred_box in enumerate(boxes_pred_coord):
                scores.append(objectness[idx])
                pred_box = np.expand_dims(pred_box,axis=0)

                overlap = compute_overlap(pred_box,true_box)
                args = np.argmax(overlap,axis=1)

                max_overlap = overlap[0,args[0]]
                if(max_overlap>0.5 and args[0] not in anns):
                    true_pos.append(1)
                    false_pos.append(0)
                    anns.append(args[0])
                else:
                    true_pos.append(0)
                    false_pos.append(1)
        except:
            print(None)
        # Uncomment to see boxes.
        # img = (tf.squeeze(feature[6]).numpy() * 255).astype(dtype=np.uint8)
        # for box in boxes_pred:
        #     cv.rectangle(img,(box[0],box[2]),(box[1],box[3]),(0,255,0),2)
        # cv2_imshow(img)

    scores = np.asarray(scores)
    indices         = np.argsort(-scores)
    false_pos = np.asarray(false_pos)
    true_pos = np.asarray(true_pos)
    false_positives = false_pos[indices]
    true_positives  = true_pos[indices]
    false_positives = np.cumsum(false_positives)
    true_positives  = np.cumsum(true_positives)
    recall    = true_positives / num_anns
    precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
    print(compute_ap(recall,precision))