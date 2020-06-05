import tensorflow as tf
from tensorflow import keras
import json
with open("config",'r') as fp:
    config = json.load(fp)

# Need static grid. Fix later.
max_grid_h, max_grid_w = 56, 56
batch_size = 32
cell_x = tf.cast(tf.reshape(tf.tile(tf.range(max_grid_w), [
                max_grid_h]), (max_grid_h, max_grid_w, 1, 1)), tf.float32)
cell_y = tf.transpose(cell_x, (1, 0, 2, 3))
cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1), [1, 1, 3, 1])

ignore_thresh = config["input_parameters"]["ignore_thresh"]
xywh_scale = config["input_parameters"]["xywh_scale"]
obj_scale = config["input_parameters"]["obj_scale"]
noobj_scale = config["input_parameters"]["noobj_scale"]
class_scale = config["input_parameters"]["class_scale"]

def save_model_to_json(model, path="model.json"):
    json_config = model.to_json()
    d = json.loads(json_config)
    with open(path, 'w') as fp:
        json.dump(d,fp,indent=4)
    print("Model saved to " + path)

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
  dataset_size = len(list(parsed_image_dataset))
  train_test_split = 0.8
  test_split = 0.2
  train_take = int(train_test_split*dataset_size)
  test_take = int(test_split*dataset_size)
  train = parsed_image_dataset.take(train_take)
  test = parsed_image_dataset.skip(train_take).take(test_take)

  return train, test


def get_model(anchors=config['generated_parameters']['anchors'], labels=config['generated_parameters']['labels']):

    model_path = "/home/wisenet/Downloads/keras-yolo3/backend.h5"
    model_arch_path = "model.json"
    with open(model_arch_path, 'r') as fp:
        model_arch = json.load(fp)
    output_size = int((len(anchors)/3)*(5+len(labels))) # fix division by six, it assumes 3 anchors for each grid cell
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
    for i, layer in enumerate(model_trained.layers[0:-3]):
        model.layers[i].set_weights(layer.get_weights())
    del model_trained
    return model


@tf.function
def get_loss(y_pred_arr, y_true_arr, true_boxes, net_h, net_w, anchors=config['generated_parameters']['anchors']):
  anchors = [anchors[6:],anchors[3:6],anchors[0:3]]
  loss_total = tf.zeros([])
  for y_pred, y_true, layer_anchors in zip(y_pred_arr,y_true_arr,anchors):
    y_pred = tf.reshape(y_pred, tf.concat([tf.shape(y_pred)[:3], tf.constant([3, -1])], axis=0))
    object_mask = tf.expand_dims(y_true[..., 4], 4)
    grid_h      = tf.shape(y_true)[1]
    grid_w      = tf.shape(y_true)[2]
    grid_factor = tf.reshape(tf.cast([grid_w, grid_h], tf.float32), [1,1,1,1,2])
    net_factor  = tf.reshape(tf.cast([net_w, net_h], tf.float32), [1,1,1,1,2])
    pred_box_xy    = (cell_grid[:grid_h,:grid_w,:,:] + tf.sigmoid(y_pred[..., :2]))
    pred_box_wh    = y_pred[..., 2:4]
    pred_box_conf  = tf.expand_dims(tf.sigmoid(y_pred[..., 4]), 4)
    pred_box_class = y_pred[..., 5:]
    true_box_xy    = y_true[..., 0:2]
    true_box_wh    = y_true[..., 2:4]
    true_box_conf  = tf.expand_dims(y_true[..., 4], 4)
    true_box_class = tf.argmax(y_true[..., 5:], -1)
    conf_delta  = pred_box_conf - 0
    true_xy = true_boxes[..., 0:2] / grid_factor
    true_wh = true_boxes[..., 2:4] / net_factor

    true_wh_half = true_wh / 2.

    true_mins    = true_xy - true_wh_half
    true_maxes   = true_xy + true_wh_half
    pred_xy = tf.expand_dims(pred_box_xy / grid_factor, 4)
    pred_wh = tf.expand_dims(tf.exp(pred_box_wh) * layer_anchors / net_factor, 4)

    pred_wh_half = pred_wh / 2.
    pred_mins    = pred_xy - pred_wh_half
    pred_maxes   = pred_xy + pred_wh_half
    intersect_mins  = tf.maximum(pred_mins,  true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_areas = true_wh[..., 0] * true_wh[..., 1]
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores  = tf.truediv(intersect_areas, union_areas)
    best_ious   = tf.reduce_max(iou_scores, axis=4)
    conf_delta *= tf.expand_dims(tf.cast(best_ious < ignore_thresh,tf.float32), 4)
    wh_scale = tf.exp(true_box_wh) * layer_anchors / net_factor
    wh_scale = tf.expand_dims(2 - wh_scale[..., 0] * wh_scale[..., 1], axis=4) # the smaller the box, the bigger the scale
    xy_delta    = object_mask   * (pred_box_xy-true_box_xy) * wh_scale * xywh_scale
    wh_delta    = object_mask   * (pred_box_wh-true_box_wh) * wh_scale * xywh_scale
    conf_delta  = object_mask * (pred_box_conf-true_box_conf) * obj_scale + (1-object_mask) * conf_delta * noobj_scale
    class_delta = object_mask * \
                  tf.expand_dims(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class), 4) * \
                  class_scale
    loss_xy    = tf.reduce_sum(tf.square(xy_delta),       list(range(1,5)))
    loss_wh    = tf.reduce_sum(tf.square(wh_delta),       list(range(1,5)))
    loss_conf  = tf.reduce_sum(tf.square(conf_delta),     list(range(1,5)))
    loss_class = tf.reduce_sum(class_delta,               list(range(1,5)))
    loss = loss_xy + loss_wh + loss_conf + loss_class
    loss = tf.reduce_sum(loss)
    loss_total = loss_total + loss
  loss_total = tf.sqrt(loss_total)
  return loss_total
