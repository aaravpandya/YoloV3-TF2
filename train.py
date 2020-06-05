
import time
import numpy as np
import json
from utils import save_model_to_json, get_dataset, get_model, get_loss
from augmentation import data_augmentation
from pred_utils import compute_map
import tensorflow as tf
from tensorflow import keras

config_path = "config"
with open(config_path, 'r') as fp:
    config = json.load(fp)
TF_FORCE_GPU_ALLOW_GROWTH = True

nb_class = 1


@tf.function()
def loss(model, x, ds, training):

    y_ = model(x, training=training)

    return get_loss(y_, ds[1:4], ds[4], ds[5][0], ds[6][0])


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

def train(model , train_ds, writer):
  print("in train")
  optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4)
  loss_history = []
  min_loss = float("inf")

  for epoch in range(config['input_parameters']['epochs']):
      train_ = train_ds.cache().shuffle(200, reshuffle_each_iteration=True).map(
          data_augmentation).batch(config['input_parameters']['batch_size']).prefetch(tf.data.experimental.AUTOTUNE)
      start_time = time.time()
      loss_epoch = 0

      for features in train_:

          loss_value, grads = grad(model, features[0], features)

          loss_epoch += loss_value.numpy()

          optimizer.apply_gradients(zip(grads, model.trainable_variables))

      loss_history.append(loss_epoch)
      with writer.as_default():
          tf.summary.scalar('loss', data=loss_epoch, step=epoch)

      if(loss_epoch < min_loss):
          model.save(config["inpute_parameters"]["min_loss_model_save_dir"])
          min_loss = loss_epoch
      if(epoch % 100 == 0):
          model.save(config["inpute_parameters"]["model_save_dir"])
      print("loss: " + str(loss_epoch), "learning rate " +
            str(optimizer._decayed_lr(tf.float32).numpy()), "min loss" + str(min_loss))
      print("time iter:  " + str(time.time()-start_time))


def main():
  print("in main")
  model = get_model()
  train_ds, test_ds = get_dataset()
  writer = tf.summary.create_file_writer(config['input_parameters']['log_dir'])

  train(model, train_ds,writer)
  compute_map(model, test_ds)

if __name__ == "__main__":
    main()