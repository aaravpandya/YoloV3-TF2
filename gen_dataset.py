from sklearn.cluster import KMeans
import tensorflow as tf
import os
import xml.etree.ElementTree as ET
import numpy as np
import json

def _bytes_feature(value):
	"""Returns a bytes_list from a string / byte."""
	if isinstance(value, type(tf.constant(0))):
		# BytesList won't unpack a string from an EagerTensor.
		value = value.numpy()
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_list_feature(value):
	"""Returns a bytes_list from a string / byte."""
	if isinstance(value, type(tf.constant(0))):
		# BytesList won't unpack a string from an EagerTensor.
		value = value.numpy()
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_list_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_list_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def image_example(image_string, width, height, depth, name, xmin, xmax, ymin, ymax):
  #   image_shape = tf.image.decode_jpeg(image_string).shape

  feature = {
	  'height': _int64_feature(height),
   	  'width': _int64_feature(width),
   	  'depth': _int64_feature(depth),
   	  'name': _bytes_list_feature(name),
   	  'xmin': _int64_list_feature(xmin),
   	  'xmax': _int64_list_feature(xmax),
   	  'ymin': _int64_list_feature(ymin),
   	  'ymax': _int64_list_feature(ymax),
   	  'image_raw': _bytes_feature(image_string),
  }

  return tf.train.Example(features=tf.train.Features(feature=feature))

def get_dataset(paths=[]):
	if len(paths) == 0:
		print("Empty path")
		return

	raw_image_dataset = tf.data.TFRecordDataset(paths)

	feature = {
		'height' : tf.io.FixedLenFeature([], tf.int64),
		'width':tf.io.FixedLenFeature([], tf.int64),
		'depth':tf.io.FixedLenFeature([], tf.int64),
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

def gen_records(config):
	ann_dir = config['input_parameters']['train_ann_dir']
	images_dir = config['input_parameters']['train_image_dir']
	shard_size = config['input_parameters']['shard_size']
	record = 'tf_record_{}.tfrecords'
	anns = os.listdir(ann_dir)
	num_anns = len(anns)
	num_shards = int((num_anns-(num_anns % shard_size))/shard_size) + 1
	max_objs_per_image = 0

	for shard_index in range(num_shards):
		with tf.io.TFRecordWriter(record.format(shard_index)) as writer:
			i = shard_index*shard_size
			if(num_anns - i > shard_size):
				j = i + 500
			else:
				j = num_anns-i
			for ann in anns[i:j]:
				tree = ET.parse(ann_dir + ann)
				root = tree.getroot()
				# print(ann)
				filename = root.find("filename").text
				size = root.find("size")
				width = int(size.find("width").text)
				height = int(size.find("height").text)
				depth = int(size.find("depth").text)
				image_raw = open(images_dir + filename, 'rb').read()
				name, xmin, xmax, ymin, ymax = list(), list(), list(), list(), list()
				ctr = 0
				for obj in root.findall("object"):
					ctr += 1
					if (ctr > max_objs_per_image):
						max_objs_per_image = ctr
					name.append(str.encode(obj.find("name").text))
					bndbox = obj.find("bndbox")
					xmin.append(int(bndbox.find("xmin").text))
					ymin.append(int(bndbox.find("ymin").text))
					xmax.append(int(bndbox.find("xmax").text))
					ymax.append(int(bndbox.find("ymax").text))
				tf_example = image_example(
					image_raw, width, height, depth, name, xmin, xmax, ymin, ymax)
				writer.write(tf_example.SerializeToString())
	paths = []
	for shard_index in range(num_shards):
		paths.append(record.format(shard_index))
	return paths, max_objs_per_image
# def gen_anchors():

def gen_labels(config):

	labels = []
	dataset = get_dataset(config['generated_parameters']['paths'])
	for features in dataset:
		names = np.unique(features['name'].values.numpy())
		labels.extend(names.tolist())
		# labels.append(features['name'].values
		# )
	labels = list(map(lambda s: s.decode(),list(set(labels))))
	return labels

def gen_anchors(config):
	dataset = get_dataset(config['generated_parameters']['paths'])
	anchors = []
	for features in dataset:
		xmin = features['xmin'].values.numpy()
		xmax = features['xmax'].values.numpy()
		height = xmax - xmin
		ymin = features['ymin'].values.numpy()
		ymax = features['ymax'].values.numpy()
		width = ymax - ymin

		for i in range(height.shape[0]):
			anchors.append([height[i],width[i]])
		# labels.extend(names.tolist())

	anchors = np.array(anchors)
	kmeans = KMeans(n_clusters=config['input_parameters']['num_anchors'], random_state=0).fit(anchors)
	centers = kmeans.cluster_centers_
	print(centers)

def generate():
	config_path = "config"
	with open(config_path, 'r') as fp:
		config = json.load(fp)
	if(config['input_parameters']['regen_records']):
		print("Generating TFRecords")
		paths, max_objs_per_image = gen_records(config)
		# config['generated_parameters'] = {}
		config['generated_parameters']['paths'] = paths
		config['generated_parameters']['max_objs_per_image'] = max_objs_per_image
		config['input_parameters']['regen_records'] = 0
		with open(config_path, 'w') as fp:
			json.dump(config, fp, indent=4)
	labels = gen_labels(config)
	config['generated_parameters']['labels'] = labels
	with open(config_path, 'w') as fp:
		json.dump(config, fp, indent=4)
	# gen_anchors(config)



generate()
