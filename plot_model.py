from keras.models import load_model
from keras.utils import plot_model

# identical to the previous one
model = load_model('/home/wisenet/Downloads/keras-yolo3/backend.h5')
plot_model(model, to_file='model.png', show_shapes=True,show_layer_names=True) #use it to plot the graph in a png
print(model.summary())
i = input()
