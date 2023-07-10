import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

img = image.load_img("C:\Users\TANISHQ\Pictures\Screenshots\Screenshot (2).png")

plt.imshow(img)

cv2.imread("C:\Users\TANISHQ\Pictures\Screenshots\Screenshot (2).png").shape

objects = ImageDataGenerator(rescale= 1/255)
validation = ImageDataGenerator(rescale= 1/255)

objects_dataset = objects.flow_from_directory('C:\Users\TANISHQ\Pictures\Screenshots\' ,
                                             target_size= (200, 200),
                                             batch_size = 3 ,
                                             clas_mode = 'binary')
validation_dataset = objects.flow_from_directory('C:\Users\TANISHQ\Pictures\Screenshots\' ,
                                             target_size= (200, 200),
                                             batch_size = 3 ,
                                             clas_mode = 'binary')



model = tf.keras.models.Sequential([tf.leras.layers.Conv2D(16),(3,3),activation = 'relu', input_shape =(200,200,3)
                                   ]
                                  )
