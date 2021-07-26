from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from IPython.core.display import Image, display
import numpy as np
import os, random

neural_net = VGG16(weights='imagenet')
model = Model(inputs=neural_net.input, outputs=neural_net.get_layer('fc2').output)

img1_path = random.choice(os.listdir('/Path/to/folder/datasets/array/AllObjCollection (images)/cate16'))
img2_path = random.choice(os.listdir('/Path/to/folder/datasets/array/AllObjCollection (images)/cate18'))
img3_path = random.choice(os.listdir('/Path/to/folder/datasets/array/AllObjCollection (images)/cate19'))
img4_path = random.choice(os.listdir('/Path/to/folder/datasets/array/AllObjCollection (images)/cate20'))
img5_path = random.choice(os.listdir('/Path/to/folder/datasets/array/AllObjCollection (images)/cate34'))
img6_path = random.choice(os.listdir('/Path/to/folder/datasets/array/AllObjCollection (images)/cate78'))
options = ['16', '18', '19', '20', '34', '78']
choice = random.choice(options)
error_fixation_path = random.choice(os.listdir('/Path/to/folder//datasets/array/AllObjCollection (images)/cate' + choice))

img1_path = '/Path/to/folder/datasets/array/AllObjCollection (images)/cate16/' + img1_path
img2_path = '/Path/to/folder/datasets/array/AllObjCollection (images)/cate18/' + img2_path
img3_path = '/Path/to/folder/datasets/array/AllObjCollection (images)/cate19/' + img3_path
img4_path = '/Path/to/folder/datasets/array/AllObjCollection (images)/cate20/' + img4_path
img5_path = '/Path/to/folder/datasets/array/AllObjCollection (images)/cate34/' + img5_path
img6_path = '/Path/to/folder/datasets/array/AllObjCollection (images)/cate78/' + img6_path
error_fixation_path = '/Path/to/folder/datasets/array/AllObjCollection (images)/cate' + choice + '/' + error_fixation_path

img1 = image.load_img(img1_path, target_size=(224,224))
img2 = image.load_img(img2_path, target_size=(224,224))
img3 = image.load_img(img3_path, target_size=(224,224))
img4 = image.load_img(img4_path, target_size=(224,224))
img5 = image.load_img(img5_path, target_size=(224,224))
img6 = image.load_img(img6_path, target_size=(224,224))
error_fixation = image.load_img(error_fixation_path, target_size=(224, 224))

img1a = image.img_to_array(img1)
img2a = image.img_to_array(img2)
img3a = image.img_to_array(img3)
img4a = image.img_to_array(img4)
img5a = image.img_to_array(img5)
img6a = image.img_to_array(img6)
error_fixation_a = image.img_to_array(error_fixation)

img1a = np.expand_dims(img1a, axis=0)
img2a = np.expand_dims(img2a, axis=0)
img3a = np.expand_dims(img3a, axis=0)
img4a = np.expand_dims(img4a, axis=0)
img5a = np.expand_dims(img5a, axis=0)
img6a = np.expand_dims(img6a, axis=0)
error_fixation_a = np.expand_dims(error_fixation_a, axis=0)

img1a = preprocess_input(img1a)
img2a = preprocess_input(img2a)
img3a = preprocess_input(img3a)
img4a = preprocess_input(img4a)
img5a = preprocess_input(img5a)
img6a = preprocess_input(img6a)
error_fixation_a = preprocess_input(error_fixation_a)

features1 = model.predict(img1a)
features2 = model.predict(img2a)
features3 = model.predict(img3a)
features4 = model.predict(img4a)
features5 = model.predict(img5a)
features6 = model.predict(img6a)
features_ef = model.predict(error_fixation_a)

features1 = features1[0]
features2 = features2[0]
features3 = features3[0]
features4 = features4[0]
features5 = features5[0]
features6 = features6[0]
features_ef = features_ef[0]

dist1 = np.linalg.norm(features1 - features_ef)
dist2 = np.linalg.norm(features2 - features_ef)
dist3 = np.linalg.norm(features3 - features_ef)
dist4 = np.linalg.norm(features4 - features_ef)
dist5 = np.linalg.norm(features5 - features_ef)
dist6 = np.linalg.norm(features5 - features_ef)

distances = {}
distances['img1'] = dist1
distances['img2'] = dist2
distances['img3'] = dist3
distances['img4'] = dist4
distances['img5'] = dist5
distances['img6'] = dist6

paths ={}
paths['img1'] = img1_path
paths['img2'] = img2_path
paths['img3'] = img3_path
paths['img4'] = img4_path
paths['img5'] = img5_path
paths['img6'] = img6_path

sort_images = sorted(distances.items(), key=lambda x: x[1])

print('Error fixation: ')
display(Image(filename=error_fixation_path))
print('')
print('All potential targets: ')
display(Image(filename=img1_path))
display(Image(filename=img2_path))
display(Image(filename=img3_path))
display(Image(filename=img4_path))
display(Image(filename=img5_path))
display(Image(filename=img6_path))
print()

print('Error fixation: ')
display(Image(filename=error_fixation_path))
print('')
print('Targets ranked from most to least similar')
print('#1: Distance: ' + distances[sort_images[0][0]].astype('str'))
display(Image(filename=paths[sort_images[0][0]]))
print('')
print('#2: Distance: ' + distances[sort_images[1][0]].astype('str'))
display(Image(filename=paths[sort_images[1][0]]))
print('')
print('#3: Distance: ' + distances[sort_images[2][0]].astype('str'))
display(Image(filename=paths[sort_images[2][0]]))
print('')
print('#4: Distance: ' + distances[sort_images[3][0]].astype('str'))
display(Image(filename=paths[sort_images[3][0]]))
print('')
print('#5: Distance: ' + distances[sort_images[4][0]].astype('str'))
display(Image(filename=paths[sort_images[4][0]]))
print('')
print('#6: Distance: ' + distances[sort_images[5][0]].astype('str'))
display(Image(filename=paths[sort_images[5][0]]))
print('')
