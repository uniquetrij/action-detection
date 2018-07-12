# -*- coding: utf-8 -*-
"""Hello, Colaboratory

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/notebooks/welcome.ipynb
"""

!mkdir project
!ls
import os
os.chdir('project')
!pwd

!mkdir dataset
os.chdir('dataset')
!pwd

!wget http://www.nada.kth.se/cvap/actions/walking.zip
!wget http://www.nada.kth.se/cvap/actions/jogging.zip
!mkdir walking
!unzip walking.zip -d walking

!mkdir jogging
!unzip jogging.zip -d jogging

os.chdir("..")
!pwd

!pwd

import os
from zipfile import ZipFile
import random

import pickle
import csv
import cv2
import numpy as np

from keras.utils import np_utils, generic_utils
from keras.layers.convolutional import Conv3D
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Activation, Dropout
from keras.layers.convolutional import Convolution3D
from keras.layers.pooling import MaxPooling3D
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn import preprocessing

import matplotlib
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile,join 
from fnmatch import fnmatch
import pathlib

def split(video_paths, split_fractions):
    start_index = 0
    splits = []
    
    for i in range(len(split_fractions)-1):
        print(i)
        end_index=int(split_fractions[i]*len(video_paths))
        splits.append(video_paths[start_index:start_index + end_index])
        start_index+=end_index
    splits.append(video_paths[start_index:])
    return splits
    
def load_dataset(data_path, file_exts):
    pattern = '*['
    for ext in file_exts:
        pattern += ext+'|'
    pattern = pattern[:-1]+']'
    listing = []
    for path, subdirs, files in os.walk(data_path):
#         print(subdirs)
        for name in files:
            if fnmatch(name, pattern):
                listing.append(os.fspath(pathlib.PurePath(path,name)).replace('\\','/'))
    return listing
  
def get_label_from_path(path):
    return path.split('/')[-2]

def save_pickle(obj, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(filename):
    with open(filename, 'rb') as handle:
        return pickle.load(handle)
    
def gnerate_labels(data_path, file_exts, pickle_video_label, pickle_label_encoding):
    video_paths = load_dataset(data_path, file_exts)
    
    dict_label_endoded = {}
    dict_video_label = {}
     
    for v in video_paths:
        label = get_label_from_path(v)
        if not label in dict_label_endoded:
            dict_label_endoded[label] = len(dict_label_endoded)
        dict_video_label[v] = label
    
    for key in dict_label_endoded:
        value = dict_label_endoded[key]
        l  = np.zeros(len(dict_label_endoded))
        l[value] = 1
        dict_label_endoded[key] = l
    
    save_pickle([ [k,v] for k, v in dict_video_label.items()], pickle_video_label)
    save_pickle(dict_label_endoded, pickle_label_encoding)

def get_batches_fn(video_paths, label_encoded, batch_size, width, height, frame_count):
#     print(video_paths[1])
    """
    Create batches of training data
    :param batch_size: Batch Size
    :return: Batches of training data
    """
    counter = 100
    while(True):
      random.shuffle(video_paths)
      for batch_i in range(0, len(video_paths), batch_size):
          videos = []
          labels = []
          for video_file in video_paths[batch_i:batch_i+batch_size]:

              frames = []
              cap = cv2.VideoCapture(video_file[0])
              frames_per_second = cap.get(5)

              for k in range(frame_count):
                  counter-=1

                  ret, frame = cap.read()  

                  frame = cv2.resize(frame,(height, width), interpolation=cv2.INTER_AREA)
                  
                  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                  frames.append(gray)

                  if counter == 0:
                      plt.imshow(gray)

                  if cv2.waitKey(1) & 0xFF == ord('q'):
                      break
              cap.release()
              cv2.destroyAllWindows()

              inpt = np.array(frames)


              video = np.rollaxis(np.rollaxis(inpt,2,0),2,0)



              label = video_file[1]

              videos.append(video)
              labels.append(label_encoded[label])


          videos = np.array(videos)
          videos = np.reshape(videos,(len(videos),height, width, frame_count))


          train_set = np.zeros((len(videos), 1,height, width, frame_count))


          for h in range(len(videos)):
              train_set[h][0][:][:][:] = videos[h,:,:,:]

          train_set=np.rollaxis(train_set,1,5)            

          yield np.array(train_set), np.array(labels)
        
def build_model(class_count, height, width, frame_count):
    model = Sequential()

    model.add(Conv3D(32, kernel_size=(3,3,3), input_shape=(height, width, frame_count, 1), activation='relu'))
    model.add(MaxPooling3D(pool_size=(3, 3, 3)))
    model.add(Dropout(0.75))

    model.add(Conv3D(48, kernel_size=(3,3,3), activation='relu' ))
    model.add(Dropout(0.75))

    model.add(Conv3D(64, kernel_size=(5,5,1), activation='relu' ))
    model.add(Dropout(0.75))

    model.add(Conv3D(64, kernel_size=(5,5,1), activation='relu' ))
    model.add(Dropout(0.5))

    model.add(Conv3D(64, kernel_size=(7,7,1), activation='relu' ))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(128, init='normal', activation='relu'))
    model.add(Dropout(0.75))

    model.add(Dense(class_count, init='normal'))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    model.summary()

    return model;

gnerate_labels('dataset',['avi'],'list_video_label.pickle' ,'dict_label_encoded.pickle')

def train(g_train, g_val, model, batch_size, epochs, steps_per_epoch, validation_steps, model_name):
    hist = model.fit_generator(g_train,
                       validation_data=g_val,
                       epochs=epochs,
                              steps_per_epoch=steps_per_epoch,
                              validation_steps=validation_steps)

    model_json = model.to_json()
    with open(model_name+".json", "w") as json_file:
        json_file.write(model_json)
        # serialize weights to HDF5
    model.save_weights(model_name+".h5")

def evaluate_model(g_test, test_steps, model_name):
  # load json and create model
    json_file = open(model_name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_name+".h5")
    print("Loaded model from disk")
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam')

    return loaded_model.evaluate_generator(self, g_test, steps=test_steps, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)


def run(data_path, file_exts, class_count, batch_size, width, height, frame_count, split_per, epochs):
    labeled_list = load_pickle('list_video_label.pickle')
    labels_encoded = load_pickle('dict_label_encoded.pickle')
    random.shuffle(labeled_list)
    X_train, X_val, X_test =  split(labeled_list, split_per)

    g_train = get_batches_fn(X_train,labels_encoded,batch_size, width, height, frame_count )
    g_val = get_batches_fn(X_val,labels_encoded,batch_size, width, height, frame_count)
    g_test = get_batches_fn(X_test,labels_encoded,batch_size, width, height, frame_count)

    model = build_model(class_count, height, width, frame_count)

    steps_per_epoch = len(X_train)//batch_size
    validation_steps = len(X_val)//batch_size
    test_steps = len(X_test)//batch_size
    train(g_train, g_val, model, batch_size, epochs, steps_per_epoch, validation_steps, "model")

#     dataset, exts, height, width, split_fraction = 'dataset', ['avi'], 160, 120, 15, [0.2, 0.1, 0.1]
    score = evaluate_model(g_test,test_steps, "model")
    print(score)

run('dataset',['avi'], 2, 10, 160, 120, 15, [0.7, 0.2, 0.1], 100)







"""## Welcome to Colaboratory!

Colaboratory is a Google research project created to help disseminate machine learning education and research. It's a Jupyter notebook environment that requires no setup to use and runs entirely in the cloud.

Colaboratory notebooks are stored in [Google Drive](https://drive.google.com) and can be shared just as you would with Google Docs or Sheets. Colaboratory is free to use.

For more information, see our [FAQ](https://research.google.com/colaboratory/faq.html).

## Local runtime support

Colab also supports connecting to a Jupyter runtime on your local machine. For more information, see our [documentation](https://research.google.com/colaboratory/local-runtimes.html).

## Python 3

Colaboratory supports both Python2 and Python3 for code execution. 

* When creating a new notebook, you'll have the choice between Python 2 and Python 3.
* You can also change the language associated with a notebook; this information will be written into the `.ipynb` file itself, and thus will be preserved for future sessions.
"""

import sys
print('Hello, Colaboratory from Python {}!'.format(sys.version_info[0]))

"""## TensorFlow execution

Colaboratory allows you to execute TensorFlow code in your browser with a single click. The example below adds two matrices.

$\begin{bmatrix}
  1. & 1. & 1. \\
  1. & 1. & 1. \\
\end{bmatrix} +
\begin{bmatrix}
  1. & 2. & 3. \\
  4. & 5. & 6. \\
\end{bmatrix} =
\begin{bmatrix}
  2. & 3. & 4. \\
  5. & 6. & 7. \\
\end{bmatrix}$
"""

import tensorflow as tf
import numpy as np

with tf.Session():
  input1 = tf.constant(1.0, shape=[2, 3])
  input2 = tf.constant(np.reshape(np.arange(1.0, 7.0, dtype=np.float32), (2, 3)))
  output = tf.add(input1, input2)
  result = output.eval()

result

"""## Visualization

Colaboratory includes widely used libraries like [matplotlib](https://matplotlib.org/), simplifying visualization.
"""

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(20)
y = [x_i + np.random.randn(1) for x_i in x]
a, b = np.polyfit(x, y, 1)
_ = plt.plot(x, y, 'o', np.arange(20), a*np.arange(20)+b, '-')

"""Want to use a new library?  `pip install` it. For recipes to import commonly used libraries, refer to the [importing libraries example notebook](/notebooks/snippets/importing_libraries.ipynb)"""

# Only needs to be run once at the top of the notebook.
!pip install -q matplotlib-venn

# Now the newly-installed library can be used anywhere else in the notebook.
from matplotlib_venn import venn2
_ = venn2(subsets = (3, 2, 1))

"""# Forms

Forms can be used to parameterize code. See the [forms example notebook](/notebooks/forms.ipynb) for more details.
"""

#@title Examples

text = 'value' #@param
date_input = '2018-03-22' #@param {type:"date"}
number_slider = 0 #@param {type:"slider", min:-1, max:1, step:0.1}
dropdown = '1st option' #@param ["1st option", "2nd option", "3rd option"]

"""# For more information:
- [Overview of Colaboratory](/notebooks/basic_features_overview.ipynb)
- [Importing libraries and installing dependencies](/notebooks/snippets/importing_libraries.ipynb)
- [Markdown guide](/notebooks/markdown_guide.ipynb)
- [Charts](/notebooks/charts.ipynb)
- [Widgets](/notebooks/widgets.ipynb)
- [Loading and saving data: local files, Drive, Sheets, Google Cloud Storage](/notebooks/io.ipynb)
- [Example Google Cloud BigQuery notebook](/notebooks/bigquery.ipynb)
- [TensorFlow with GPU](/notebooks/gpu.ipynb)
- [Forms](/notebooks/forms.ipynb)
"""