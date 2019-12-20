import numpy as np # linear algebra
import os

from random import shuffle

list_paths = []
for subdir, dirs, files in os.walk('C:\\Users\\singl\\Downloads\\dataset\\dataset'):
    for file in files:
        #print os.path.join(subdir, file)
        filepath = subdir + os.sep + file
        list_paths.append(filepath)

list_train = [filepath for filepath in list_paths if "train" in filepath]
shuffle(list_train)
list_test = [filepath for filepath in list_paths if "test" in filepath]


index = [os.path.basename(filepath) for filepath in list_test]
list_classes = list(set([os.path.dirname(filepath).split(os.sep)[-1] for filepath in list_paths if "train" in filepath]))

ROWS=139
COLS=139
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input
from keras import optimizers
train_idg = ImageDataGenerator(vertical_flip=True,
                               horizontal_flip=True,
                               height_shift_range=0.1,
                               width_shift_range=0.1,
                               preprocessing_function=preprocess_input)
train_gen = train_idg.flow_from_directory(
    'C:\\Users\\singl\\Downloads\\dataset\\dataset\\train',
    target_size=(ROWS, COLS),
    batch_size = 50
)

test_idg = ImageDataGenerator(vertical_flip=True,
                               horizontal_flip=True,
                               height_shift_range=0.1,
                               width_shift_range=0.1,
                               preprocessing_function=preprocess_input)
test_gen = test_idg.flow_from_directory(
    'C:\\Users\\singl\\Downloads\\dataset\\dataset\\test',
    target_size=(ROWS, COLS),
    batch_size = 1
)





from keras.models import Sequential
from keras.models import Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import optimizers, losses, activations, models
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, GlobalAveragePooling1D, Concatenate, Conv1D
from keras import applications

input_shape = (ROWS, COLS)


base_model = applications.InceptionV3(weights='imagenet',
                                include_top=False,
                                      input_shape=(ROWS, COLS, 3))
base_model.trainable = False

add_model = Sequential()
add_model.add(base_model)

add_model.add(Convolution2D(4,(3,3)))
add_model.add(Flatten())
add_model.add(Dense(2,activation='softmax'))

model = add_model
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Nadam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004),
              metrics=['accuracy'])


file_path="C:\\python_practice\\AI project Data"

checkpoint = ModelCheckpoint(file_path, monitor='acc', verbose=1, save_best_only=True, mode='max')

early = EarlyStopping(monitor="acc", mode="max", patience=15)

callbacks_list = [checkpoint, early] #early

history = model.fit_generator(train_gen,
                              epochs=10,
                              shuffle=True,
                              verbose=True,
                              callbacks=callbacks_list)
test_gen.reset()
pred=model.predict_generator(test_gen,verbose=1)

predicted_class_indices=np.argmax(pred,axis=1)

labels = (train_gen.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

print (predictions)