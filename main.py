import csv
import cv2
import numpy as np
import sklearn
import os

def all_images():
    center = []
    left = []
    right = []
    steers = []
    #header = 0
    with open('addData/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            """if header == 0:
                header += 1
                continue"""
            center.append(line[0].strip())
            left.append(line[1].strip())
            right.append(line[2].strip())
            steers.append(float(line[3].strip()))
    return (center, left, right, steers)
  
def combine_images(center, left, right, steers, correction = 0.2):
    Allimages = []
    Allimages.extend(center)
    Allimages.extend(left)
    Allimages.extend(right)
    Allsteers = []
    Allsteers.extend(steers)
    Allsteers.extend([i + correction for i in steers])
    Allsteers.extend([i - correction for i in steers])
    return (Allimages, Allsteers)
      
samples = []
center, left, right, steers = all_images()
images, steers = combine_images(center, left, right, steers)
samples = list(zip(images,steers))

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
train_samples, validation_samples = train_test_split(samples, test_size = 0.2)

def preprocess_image(img):
    # original shape: 160x320x3, input shape for neural net: 66x200x3
    new_img = img[50:140,:,:]
    new_img = cv2.GaussianBlur(new_img, (3,3), 0)
    new_img = cv2.resize(new_img,(200, 66), interpolation = cv2.INTER_AREA)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2YUV)
    return new_img

"""
img = cv2.imread("simData/data/" + samples[0][0])
img = preprocess_image(img)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

def generator(samples, batch_size=32, correction = 0.2):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for path_name, steer in batch_samples:
                try:
                    name = os.path.abspath(path_name)
                    #We add center images
                    image = cv2.imread(name)
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(rgb)
                    angles.append(steer)
                    
                    #We add flipped images
                    image_flipped = np.fliplr(rgb)
                    steer_flipped = -steer
                    images.append(image_flipped)
                    angles.append(steer_flipped)
                    
                    # trim image to only see section with road
                except: 
                    continue


            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size= 32 )
validation_generator = generator(validation_samples, batch_size = 32)

"""
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'simData/data/IMG/' + filename
    image = cv2.imread(current_path)
    image = cv2.cvtColor(Qimage, cv2.COLOR_BGR2GRAY)
    images.append(image)    
    measurement = float(line[3])
    measurements.append(measurement)
    
    image_flipped = np.fliplr(image)
    measurement_flipped = -measurement
    image_flipped = cv2.cvtColor(image_flipped, cv2.COLOR_BGR2GRAY)
    images.append(image_flipped)
    measurements.append(measurement_flipped)

X_train = np.array(images)
y_train = np.array(measurements)

"""

ch, row, col = 3, 160, 320

from keras.models import Sequential
from keras.layers import Flatten, Dense, Cropping2D,Conv2D, Lambda, Dropout

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (row,col,ch)))
model.add(Cropping2D(cropping=((50,20), (0,0))))
model.add(Conv2D(24,5,5, subsample = (2,2), activation = 'elu'))
model.add(Conv2D(36,5,5, subsample = (2,2), activation = 'elu'))
model.add(Conv2D(48,5,5, subsample = (2,2), activation = 'elu'))
model.add(Conv2D(64,3,3, activation = 'elu'))
model.add(Conv2D(64,3,3, activation = 'elu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
#batch_size = 32
results = model.fit_generator(train_generator, steps_per_epoch= len(train_samples),
validation_data=validation_generator, validation_steps=len(validation_samples), epochs=2, verbose = 1)

model.save('model.h5')
model.save_weights('model_W.h5')
print(results.history.keys())
print('Loss')
print(results.history['loss'])
print('Validation Loss')
print(results.history['val_loss'])