from __future__ import print_function
import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
  # 텐서플로가 첫 번째 GPU만 사용하도록 제한
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
  except RuntimeError as e:
    # 프로그램 시작시에 접근 가능한 장치가 설정되어야만 합니다
    print(e)
    
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
    print(e)
    
  # 텐서플로가 첫 번째 GPU에 1GB 메모리만 할당하도록 제한
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
  except RuntimeError as e:
    # 프로그램 시작시에 가상 장치가 설정되어야만 합니다
    print(e)
    
batch_size = 128        # 배치 사이즈 128로 설정.
num_classes = 10        # 정답 라벨링 개수 ( 0 ~ 10 )
epochs = 3             # Epoch 반복 회숫 3회로 설정

# input image dimensions
img_rows, img_cols = 28, 28         # 이미지 가로 세로 배열.


# MNIST 데이터 불러오기
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))           # Input Layer : 32개의 필터를 가진다.

model.add(Conv2D(64, (3, 3), activation='relu'))                                                # 첫번째 hidden layer이며 64개의 필터를 가진다.
model.add(MaxPooling2D(pool_size=(2, 2)))                                                       # Pooling Layer를 의미한다.
model.add(Dropout(0.25))                                                                        # Dropout을 0.25로 하여 overfitting을 방지한다.
model.add(Flatten())                                                                            # 데이터를 1차원으로 바꿔주는 Layer. 아래의 fully_connected_layer에 연결해주기 위해 1차원으로 바꾸어준다.
model.add(Dense(128, activation='relu'))                                                        # Dense는 Fully Connected Layer를 의미한다.
model.add(Dropout(0.5))                                                                         # Dropout은 0.5로 하여 overfitting을 방지한다.

model.add(Dense(num_classes, activation='softmax'))                                             # Output Layer를 의미하며 activation은 softmax로 분류흘 한다.

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()                                                                                 # 모델의 요약을 보여준다.

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])