**Name : Bikash Ranjan Bhoi**

**Reg. email Id : bhoi.bikash@gmail.com**

**Batch M6**


### Submission Details:
```
Total params: 66,309
Trainable params: 64,965
Non-trainable params: 1,344

Vacc : 0.8481 at 42nd Epoch
```

### Final Validation accuracy for Base Network:

```
Epoch 50/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3270 - acc: 0.8897 - val_loss: 0.5987 - val_acc: 0.8245

Accuracy on test data is: 82.45

```

### Model Definition:
```
# Define the New model beat Vacc 0.8270
model1 = Sequential()

model1.add(SeparableConv2D(32, 3, depth_multiplier=1, activation='relu', input_shape=(32, 32, 3)))  #30x30 3x3
model1.add(BatchNormalization())
model1.add(Dropout(0.2))

model1.add(SeparableConv2D(64, 3, depth_multiplier=1, activation='relu' ))  #28x28 5x5
model1.add(BatchNormalization())
model1.add(Dropout(0.2))

model1.add(SeparableConv2D(128, 3, depth_multiplier=1, activation='relu' ))  #26x26 7x7
model1.add(BatchNormalization())
model1.add(Dropout(0.2))

model1.add(Conv2D(32, 1, activation='relu'  ))  #26x26 7x7
model1.add(MaxPooling2D(pool_size=(2, 2))) #13x13 8x8

model1.add(SeparableConv2D(64, 3, depth_multiplier=1, activation='relu' ))  #11x11 12x12
model1.add(BatchNormalization())
model1.add(Dropout(0.1))

model1.add(SeparableConv2D(128, 3, depth_multiplier=1, activation='relu' ))  #9x9 16x16
model1.add(BatchNormalization())
model1.add(Dropout(0.1))

model1.add(SeparableConv2D(256, 3, depth_multiplier=1, activation='relu' ))  #7x7 20x20
model1.add(BatchNormalization())
model1.add(Dropout(0.1))

model1.add(Conv2D(num_classes, 1)) #7x7x10 20x20

model1.add(GlobalAveragePooling2D())  #10 32x32 
#model1.add(Flatten())
model1.add(Activation('softmax'))

model1.summary()

#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### My 50 epoch logs
```
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
def scheduler(epoch, lr):
  return round(lr * 1/(1 + 0.0051 * epoch), 10)

model1.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.019), metrics=['accuracy'])

# set up image augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    horizontal_flip=False,
    width_shift_range=0.11,
    height_shift_range=0.11)
datagen.fit(train_features)

# train the model
start = time.time()
# Train the model
model_info = model1.fit_generator(datagen.flow(train_features, train_labels, batch_size = 128),
                                 samples_per_epoch = train_features.shape[0], nb_epoch = 50, 
                                 validation_data = (test_features, test_labels), verbose=1,
                                 callbacks=[LearningRateScheduler(scheduler, verbose=1)])
end = time.time()
print ("Model took %0.2f seconds to train"%(end - start))
# plot model history
plot_model_history(model_info)
# compute test accuracy
print ("Accuracy on test data is: %0.2f"%accuracy(test_features, test_labels, model1))
```

```
Epoch 1/50

Epoch 00001: LearningRateScheduler setting learning rate to 0.0189999994.
390/390 [==============================] - 31s 79ms/step - loss: 0.6017 - acc: 0.7903 - val_loss: 0.6244 - val_acc: 0.7891
Epoch 2/50

Epoch 00002: LearningRateScheduler setting learning rate to 0.0189035911.
390/390 [==============================] - 27s 70ms/step - loss: 0.6031 - acc: 0.7884 - val_loss: 0.6086 - val_acc: 0.7997
Epoch 3/50

Epoch 00003: LearningRateScheduler setting learning rate to 0.018712721.
390/390 [==============================] - 27s 70ms/step - loss: 0.5993 - acc: 0.7894 - val_loss: 0.7033 - val_acc: 0.7786
Epoch 4/50

Epoch 00004: LearningRateScheduler setting learning rate to 0.0184307316.
390/390 [==============================] - 27s 69ms/step - loss: 0.6002 - acc: 0.7883 - val_loss: 0.6875 - val_acc: 0.7761
Epoch 5/50

Epoch 00005: LearningRateScheduler setting learning rate to 0.018062262.
390/390 [==============================] - 27s 69ms/step - loss: 0.5904 - acc: 0.7928 - val_loss: 0.7124 - val_acc: 0.7808
Epoch 6/50

Epoch 00006: LearningRateScheduler setting learning rate to 0.0176131271.
390/390 [==============================] - 27s 69ms/step - loss: 0.5915 - acc: 0.7931 - val_loss: 0.5946 - val_acc: 0.8018
Epoch 7/50

Epoch 00007: LearningRateScheduler setting learning rate to 0.0170901687.
390/390 [==============================] - 27s 69ms/step - loss: 0.5855 - acc: 0.7935 - val_loss: 0.5930 - val_acc: 0.7978
Epoch 8/50

Epoch 00008: LearningRateScheduler setting learning rate to 0.0165010793.
390/390 [==============================] - 27s 69ms/step - loss: 0.5740 - acc: 0.7992 - val_loss: 0.6652 - val_acc: 0.7913
Epoch 9/50

Epoch 00009: LearningRateScheduler setting learning rate to 0.015854226.
390/390 [==============================] - 27s 69ms/step - loss: 0.5689 - acc: 0.8018 - val_loss: 0.5983 - val_acc: 0.8015
Epoch 10/50

Epoch 00010: LearningRateScheduler setting learning rate to 0.0151584534.
390/390 [==============================] - 27s 69ms/step - loss: 0.5577 - acc: 0.8042 - val_loss: 0.5923 - val_acc: 0.8022
Epoch 11/50

Epoch 00011: LearningRateScheduler setting learning rate to 0.0144228858.
390/390 [==============================] - 27s 69ms/step - loss: 0.5544 - acc: 0.8056 - val_loss: 0.5973 - val_acc: 0.8009
Epoch 12/50

Epoch 00012: LearningRateScheduler setting learning rate to 0.0136567428.
390/390 [==============================] - 27s 70ms/step - loss: 0.5458 - acc: 0.8082 - val_loss: 0.6054 - val_acc: 0.8010
Epoch 13/50

Epoch 00013: LearningRateScheduler setting learning rate to 0.0128691508.
390/390 [==============================] - 27s 69ms/step - loss: 0.5310 - acc: 0.8135 - val_loss: 0.5568 - val_acc: 0.8121
Epoch 14/50

Epoch 00014: LearningRateScheduler setting learning rate to 0.0120689772.
390/390 [==============================] - 27s 69ms/step - loss: 0.5217 - acc: 0.8174 - val_loss: 0.5520 - val_acc: 0.8209
Epoch 15/50

Epoch 00015: LearningRateScheduler setting learning rate to 0.0112646795.
390/390 [==============================] - 27s 69ms/step - loss: 0.5089 - acc: 0.8219 - val_loss: 0.5476 - val_acc: 0.8208
Epoch 16/50

Epoch 00016: LearningRateScheduler setting learning rate to 0.0104641709.
390/390 [==============================] - 27s 69ms/step - loss: 0.5074 - acc: 0.8239 - val_loss: 0.5632 - val_acc: 0.8112
Epoch 17/50

Epoch 00017: LearningRateScheduler setting learning rate to 0.0096747143.
390/390 [==============================] - 27s 69ms/step - loss: 0.4904 - acc: 0.8277 - val_loss: 0.5426 - val_acc: 0.8184
Epoch 18/50

Epoch 00018: LearningRateScheduler setting learning rate to 0.0089028379.
390/390 [==============================] - 27s 69ms/step - loss: 0.4925 - acc: 0.8267 - val_loss: 0.5184 - val_acc: 0.8234
Epoch 19/50

Epoch 00019: LearningRateScheduler setting learning rate to 0.0081542751.
390/390 [==============================] - 27s 70ms/step - loss: 0.4825 - acc: 0.8294 - val_loss: 0.5447 - val_acc: 0.8278
Epoch 20/50

Epoch 00020: LearningRateScheduler setting learning rate to 0.0074339273.
390/390 [==============================] - 27s 69ms/step - loss: 0.4725 - acc: 0.8332 - val_loss: 0.5663 - val_acc: 0.8157
Epoch 21/50

Epoch 00021: LearningRateScheduler setting learning rate to 0.0067458504.
390/390 [==============================] - 27s 69ms/step - loss: 0.4634 - acc: 0.8377 - val_loss: 0.5587 - val_acc: 0.8143
Epoch 22/50

Epoch 00022: LearningRateScheduler setting learning rate to 0.0060932619.
390/390 [==============================] - 27s 69ms/step - loss: 0.4556 - acc: 0.8405 - val_loss: 0.5178 - val_acc: 0.8340
Epoch 23/50

Epoch 00023: LearningRateScheduler setting learning rate to 0.0054785666.
390/390 [==============================] - 27s 69ms/step - loss: 0.4503 - acc: 0.8433 - val_loss: 0.5172 - val_acc: 0.8295
Epoch 24/50

Epoch 00024: LearningRateScheduler setting learning rate to 0.0049033979.
390/390 [==============================] - 27s 69ms/step - loss: 0.4401 - acc: 0.8449 - val_loss: 0.4994 - val_acc: 0.8372
Epoch 25/50

Epoch 00025: LearningRateScheduler setting learning rate to 0.0043686725.
390/390 [==============================] - 27s 69ms/step - loss: 0.4346 - acc: 0.8456 - val_loss: 0.4984 - val_acc: 0.8383
Epoch 26/50

Epoch 00026: LearningRateScheduler setting learning rate to 0.0038746542.
390/390 [==============================] - 27s 70ms/step - loss: 0.4343 - acc: 0.8477 - val_loss: 0.4951 - val_acc: 0.8380
Epoch 27/50

Epoch 00027: LearningRateScheduler setting learning rate to 0.0034210261.
390/390 [==============================] - 28s 71ms/step - loss: 0.4266 - acc: 0.8490 - val_loss: 0.4880 - val_acc: 0.8418
Epoch 28/50

Epoch 00028: LearningRateScheduler setting learning rate to 0.0030069667.
390/390 [==============================] - 28s 71ms/step - loss: 0.4207 - acc: 0.8522 - val_loss: 0.4863 - val_acc: 0.8416
Epoch 29/50

Epoch 00029: LearningRateScheduler setting learning rate to 0.0026312275.
390/390 [==============================] - 27s 70ms/step - loss: 0.4180 - acc: 0.8540 - val_loss: 0.5053 - val_acc: 0.8350
Epoch 30/50

Epoch 00030: LearningRateScheduler setting learning rate to 0.0022922097.
390/390 [==============================] - 28s 71ms/step - loss: 0.4090 - acc: 0.8568 - val_loss: 0.5032 - val_acc: 0.8358
Epoch 31/50

Epoch 00031: LearningRateScheduler setting learning rate to 0.0019880397.
390/390 [==============================] - 28s 71ms/step - loss: 0.4102 - acc: 0.8541 - val_loss: 0.4899 - val_acc: 0.8415
Epoch 32/50

Epoch 00032: LearningRateScheduler setting learning rate to 0.001716639.
390/390 [==============================] - 27s 70ms/step - loss: 0.4102 - acc: 0.8549 - val_loss: 0.4882 - val_acc: 0.8440
Epoch 33/50

Epoch 00033: LearningRateScheduler setting learning rate to 0.0014757901.
390/390 [==============================] - 27s 69ms/step - loss: 0.4018 - acc: 0.8585 - val_loss: 0.4847 - val_acc: 0.8443
Epoch 34/50

Epoch 00034: LearningRateScheduler setting learning rate to 0.0012631944.
390/390 [==============================] - 27s 69ms/step - loss: 0.4005 - acc: 0.8597 - val_loss: 0.4801 - val_acc: 0.8443
Epoch 35/50

Epoch 00035: LearningRateScheduler setting learning rate to 0.0010765249.
390/390 [==============================] - 27s 69ms/step - loss: 0.3955 - acc: 0.8597 - val_loss: 0.4780 - val_acc: 0.8446
Epoch 36/50

Epoch 00036: LearningRateScheduler setting learning rate to 0.0009134704.
390/390 [==============================] - 27s 69ms/step - loss: 0.3984 - acc: 0.8588 - val_loss: 0.4795 - val_acc: 0.8455
Epoch 37/50

Epoch 00037: LearningRateScheduler setting learning rate to 0.0007717729.
390/390 [==============================] - 27s 70ms/step - loss: 0.3916 - acc: 0.8614 - val_loss: 0.4809 - val_acc: 0.8447
Epoch 38/50

Epoch 00038: LearningRateScheduler setting learning rate to 0.0006492579.
390/390 [==============================] - 27s 69ms/step - loss: 0.3900 - acc: 0.8646 - val_loss: 0.4836 - val_acc: 0.8443
Epoch 39/50

Epoch 00039: LearningRateScheduler setting learning rate to 0.0005438582.
390/390 [==============================] - 27s 70ms/step - loss: 0.3897 - acc: 0.8622 - val_loss: 0.4813 - val_acc: 0.8454
Epoch 40/50

Epoch 00040: LearningRateScheduler setting learning rate to 0.000453631.
390/390 [==============================] - 27s 69ms/step - loss: 0.3893 - acc: 0.8627 - val_loss: 0.4814 - val_acc: 0.8446
Epoch 41/50

Epoch 00041: LearningRateScheduler setting learning rate to 0.0003767699.
390/390 [==============================] - 27s 69ms/step - loss: 0.3841 - acc: 0.8655 - val_loss: 0.4853 - val_acc: 0.8435
Epoch 42/50

Epoch 00042: LearningRateScheduler setting learning rate to 0.0003116119.
390/390 [==============================] - 27s 70ms/step - loss: 0.3882 - acc: 0.8633 - val_loss: 0.4764 - val_acc: 0.8481
Epoch 43/50

Epoch 00043: LearningRateScheduler setting learning rate to 0.0002566397.
390/390 [==============================] - 27s 69ms/step - loss: 0.3903 - acc: 0.8622 - val_loss: 0.4797 - val_acc: 0.8463
Epoch 44/50

Epoch 00044: LearningRateScheduler setting learning rate to 0.0002104812.
390/390 [==============================] - 27s 69ms/step - loss: 0.3878 - acc: 0.8628 - val_loss: 0.4795 - val_acc: 0.8456
Epoch 45/50

Epoch 00045: LearningRateScheduler setting learning rate to 0.0001719056.
390/390 [==============================] - 27s 70ms/step - loss: 0.3847 - acc: 0.8640 - val_loss: 0.4801 - val_acc: 0.8466
Epoch 46/50

Epoch 00046: LearningRateScheduler setting learning rate to 0.0001398175.
390/390 [==============================] - 27s 69ms/step - loss: 0.3901 - acc: 0.8639 - val_loss: 0.4802 - val_acc: 0.8461
Epoch 47/50

Epoch 00047: LearningRateScheduler setting learning rate to 0.0001132492.
390/390 [==============================] - 27s 69ms/step - loss: 0.3860 - acc: 0.8637 - val_loss: 0.4795 - val_acc: 0.8457
Epoch 48/50

Epoch 00048: LearningRateScheduler setting learning rate to 9.13521e-05.
390/390 [==============================] - 27s 69ms/step - loss: 0.3807 - acc: 0.8652 - val_loss: 0.4823 - val_acc: 0.8451
Epoch 49/50

Epoch 00049: LearningRateScheduler setting learning rate to 7.3387e-05.
390/390 [==============================] - 27s 69ms/step - loss: 0.3902 - acc: 0.8623 - val_loss: 0.4791 - val_acc: 0.8458
Epoch 50/50

Epoch 00050: LearningRateScheduler setting learning rate to 5.87143e-05.
390/390 [==============================] - 27s 69ms/step - loss: 0.3850 - acc: 0.8651 - val_loss: 0.4797 - val_acc: 0.8447
Model took 1359.75 seconds to train

Accuracy on test data is: 84.47

```
![Loss and Accuracy](https://github.com/bikash-bhoi/EIP4/blob/master/Session3%20Assignment/acc%20and%20loss.png)


