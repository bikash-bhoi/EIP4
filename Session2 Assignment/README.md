**Name : Bikash Ranjan Bhoi**

**Reg. email Id : bhoi.bikash@gmail.com**

**Batch M6**

### Submission Details:
```
Total params: 9,680
Trainable params: 9,548
Non-trainable params: 132

Vacc : 0.9948 at 13th Epoch
```

### Model evaluation
```
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
def scheduler(epoch, lr):
  return round(lr * 1/(1 + 0.023 * epoch), 10)

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.2), metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=128, epochs=20, verbose=1, validation_data=(X_test, Y_test), callbacks=[LearningRateScheduler(scheduler, verbose=1)])
```

```
Train on 60000 samples, validate on 10000 samples
Epoch 1/20

Epoch 00001: LearningRateScheduler setting learning rate to 0.200000003.
60000/60000 [==============================] - 19s 317us/step - loss: 0.0713 - acc: 0.9777 - val_loss: 0.0309 - val_acc: 0.9897
Epoch 2/20

Epoch 00002: LearningRateScheduler setting learning rate to 0.1955034242.
60000/60000 [==============================] - 10s 175us/step - loss: 0.0607 - acc: 0.9814 - val_loss: 0.0449 - val_acc: 0.9856
Epoch 3/20

Epoch 00003: LearningRateScheduler setting learning rate to 0.1869057635.
60000/60000 [==============================] - 10s 173us/step - loss: 0.0533 - acc: 0.9832 - val_loss: 0.0342 - val_acc: 0.9891
Epoch 4/20

Epoch 00004: LearningRateScheduler setting learning rate to 0.1748416806.
60000/60000 [==============================] - 10s 173us/step - loss: 0.0500 - acc: 0.9844 - val_loss: 0.0384 - val_acc: 0.9883
Epoch 5/20

Epoch 00005: LearningRateScheduler setting learning rate to 0.1601114351.
60000/60000 [==============================] - 10s 173us/step - loss: 0.0488 - acc: 0.9850 - val_loss: 0.0283 - val_acc: 0.9906
Epoch 6/20

Epoch 00006: LearningRateScheduler setting learning rate to 0.143597706.
60000/60000 [==============================] - 10s 174us/step - loss: 0.0435 - acc: 0.9860 - val_loss: 0.0253 - val_acc: 0.9920
Epoch 7/20

Epoch 00007: LearningRateScheduler setting learning rate to 0.1261842769.
60000/60000 [==============================] - 10s 174us/step - loss: 0.0413 - acc: 0.9870 - val_loss: 0.0262 - val_acc: 0.9916
Epoch 8/20

Epoch 00008: LearningRateScheduler setting learning rate to 0.1086858482.
60000/60000 [==============================] - 10s 174us/step - loss: 0.0383 - acc: 0.9877 - val_loss: 0.0259 - val_acc: 0.9910
Epoch 9/20

Epoch 00009: LearningRateScheduler setting learning rate to 0.0917954823.
60000/60000 [==============================] - 10s 173us/step - loss: 0.0346 - acc: 0.9891 - val_loss: 0.0230 - val_acc: 0.9927
Epoch 10/20

Epoch 00010: LearningRateScheduler setting learning rate to 0.0760525946.
60000/60000 [==============================] - 10s 173us/step - loss: 0.0344 - acc: 0.9888 - val_loss: 0.0232 - val_acc: 0.9925
Epoch 11/20

Epoch 00011: LearningRateScheduler setting learning rate to 0.061831375.
60000/60000 [==============================] - 10s 173us/step - loss: 0.0324 - acc: 0.9897 - val_loss: 0.0226 - val_acc: 0.9930
Epoch 12/20

Epoch 00012: LearningRateScheduler setting learning rate to 0.049346667.
60000/60000 [==============================] - 10s 175us/step - loss: 0.0309 - acc: 0.9900 - val_loss: 0.0226 - val_acc: 0.9927
Epoch 13/20

Epoch 00013: LearningRateScheduler setting learning rate to 0.0386729364.
60000/60000 [==============================] - 10s 173us/step - loss: 0.0284 - acc: 0.9912 - val_loss: 0.0181 - val_acc: 0.9948
Epoch 14/20

Epoch 00014: LearningRateScheduler setting learning rate to 0.0297713127.
60000/60000 [==============================] - 10s 173us/step - loss: 0.0280 - acc: 0.9912 - val_loss: 0.0191 - val_acc: 0.9941
Epoch 15/20

Epoch 00015: LearningRateScheduler setting learning rate to 0.022519904.
60000/60000 [==============================] - 10s 171us/step - loss: 0.0265 - acc: 0.9919 - val_loss: 0.0197 - val_acc: 0.9941
Epoch 16/20

Epoch 00016: LearningRateScheduler setting learning rate to 0.0167434225.
60000/60000 [==============================] - 10s 170us/step - loss: 0.0256 - acc: 0.9917 - val_loss: 0.0193 - val_acc: 0.9942
Epoch 17/20

Epoch 00017: LearningRateScheduler setting learning rate to 0.0122393446.
60000/60000 [==============================] - 10s 170us/step - loss: 0.0242 - acc: 0.9923 - val_loss: 0.0187 - val_acc: 0.9938
Epoch 18/20

Epoch 00018: LearningRateScheduler setting learning rate to 0.0087989536.
60000/60000 [==============================] - 10s 170us/step - loss: 0.0258 - acc: 0.9919 - val_loss: 0.0181 - val_acc: 0.9944
Epoch 19/20

Epoch 00019: LearningRateScheduler setting learning rate to 0.0062227391.
60000/60000 [==============================] - 10s 173us/step - loss: 0.0234 - acc: 0.9924 - val_loss: 0.0190 - val_acc: 0.9939
Epoch 20/20

Epoch 00020: LearningRateScheduler setting learning rate to 0.004330368.
60000/60000 [==============================] - 10s 171us/step - loss: 0.0240 - acc: 0.9918 - val_loss: 0.0183 - val_acc: 0.9941
```
### Final Model(20th Epoch)
```
score = model.evaluate(X_test, Y_test, verbose=0)
print(score)
```
```
[0.01926184577558306, 0.9941]
```

### Strategy to tune the model
1. Reduce the number of Kernels to reduce number of parameters
2. Tune the Initial Learning late and hte scheduler value 
