**Name : Bikash Ranjan Bhoi**

**Reg. email Id : bhoi.bikash@gmail.com**

**Batch M6**

### Submission Details:
```
Total params: 9,754
Trainable params: 9,622
Non-trainable params: 132

Vacc : 0.9950 at 13th Epoch
```

### Model evaluation
```
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
def scheduler(epoch, lr):
  return round(lr * 1/(1 + 0.039 * epoch), 10)

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.2), metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=128, epochs=20, verbose=1, validation_data=(X_test, Y_test), callbacks=[LearningRateScheduler(scheduler, verbose=1)])
```

```
Train on 60000 samples, validate on 10000 samples
Epoch 1/20

Epoch 00001: LearningRateScheduler setting learning rate to 0.200000003.
60000/60000 [==============================] - 13s 225us/step - loss: 0.0995 - acc: 0.9693 - val_loss: 0.0828 - val_acc: 0.9808
Epoch 2/20

Epoch 00002: LearningRateScheduler setting learning rate to 0.1924927844.
60000/60000 [==============================] - 11s 188us/step - loss: 0.0649 - acc: 0.9795 - val_loss: 0.0360 - val_acc: 0.9887
Epoch 3/20

Epoch 00003: LearningRateScheduler setting learning rate to 0.1785647338.
60000/60000 [==============================] - 11s 188us/step - loss: 0.0528 - acc: 0.9833 - val_loss: 0.0452 - val_acc: 0.9846
Epoch 4/20

Epoch 00004: LearningRateScheduler setting learning rate to 0.1598609913.
60000/60000 [==============================] - 11s 185us/step - loss: 0.0493 - acc: 0.9845 - val_loss: 0.0390 - val_acc: 0.9879
Epoch 5/20

Epoch 00005: LearningRateScheduler setting learning rate to 0.1382880609.
60000/60000 [==============================] - 11s 185us/step - loss: 0.0419 - acc: 0.9866 - val_loss: 0.0290 - val_acc: 0.9906
Epoch 6/20

Epoch 00006: LearningRateScheduler setting learning rate to 0.1157222308.
60000/60000 [==============================] - 11s 185us/step - loss: 0.0395 - acc: 0.9880 - val_loss: 0.0275 - val_acc: 0.9909
Epoch 7/20

Epoch 00007: LearningRateScheduler setting learning rate to 0.0937781455.
60000/60000 [==============================] - 11s 184us/step - loss: 0.0353 - acc: 0.9884 - val_loss: 0.0212 - val_acc: 0.9927
Epoch 8/20

Epoch 00008: LearningRateScheduler setting learning rate to 0.073667045.
60000/60000 [==============================] - 11s 185us/step - loss: 0.0318 - acc: 0.9902 - val_loss: 0.0267 - val_acc: 0.9914
Epoch 9/20

Epoch 00009: LearningRateScheduler setting learning rate to 0.05614866.
60000/60000 [==============================] - 11s 188us/step - loss: 0.0273 - acc: 0.9912 - val_loss: 0.0213 - val_acc: 0.9931
Epoch 10/20

Epoch 00010: LearningRateScheduler setting learning rate to 0.0415608138.
60000/60000 [==============================] - 11s 184us/step - loss: 0.0249 - acc: 0.9917 - val_loss: 0.0203 - val_acc: 0.9941
Epoch 11/20

Epoch 00011: LearningRateScheduler setting learning rate to 0.029899866.
60000/60000 [==============================] - 11s 183us/step - loss: 0.0235 - acc: 0.9926 - val_loss: 0.0200 - val_acc: 0.9935
Epoch 12/20

Epoch 00012: LearningRateScheduler setting learning rate to 0.0209236287.
60000/60000 [==============================] - 11s 184us/step - loss: 0.0206 - acc: 0.9935 - val_loss: 0.0220 - val_acc: 0.9936
Epoch 13/20

Epoch 00013: LearningRateScheduler setting learning rate to 0.0142531535.
60000/60000 [==============================] - 11s 183us/step - loss: 0.0212 - acc: 0.9932 - val_loss: 0.0186 - val_acc: 0.9950
Epoch 14/20

Epoch 00014: LearningRateScheduler setting learning rate to 0.0094579651.
60000/60000 [==============================] - 11s 182us/step - loss: 0.0192 - acc: 0.9937 - val_loss: 0.0179 - val_acc: 0.9947
Epoch 15/20

Epoch 00015: LearningRateScheduler setting learning rate to 0.0061177008.
60000/60000 [==============================] - 11s 182us/step - loss: 0.0192 - acc: 0.9938 - val_loss: 0.0167 - val_acc: 0.9946
Epoch 16/20

Epoch 00016: LearningRateScheduler setting learning rate to 0.003859748.
60000/60000 [==============================] - 11s 181us/step - loss: 0.0187 - acc: 0.9938 - val_loss: 0.0174 - val_acc: 0.9946
Epoch 17/20

Epoch 00017: LearningRateScheduler setting learning rate to 0.0023766921.
60000/60000 [==============================] - 11s 180us/step - loss: 0.0194 - acc: 0.9939 - val_loss: 0.0172 - val_acc: 0.9947
Epoch 18/20

Epoch 00018: LearningRateScheduler setting learning rate to 0.0014291594.
60000/60000 [==============================] - 11s 180us/step - loss: 0.0183 - acc: 0.9943 - val_loss: 0.0172 - val_acc: 0.9945
Epoch 19/20

Epoch 00019: LearningRateScheduler setting learning rate to 0.0008396941.
60000/60000 [==============================] - 11s 179us/step - loss: 0.0182 - acc: 0.9941 - val_loss: 0.0173 - val_acc: 0.9948
Epoch 20/20

Epoch 00020: LearningRateScheduler setting learning rate to 0.0004823056.
60000/60000 [==============================] - 11s 179us/step - loss: 0.0174 - acc: 0.9945 - val_loss: 0.0172 - val_acc: 0.9945
```
### Final Model(20th Epoch)
```
score = model.evaluate(X_test, Y_test, verbose=0)
print(score)
```
```
[0.01724281985911366, 0.9945]
```

###Strategy to tune the model
1. Reduce the number of Kernels to reduce number of parameters
2. Tune the Initial Learning late and hte scheduler value 
