import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib as pt
import random
import tensorflow as tf
import yaml
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator , load_img
from UNetModel import UNetModel
from PIL import Image , ImageEnhance
from LungsSegDataGenerator import LungsSegDataGenerator
from datasetconfig import split_dataset , create_dataset_csv
from plot_acc_loss import plot_acc_loss 
from datetime import datetime
import os 
local_dt=datetime.now()


def dice_coef(y_true, y_pred):
    y_true_f = tf.reshape(tf.dtypes.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.dtypes.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1.)


config = None
with open('config.yaml') as f: # reads .yml/.yaml files
    config = yaml.safe_load(f)

dataset_df = create_dataset_csv(config["data"]["images_dir"], 
                                config["data"]["right_masks_dir"],
                                config["data"]["left_masks_dir"],
                                config["data"]["data_csv"])

dataset_df = split_dataset(dataset_df, split_per=config['data']['split_per'], seed=1)
dataset_df.head(3)

data_gen = LungsSegDataGenerator(dataset_df, img_size=config["data"]["img_size"], batch_size=config["train"]["bs"] )
x, y = data_gen[0]
print(x.shape, y.shape)

f, axs = plt.subplots(1, 2)
axs[0].axis('off')
axs[0].set_title("Input")
axs[0].imshow((x[0]*255).astype(np.uint8))

axs[1].axis('off')
axs[1].set_title("Mask")
axs[1].imshow(y[0])
plt.show()

unet = UNetModel()
# n_channels=1, deoarece imaginea de input are un singur canal
# n_classes=1, o singura clasa de prezis -> plaman vs background
unet_model = unet.build(*config["data"]["img_size"], n_channels=3, n_classes=3)
unet_model.summary()

train_df= dataset_df.loc[dataset_df['subset']=='train']
train_gen = LungsSegDataGenerator(train_df, img_size=config["data"]["img_size"], batch_size=config["train"]["bs"], shuffle=True)

valid_df = dataset_df.loc[dataset_df['subset']=='valid']
valid_gen = LungsSegDataGenerator(valid_df, img_size=config["data"]["img_size"], batch_size=config["train"]["bs"], shuffle=True)

unet_model.compile(loss="binary_crossentropy",optimizer=tf.keras.optimizers.Adam(learning_rate=config['train']['lr']) , metrics=[dice_coef])

callbacks = [
    keras.callbacks.ModelCheckpoint('damn.h5', save_best_only=True),

    keras.callbacks.CSVLogger("file.csv{local_dt}", separator="," , append=False)
    ]
history=unet_model.fit(train_gen, validation_data=valid_gen , epochs=config['train']['epochs'],callbacks=callbacks,workers=1)

unet_model.save('saved_model/my_model')    

plot_acc_loss(history)

test_df = dataset_df.loc[dataset_df['subset']=='test']
test_gen = LungsSegDataGenerator(test_df, img_size=config["data"]["img_size"], batch_size=config["train"]["bs"], shuffle=False)
result = unet_model.evaluate(test_gen)
print(f"Dice index AVG:{ result[1]} ")

x, y = test_gen[0]
y_pred = unet_model.predict(x)



nr_exs = 4 # nr de exemple de afisat
fig, axs = plt.subplots(nr_exs, 4, figsize=(10, 10))

for i, (img, gt, pred) in enumerate(zip(x[:nr_exs], y[:nr_exs], y_pred[:nr_exs])):
    axs[i][0].axis('off')
    axs[i][0].set_title('Input')
    axs[i][0].imshow(img, cmap='gray')

    axs[i][1].axis('off')
    axs[i][1].set_title('Ground truth')
    axs[i][1].imshow(gt, cmap='gray')
    
    pred[pred > config['test']['threshold']] = 1.0
    pred[pred <= config['test']['threshold']] = 0.0
    # pred = pred.astype("uint8")

    dice_index=dice_coef(gt,pred)
    axs[i][2].axis('off')
    axs[i][2].set_title(f'Prediction. Dice Index = {dice_index}')
    axs[i][2].imshow(pred, cmap='gray')
    S
plt.show()
