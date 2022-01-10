from LungsSegDataGenerator import LungsSegDataGenerator 
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt 
from datasetconfig import split_dataset , create_dataset_csv
import yaml
import numpy as np

def dice_coef(y_true, y_pred):
    y_true_f = tf.reshape(tf.dtypes.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.dtypes.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1.)

def compute_overlaps_masks(masks1, masks2):
    """Computes IoU overlaps between two sets of masks.
    masks1, masks2: [Height, Width, instances]
    """
    
    # If either set of masks is empty return empty result
    if masks1.shape[0] == 0 or masks2.shape[0] == 0:
        return np.zeros((masks1.shape[0], masks2.shape[-1]))
    # flatten masks and compute their areas
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union

    return overlaps 

config = None
with open('config.yaml') as f: # reads .yml/.yaml files
    config = yaml.safe_load(f)

dataset_df = create_dataset_csv(config["data"]["images_dir"], 
                                config["data"]["right_masks_dir"],
                                config["data"]["left_masks_dir"],
                                config["data"]["data_csv"])

dataset_df = split_dataset(dataset_df, split_per=config['data']['split_per'], seed=1)

data_gen = LungsSegDataGenerator(dataset_df, img_size=config["data"]["img_size"], batch_size=config["train"]["bs"] )
  
new_model = keras.models.load_model(r"D:\ai intro\Retele-Convolutionale-Segmentare-\combinatii\1\damn1221_12222021.h5", custom_objects={"dice_coef":dice_coef} )

new_model.summary()
new_model.compile()
test_df = dataset_df.loc[dataset_df['subset']=='test']
test_gen = LungsSegDataGenerator(test_df, img_size=config["data"]["img_size"], batch_size=config["train"]["bs"], shuffle=False)
result = new_model.evaluate(test_gen)

print(f"Test Acc: {result}")

x, y = test_gen[0]
y_pred = new_model.predict(x)


nr_exs = 4 # nr de exemple de afisat
fig, axs = plt.subplots(nr_exs, 4, figsize=(10, 10))

for i, (img, gt, pred) in enumerate(zip(x[:nr_exs], y[:nr_exs], y_pred[:nr_exs])):
    overlap=compute_overlaps_masks(gt,pred)

    axs[i][0].axis('off')
    axs[i][0].set_title('Input')
    axs[i][0].imshow(img, cmap='gray')

    axs[i][1].axis('off')
    axs[i][1].set_title('Ground truth')
    axs[i][1].imshow(gt, cmap='gray')
  
    pred[pred > config['test']['threshold']] = 1
    pred[pred <= config['test']['threshold']] = 0
    # pred = pred.astype("uint8")
    print(img.dtype, gt.dtype, pred.dtype)
    print(gt.shape, pred.shape)
  
    dice_index=dice_coef(gt,pred)
    axs[i][2].axis('off')
    axs[i][2].set_title(f'Prediction. Dice_index={dice_index}')
    axs[i][2].imshow(pred, cmap='gray')

    axs[i][3].axis('off')
    axs[i][3].set_title('Overlap')
    axs[i][3].imshow(overlap)  

plt.show()



