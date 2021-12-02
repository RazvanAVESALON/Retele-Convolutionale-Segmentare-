from LungsSegDataGenerator import LungsSegDataGenerator 
import tensorflow as tf
import matplotlib.pyplot as plt 
from datasetconfig import split_dataset , create_dataset_csv
import yaml

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

data_gen = LungsSegDataGenerator(dataset_df, img_size=config["data"]["img_size"], batch_size=config["train"]["bs"] )

new_model = tf.keras.models.load_model('saved_model/my_model')
new_model.summary()

test_df = dataset_df.loc[dataset_df['subset']=='test']
test_gen = LungsSegDataGenerator(test_df, img_size=config["data"]["img_size"], batch_size=config["train"]["bs"], shuffle=False)
result = new_model.evaluate(test_gen)
print(f"Test Acc: {result[1] * 100}")

x, y = test_gen[0]
y_pred = new_model.predict(x)
y_pred.shape

nr_exs = 4 # nr de exemple de afisat
fig, axs = plt.subplots(nr_exs, 3, figsize=(10, 10))

for i, (img, gt, pred) in enumerate(zip(x[:nr_exs], y[:nr_exs], y_pred[:nr_exs])):
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
plt.show()
