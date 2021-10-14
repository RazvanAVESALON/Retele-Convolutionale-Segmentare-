import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib as pt
import random
import tensorflow as tf
import yaml

from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model

from UNetModel import UNetModel
config = None
with open('config.yaml') as f: # reads .yml/.yaml files
    config = yaml.safe_load(f)

def create_dataset_csv(images_dir, right_masks_dir, left_masks_dir, csv_path):
    
    

    # se citesc toate caile catre imagini si masti de segmentare
    # este important sa fie in aceeasi ordine
    images = sorted(list(pt.Path(images_dir).rglob("*.png")))
    right_masks = sorted(list(pt.Path(right_masks_dir).rglob("*.png")))
    left_masks = sorted(list(pt.Path(left_masks_dir).rglob("*.png")))

    # se verifica daca nu exista masti lipsa pentru unul dintre plamani
    assert len(right_masks) == len(left_masks), \
         f"nr. of right lung masks {len(right_masks)} != {len(left_masks)} nr. of left lung masks"
    
    # se verifica daca nu exista imagini sau masti lipsa
    assert len(images) == len(left_masks), \
         f"nr. of image{len(images)} != {len(left_masks)} nr. of masks"

    # se creaza un dictionar de liste, pe baza caruia se creaza obiectul de tip pandas.DataFrame
    dataset_data = {"image_path": images, "right_lung_mask_path": right_masks, "left_lung_mask_path": left_masks}

    dataset_df = pd.DataFrame(data=dataset_data)
    dataset_df.to_csv(csv_path, index=False)
    print(f"Saved dataset csv {csv_path}")

    return dataset_df

def split_dataset(dataset_df, split_per, seed=1):
    """Impartirea setului de date in antrenare, validare si testare in mod aleatoriu

    Args:
        dataset_df (pandas.DataFrame): contine caile catre imaginile de input si mastile de segmentare
        split_per (dict): un dictionare de forma {"train": float, "valid": float, "test": float} ce descrie
            procentajele pentru fiecare subset
        seed (int, optional): valoarea seed pentru reproducerea impartirii setului de date. Defaults to 1.
    """
    # se amesteca aleatoriu indecsii DataFrame-ului
    # indexul este un numar (de cele mai multe ori) asociat fiecarui rand
    indices = dataset_df.index.to_numpy() 
    total = len(indices)
    random.seed(seed)
    random.shuffle(indices)

    # se impart indecsii in functie de procentele primite ca input
    train_idx = int(total * split_per["train"])
    valid_idx = train_idx + int(total * split_per["valid"])
    test_idx = train_idx + valid_idx + int(total * split_per["test"])

    train_indices = indices[:train_idx]
    valid_indices = indices[train_idx:valid_idx]
    test_indices = indices[valid_idx:test_idx]

    # print(len(train_indices), len(valid_indices), len(test_indices))

    # se adauga o noua coloana la DataFrame care specifica in ce subset face parte o imagine si mastile de segmentare asociate
    dataset_df['subset'] = ""
    dataset_df.loc[train_indices, "subset"] = "train"
    dataset_df.loc[valid_indices, "subset"] = "valid"
    dataset_df.loc[test_indices, "subset"] = "test"

    return dataset_df

dataset_df = create_dataset_csv(config["data"]["images_dir"], 
                                config["data"]["right_masks_dir"],
                                config["data"]["left_masks_dir"],
                                config["data"]["data_csv"])


dataset_df = split_dataset(dataset_df, split_per=config['data']['split_per'], seed=1)
dataset_df.head(3)


class LungSegDataGenerator(keras.utils.Sequence):
    """Un DataGenerator custom pentru setul de date pentru segmentare plamanilor"""

    def __init__(self, dataset_df, img_size, batch_size, shuffle=True):
        self.dataset_df = dataset_df.reset_index(drop=True)
        self.img_size = tuple(img_size)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.dataset_df))

    def __len__(self):
        """
        Returns:
            int: Returneaza numarul de batches per epoca
        """
        return self.dataset_df/self.batch_size

    def __combine_masks(self, img_right, img_left):
        """Combina mastile pentru cei doi plamani intr-o singura masca

        Args:
            img_right (pillow.Image): masca pentru plamanul drept
            img_left (pillow.Image): masca pentru plamanul stang

        Returns:
            numpy.array: masca cu cei doi plamani
        """

        img_right = np.array(img_right, dtype="uint8") * 1/255
        img_left = np.array(img_left, dtype="uint8") * 1/255

        img = (img_right + img_left).astype("uint8")

        return img


    def __getitem__(self, idx):
        """Returneaza un tuple (input, target) care corespunde cu batch #idx.

        Args:
            idx (int): indexul batch-ului curent

        Returns:
           tuple:  (input, target) care corespunde cu batch #idx
        """

        i = idx * self.batch_size
        batch_indexes = self.indexes[i:i+self.batch_size]
        batch_df = self.dataset_df.loc[batch_indexes, :].reset_index(drop=True)

        # x, y trebuie sa aiba dimensiunea [batch size, height, width, nr de canale]
        x = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="float32")

        for i, row in batch_df.iterrows():
            # citeste imaginea de input de la calea row['image_path]
            # hint: functia load_img
            img = # de completat
            x[i] = # de completat

            # citeste mastile de segmentare pentru cei doi plamani
            img_right = # de completat
            img_left = # de completat

            img = self.__combine_masks(img_right, img_left)

            y[i] = # de completat
        
        return x, y
    
    def on_epoch_end(self):
        """
        Actualizeaza indecsii dupa fiecare epoca si ii amesteca daca parametrul shuffle este True 
        """
        self.indexes = np.arange(len(self.dataset_df))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)