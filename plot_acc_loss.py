import matplotlib.pyplot as plt

def plot_acc_loss(result):

    acc = result.history['dice_coef']
    loss = result.history['loss']
    val_acc = result.history['val_dice_coef']
    val_loss = result.history['val_loss']
    
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.plot(acc, label='Train')
    plt.plot(val_acc, label='Validation')
    plt.title('Dice Index', size=15)
    plt.legend()
    plt.grid(True)
    plt.ylabel('Dice Index')
    plt.xlabel('Epoch')
    
    plt.subplot(122)
    plt.plot(loss, label='Train')
    plt.plot(val_loss, label='Validation')
    plt.title('Loss', size=15)
    plt.legend()
    plt.grid(True)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    
    plt.show()

