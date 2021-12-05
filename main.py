import os
import torch

from trainer import train_model
import utils as ut

from loss.diceloss import diceloss
from metrics import m

from models.UnetAttention import UnnetAttention

def run_nn():
    """
        Version requirements:
            PyTorch Version:  >1.2.0
            Torchvision Version:  >0.4.0a0+6b959ee
    """

    """
        Parameters to execute the method
    """
    root_dir = r'./data'

    epochs = 100
    batch_size = 8
    # Filename of the final model weigths
    weight_filename = "weights_final.pt"
   
    data_aug = 'online'

    log_path = './weights/'
 
    """
        Main 
    """

    ut.create_nested_dir(log_path)

    # Loads the distribution of the cases between train and val
    cases = ut.load_dataset_dist()

    # Create the dataloader
    dataloaders = ut.get_data_loaders(
        data_aug, cases, root_dir, batch_size)

    model = UnnetAttention()

    model.train()

    # Load the loss object by name
    criterion = diceloss()
    # Specify the optimizer with a lower learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=7, gamma=0.9
    )

    # Specify the evalutation metrics
    metrics = {'dice': m.mean_dice_coef,
               'dice_target': m.mean_dice_coef_remove_empty}

    train_model(model, criterion, dataloaders,
                optimizer, exp_lr_scheduler, bpath=log_path, metrics=metrics, num_epochs=epochs)

    # Save the trained model
    torch.save(model, os.path.join(log_path, weight_filename))
    print('\n\n ### ===> Training finished sucessfully!\n\n')


if __name__ == '__main__':
    run_nn()
