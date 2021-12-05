import csv
import copy
import time
from tqdm import tqdm
import torch
import numpy as np
import os
from datetime import datetime
import pathlib
import matplotlib.pyplot as plt


def load_checkpoint(bpath):

    checkpoint_folder = os.path.join(bpath, 'checkpoint')
    checkpoint_filename = os.path.join(
        checkpoint_folder, 'checkpoint.pth.tar')

    bestweights_filename = os.path.join(
        checkpoint_folder, 'best_weights_checkpoint.pth.tar')

    file = pathlib.Path(checkpoint_filename)

    if not file.exists():
        return None, None, None, None, None, None

    file = pathlib.Path(bestweights_filename)

    best_weight = None
    if file.exists():
        best_weight = torch.load(bestweights_filename)
        best_weight = best_weight['state_dict']

    checkpoint = torch.load(checkpoint_filename)

    return checkpoint['epoch'], checkpoint['state_dict'], best_weight, checkpoint['optimizer'], checkpoint['best_loss'], checkpoint['best_pred']


def save_checkpoint(bpath, state, is_best=False):

    checkpoint_folder = os.path.join(bpath, 'checkpoint')

    if is_best:
        best_pred = state['best_pred']
        with open(os.path.join(checkpoint_folder, 'best_pred.txt'), 'w') as f:
            f.write(str(best_pred))

        best_pred = state['best_loss']
        with open(os.path.join(checkpoint_folder, 'best_loss.txt'), 'w') as f:
            f.write(str(best_pred))

        torch.save(state, os.path.join(checkpoint_folder,
                                       'best_weights_checkpoint.pth.tar'))

    torch.save(state, os.path.join(checkpoint_folder,
                                   'checkpoint.pth.tar'))


def train_model(model, criterion, dataloaders, optimizer, scheduler, metrics, bpath, num_epochs=3):

    start_epoch, state_dict, bweights, optm, bloss, bpred = load_checkpoint(
        bpath)

    if start_epoch is not None:
        print("")
        print("NEW CHECKPOINT FOUND! LAST EPOCH ", start_epoch)
        print("")
        model.load_state_dict(state_dict)
        start_epoch += 1

        best_model_wts = copy.deepcopy(bweights)
        best_loss = float(bloss)

        best_Train_dice = 1e-5
        best_Valid_dice = bpred
    else:
        start_epoch = 1
        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = 1e10

        best_Train_dice = 1e-5
        best_Valid_dice = 1e-5

    since = time.time()

    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Initialize the log file for training and testing loss and metrics
    fieldnames = ['epoch', 'Train_loss', 'Valid_loss'] + \
        [f'Train_{m}' for m in metrics.keys()] + \
        [f'Valid_{m}' for m in metrics.keys()]
    with open(os.path.join(bpath, 'log.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    for epoch in range(start_epoch, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        # Initialize batch summary
        batchsummary = {a: [0] for a in fieldnames}

        for phase in ['Train', 'Valid']:
            if phase == 'Train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            # Iterate over data.

            for sample in tqdm(iter(dataloaders[phase])):

                inputs = sample['image'].to(device)
                masks = sample['mask'].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # track history if only in train
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    # loss = criterion(outputs['out'], masks)
                    loss = criterion(outputs, masks)

                    # y_pred = outputs['out'].data.cpu().numpy().squeeze(1)
                    y_pred = outputs.data.cpu().numpy().squeeze(1)
                    y_true = masks.data.cpu().numpy().squeeze(1)

                    for name, metric in metrics.items():
                        if name == 'dice' or name == 'dice_target':
                            # Use a classification threshold of 0.5
                            val_metric = metric(y_pred > 0.5, y_true > 0)

                            if val_metric is not None:
                                batchsummary[f'{phase}_{name}'].append(
                                    val_metric)

                    # backward + optimize only if in training phase
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()

            batchsummary['epoch'] = epoch
            epoch_loss = loss
            batchsummary[f'{phase}_loss'] = epoch_loss.item()
            print('{} Loss: {:.4f}'.format(phase, loss))

        print('New LR: ', scheduler.get_last_lr())
        scheduler.step()

        for field in fieldnames[3:]:
            batchsummary[field] = np.mean(batchsummary[field])

        print(batchsummary)

        epoch_valid_dice = np.mean(batchsummary['Valid_dice_tumor'])
        is_best = False
        with open(os.path.join(bpath, 'log.csv'), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(batchsummary)

            SAVE_BESTLOSS_WEIGTH = False
            if SAVE_BESTLOSS_WEIGTH:
                # deep copy the model
                if phase == 'Valid' and loss < best_loss:
                    print('\nnew best loss: {:.4f} in epoch {}\n'.format(
                        loss, epoch))
                    best_loss = loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    now = datetime.now()
                    str_datetime = now.strftime("%Y%m%d_%H_%M_%S")

                    best_Train_dice = np.mean(batchsummary['Train_dice'])
                    best_Valid_dice = np.mean(batchsummary['Valid_dice'])

                    torch.save(model, os.path.join(
                        bpath, 'weights_partial_epch{}_{}.pt'.format(epoch, str_datetime)))
            else:
                # deep copy the model
                if phase == 'Valid' and epoch_valid_dice > best_Valid_dice:
                    is_best = True
                    print('\nNew valid dice: {:.4f} in epoch {}\n'.format(
                        epoch_valid_dice, epoch))
                    best_loss = loss.item()
                    best_model_wts = copy.deepcopy(model.state_dict())
                    now = datetime.now()
                    str_datetime = now.strftime("%Y%m%d_%H_%M_%S")

                    best_Train_dice = np.mean(batchsummary['Train_dice'])
                    best_Valid_dice = epoch_valid_dice

                    torch.save(model, os.path.join(
                        bpath, 'weights_partial_diceval_epch{}_{}.pt'.format(epoch, str_datetime)))

                    # torch.save(model, os.path.join(
                    #     bpath, 'model_weights_partial.pt'))

            save_checkpoint(bpath, {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_pred': best_Valid_dice,
                'best_loss': best_loss
            }, is_best=is_best)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Lowest by valid dice Loss: {:4f}'.format(best_loss))
    print('Max valid Dice: {:4f}'.format(best_Valid_dice))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return best_Train_dice, best_Valid_dice
