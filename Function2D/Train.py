import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def train_model(model, loss, optimizer, scheduler, num_epochs, train_dataloader, val_dataloader, device):
    writer = SummaryWriter("Graph")


    for epoch in range(num_epochs):
        print('Epoch {}/{}:'.format(epoch, num_epochs - 1), flush=True)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                dataloader = train_dataloader
                #scheduler.step()
                model.train()  # Set model to training mode
            else:
                dataloader = val_dataloader
                model.eval()   # Set model to evaluate mode

            running_loss = 0.

            # Iterate over data.
            for inputs, labels in tqdm(dataloader):
                # inputs = inputs.to(device)
                # labels = labels.to(device)

                optimizer.zero_grad()

                # forward and backward
                with torch.set_grad_enabled(phase == 'train'):
                    preds = model(inputs)
                    preds = torch.flatten(preds)
                    # print(preds.shape)
                    # print(labels.shape)
                    # print(preds)
                    # print(labels)
                    # return
                    loss_value = loss(preds, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss_value.backward()
                        optimizer.step()

                # statistics
                running_loss += loss_value.item()

            epoch_loss = running_loss / len(dataloader)
            if phase == 'train':
                writer.add_scalar("Loss train: ", epoch_loss, epoch)
            else:
                writer.add_scalar("Loss validation: ", epoch_loss, epoch)

            print('{} Loss: {:.4f}'.format(phase, epoch_loss), flush=True)
    writer.close()
    return model