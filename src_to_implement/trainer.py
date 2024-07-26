import os
import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm


class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 batch_size=32,                # Batch size
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda
        self.batch_size = batch_size
        

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()
            
    def save_checkpoint(self, epoch):
        # Create directory if it does not exist
        os.makedirs('checkpoints', exist_ok=True)
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
    
    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else 'cpu')
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(self.batch_size, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=17,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})
            
    def train_step(self, x, y):
        # Reset the gradients
        self._optim.zero_grad()

        # Propagate through the network
        outputs = self._model(x)

        # Calculate the loss
        loss = self._crit(outputs, y)

        # Compute gradient by backward propagation
        loss.backward()

        # Update weights
        self._optim.step()

        # Return the loss
        return loss
    
    def val_test_step(self, x, y):
        # Set the model to evaluation mode
        self._model.eval()

        # Disable gradient computation
        with t.no_grad():
            # Transfer the batch to the GPU if given
            if self._cuda:
                x = x.cuda()
                y = y.cuda()

            # Propagate through the network and calculate the loss and predictions
            outputs = self._model(x)
            loss = self._crit(outputs, y)

            predictions = (outputs > 0.5).float()

        # Return the loss and predictions
        return loss, predictions
    
    def train_epoch(self):
        # Set the model to training mode
        self._model.train()

        # Initialize variables for loss and number of samples
        total_loss = 0.0
        num_samples = 0

        # Iterate through the training set with a progress bar
        for x, y in tqdm(self._train_dl, desc="Training", leave=False):
            # Transfer the batch to the GPU if given
            if self._cuda:
                x = x.cuda()
                y = y.cuda()

            # Perform a training step
            loss = self.train_step(x, y)

            # Update the total loss and number of samples
            total_loss += loss.item() * x.size(0)
            num_samples += x.size(0)

        # Calculate the average loss for the epoch
        avg_loss = total_loss / num_samples

        # Return the average loss
        return avg_loss
    
    def val_test(self):
        # Set the model to evaluation mode
        self._model.eval()

        # Disable gradient computation
        with t.no_grad():
            # Initialize variables for loss and metrics
            total_loss = 0.0
            total_f1 = [0.0, 0.0]
            num_samples = 0

            # Iterate through the validation/test set with a progress bar
            for x, y in tqdm(self._val_test_dl, desc="Validation", leave=False):
                # Transfer the batch to the GPU if given
                if self._cuda:
                    x = x.cuda()
                    y = y.cuda()

                # Perform a validation/test step
                loss, predictions = self.val_test_step(x, y)

                # Update the total loss and number of samples
                total_loss += loss.item() * x.size(0)
                num_samples += x.size(0)

                # Calculate F1 score for each class
                f1 = f1_score(y.cpu().numpy(), predictions.cpu().numpy(), average=None)
                total_f1[0] += f1[0] * x.size(0)
                total_f1[1] += f1[1] * x.size(0)

            # Calculate the average loss and F1 score for each class
            avg_loss = total_loss / num_samples
            avg_f1 = [total_f1[0] / num_samples, total_f1[1] / num_samples]
            mean_f1 = sum(avg_f1) / len(avg_f1)

            # Return the loss and F1 scores
            return avg_loss, avg_f1, mean_f1
        
    
    def fit(self, epochs=-1):
        best_val_loss = None
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch 
        train_losses, val_losses, val_f1_crack, val_f1_inactive, val_f1_means = [], [], [], [], []
        epoch = 0
        best_val_loss = float('inf')
        best_epoch = 0
        for epoch in range(epochs):
            print(f"Epoch {epoch}/{epochs - 1}")
            # Train for an epoch
            train_loss = self.train_epoch()

            # Calculate the loss and F1 score on the validation set
            val_loss, val_f1, mean_f1 = self.val_test()

            # Append the losses to the respective lists
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_f1_crack.append(val_f1[0])
            val_f1_inactive.append(val_f1[1])
            val_f1_means.append(mean_f1)

            # Save the model checkpoint if there is improvement in validation loss
            if val_loss < best_val_loss or best_val_loss is None:
                best_val_loss = val_loss
                self.save_checkpoint(epoch)
                best_epoch = epoch

            # Check if early stopping should be performed
            if self._early_stopping_patience > 0:
                if epoch - best_epoch >= self._early_stopping_patience:
                    break

            # Increment the epoch counter
            epoch += 1

            # Print the progress
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1 (crack): {val_f1[0]:.4f}, Val F1 (inactive): {val_f1[1]:.4f}, Mean F1: {mean_f1:.4f}")

        # Return the losses and F1 scores for both training and validation
        return train_losses, val_losses, val_f1_crack, val_f1_inactive, val_f1_means