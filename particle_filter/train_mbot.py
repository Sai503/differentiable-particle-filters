import torch # Import torch to check for device
from dpf.dpf import DPF
from utils.data_utils import load_data, noisyfy_data, make_batch_iterator, remove_state
from utils.exp_utils import get_default_hyperparams


def train_dpf(task='nav01', data_path='../data/100s', model_path='../models/tmp', plot=False):
    # load training data and add noise
    train_data = load_data(trial_numbers=list(range(5,14)),data_root="../data")
    noisy_train_data = noisyfy_data(train_data)

    # instantiate method with hyperparameters
    hyperparams = get_default_hyperparams()
    method = DPF(**hyperparams['global'])
    
    print(noisy_train_data.keys())
    print(noisy_train_data['l'].shape)
    print(noisy_train_data['o'].shape)
    print(noisy_train_data['s'].shape)
    print(noisy_train_data['a'].shape)
    print("permuting images")
    noisy_train_data['o'] = torch.permute(noisy_train_data['o'], (0, 1, 3, 4, 2)) # [batch, seq_len, height, width, channels]
    print(noisy_train_data['o'].shape)

    # train method and save result in model_path
    # The fit method now handles device detection internally.
    print("--- Starting Training ---")
    method.fit(noisy_train_data, model_path,
               **hyperparams['train'],
               plot_task=task,
               plot=plot)
    
    # save the model
    print("models state dict keys:", method.state_dict().keys())
    torch.save(method.state_dict(), "../models/full_model.pth")
    print("--- Training Finished ---")


if __name__ == '__main__':
    # Example usage: train first, then test
    train_dpf(plot=True) # Set plot=True if you want plots during training
    # test_dpf()
