import torch # Import torch to check for device
from dpf.dpf import DPF
from utils.data_utils import load_data, noisyfy_data, make_batch_iterator, remove_state
from utils.exp_utils import get_default_hyperparams


def train_dpf(task='nav01', data_path='../data/100s', model_path='../models/tmp', plot=False):
    # load training data and add noise
    train_data = load_data(trial_numbers=[5,6,7,8,9,10,11,12,13],data_root="../data")
    noisy_train_data = noisyfy_data(train_data)

    # instantiate method with hyperparameters
    hyperparams = get_default_hyperparams()
    method = DPF(**hyperparams['global'])
    
    print(noisy_train_data.keys())
    print(noisy_train_data['l'].shape)
    print(noisy_train_data['o'].shape)
    print(noisy_train_data['s'].shape)
    print(noisy_train_data['a'].shape)

    # train method and save result in model_path
    # The fit method now handles device detection internally.
    print("--- Starting Training ---")
    method.fit(noisy_train_data, model_path,
               **hyperparams['train'],
               plot_task=task,
               plot=plot)
    
    # save the model
    method.save_model("../models/full_model", device=torch.device("cpu"))
    print("--- Training Finished ---")


def test_dpf(task='nav01', data_path='../data/100s', model_path='../models/tmp'):
    # --- Determine Device for Testing ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Testing on GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Testing on CPU.")
    # --- End Device Setup ---

    # load test data
    test_data = load_data(data_path=data_path, filename=task + '_test')
    noisy_test_data = noisyfy_data(test_data)
    # Use seq_len from hyperparams if available, otherwise default (e.g., 50)
    hyperparams = get_default_hyperparams()
    test_seq_len = hyperparams['train'].get('seq_len', 50) # Get seq_len used in training
    test_batch_iterator = make_batch_iterator(noisy_test_data, seq_len=test_seq_len)

    # instantiate method and load saved parameters onto the chosen device
    method = DPF(**hyperparams['global'])
    # --- Load model onto the determined device ---
    method.load_model(model_path, device=device)
    # method.eval() is called within load_model now

    # run prediction on 10 test batches
    print("--- Starting Testing ---")
    # The predict method uses the device the model is already on
    num_test_batches = 10
    results = []
    try:
        for i in range(num_test_batches):
            test_batch = next(test_batch_iterator)
            # remove_state might not be needed if predict handles the full batch dict
            # test_batch_input = remove_state(test_batch, provide_initial_state=False)
            # Pass the full batch, predict will handle moving to device
            result = method.predict(test_batch, 100)
            results.append(result)
            # You might want to process or display the result here.
            print(f"Test batch {i} prediction shape: {result.shape}")
    except StopIteration:
        print(f"Warning: Only processed {len(results)} batches, test dataset smaller than {num_test_batches} batches.")
    print("--- Testing Finished ---")
    # You can further process 'results' here


if __name__ == '__main__':
    # Example usage: train first, then test
    train_dpf(plot=True) # Set plot=True if you want plots during training
    # test_dpf()
