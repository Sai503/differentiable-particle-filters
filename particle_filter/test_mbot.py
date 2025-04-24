import torch # Import torch to check for device
from dpf.dpf import DPF
from utils.data_utils import load_data, noisyfy_data, make_batch_iterator, remove_state
from utils.exp_utils import get_default_hyperparams
import time



def test_dpf():
    print("TEST START")
    start = time.time()

    # --- Determine Device for Testing ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Testing on GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Testing on CPU.")
    # --- End Device Setup ---

    



    # load test data
    test_data = load_data(trial_numbers=[5,6,7,8],data_root="../data1")
    noisy_test_data = noisyfy_data(test_data)
    # reshape image data from [batch, seq_len, channels, height, width] to [batch, seq_len, height, width, channels]
    noisy_test_data['o'] = torch.permute(noisy_test_data['o'], (0, 1, 3, 4, 2)) # [batch, seq_len, height, width, channels]
    # Use seq_len from hyperparams if available, otherwise default (e.g., 50)
    # print('test data shape')
    print(noisy_test_data['o'].shape)
    hyperparams = get_default_hyperparams()
    test_seq_len = 10 # Get seq_len used in training
    test_batch_iterator = make_batch_iterator(noisy_test_data, seq_len=test_seq_len)

    # instantiate method and load saved parameters onto the chosen device
    # model path
    model_path = "../models_trained/full_model.pth"
    # load model
    method = DPF(**hyperparams['global'])
    # load state dict
    # The load_state_dict method now handles device detection internally.
    print("--- Loading Model ---")
    method.load_state_dict(torch.load(model_path, map_location=device))
    # move model to device
    method.to(device)
    # stats to tensor
    # method._stats_to_tensors(device)
    print("--- Model Loaded ---")

    loaded = time.time()
    # print model state dict keys


    # run prediction on 10 test batches
    print("--- Starting Testing ---")
    # The predict method uses the device the model is already on
    
    num_test_batches = 10
    num_steps = 0
    results = []
    try:
        for i in range(num_test_batches):
            test_batch = next(test_batch_iterator)
            # remove_state might not be needed if predict handles the full batch dict
            # test_batch_input = remove_state(test_batch, provide_initial_state=False)
            # Pass the full batch, predict will handle moving to device
            num_steps += (test_batch['o'].shape[0]  * test_batch['o'].shape[1])
            result = method.predict(test_batch, 100)
            results.append(result)
            # You might want to process or display the result here.
            print(f"Test batch {i} prediction shape: {result.shape}")

            # calculate error
            # Assuming result and test_batch have the same shape
            error = torch.abs(result - test_batch['s'])
            print(f"Test batch {i} result shape: {result.shape}")
            print(f"Test batch {i} s shape: {test_batch['o'].shape}")
            print(f"Test batch {i} error shape: {error.shape}")
            print(f"Test batch {i} error mean: {error.mean().item()}")
            # print first 5 errors
            print(f"Test batch {i} error first 5: {error.flatten()[:5]}")
            # print first 5 predictions
            print(f"Test batch {i} prediction first 5: {result.flatten()[:5]}")
            # print first 5 ground truth
            print(f"Test batch {i} ground truth first 5: {test_batch['s'].flatten()[:5]}")

    except StopIteration:
        print(f"Warning: Only processed {len(results)} batches, test dataset smaller than {num_test_batches} batches.")
    print("--- Testing Finished ---")
    done = time.time()
    print("total time")
    print(done - start)
    print("loaded time")
    print(loaded - start)
    print("loading per img")
    print((loaded-start) / (noisy_test_data['o'].shape[0] * noisy_test_data['o'].shape[1] ) )
    print("total per step")
    print((done - start) / (num_steps))

    # You can further process 'results' here


if __name__ == '__main__':
    # Example usage: train first, then test
    # train_dpf(plot=True) # Set plot=True if you want plots during training
    # start = time.time()
    test_dpf()
    # end = time.time()
    # print("runtime:")
    # print(end - start)
