import os
import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# For file I/O we still use numpy; alternatively you could switch to torch.save/load.
import numpy as np

from utils.plotting_utils import plot_trajectories, plot_maze, plot_observations, plot_trajectory

# Use torch.remainder to mimic modulo for tensors.
def wrap_angle(angle):
    # angle: a torch.Tensor (or scalar) in radians.
    return torch.remainder(angle - math.pi, 2 * math.pi) - math.pi


def mix_data(file_in1, file_in2, file_out, steps_per_episode=100, num_episodes=1000):
    # Load data from npz files and convert to torch tensors.
    data1 = {k: torch.tensor(v) for k, v in np.load(file_in1).items()}
    data2 = {k: torch.tensor(v) for k, v in np.load(file_in2).items()}
    data_mix = {}
    half = steps_per_episode * num_episodes // 2
    for key in data1.keys():
        d1 = data1[key][:half]
        d2 = data2[key][:half]
        data_mix[key] = torch.cat((d1, d2), dim=0)
    # Save using torch.save; note that the file format will be a .pt file.
    torch.save(data_mix, file_out)


def average_nn(states_from, states_to, step_sizes, num_from=10, num_to=100):
    # Reshape states to [-1, 3] and sample random indices.
    states_from = states_from.view(-1, 3)
    states_to = states_to.view(-1, 3)
    idx_from = torch.randint(0, states_from.shape[0], (num_from,))
    idx_to = torch.randint(0, states_to.shape[0], (num_to,))
    total = 0.0
    for i in range(3):
        diff = states_from[idx_from, None, i] - states_to[None, idx_to, i]
        if i == 2:
            diff = wrap_angle(diff)
        total += (diff / step_sizes[i]) ** 2
    # Compute the fraction of cases where the minimal distance is > 0.5.
    average_dist = torch.mean((torch.min(total, dim=1)[0] > 0.5).float())
    return average_dist.item()


def load_data(data_path='../data/100s', filename='nav01_train', steps_per_episode=100, num_episodes=None):
    # Load from npz and convert to torch tensors.
    file_path = os.path.join(data_path, filename + '.npz')
    raw = np.load(file_path)
    data = {k: torch.tensor(v, dtype=torch.float32) for k, v in raw.items()}
    # Reshape each array: [total_steps] -> [-1, steps_per_episode, ...]
    for key in data.keys():
        if num_episodes is not None:
            data[key] = data[key][:num_episodes * steps_per_episode]
        new_shape = [-1, steps_per_episode] + list(data[key].shape[1:])
        data[key] = data[key].view(*new_shape)
    # Convert degrees into radians for pose and velocity.
    for key in ['pose', 'vel']:
        data[key][:, :, 2] *= math.pi / 180.0
    data['pose'][:, :, 2] = wrap_angle(data['pose'][:, :, 2])
    abs_d_x = data['pose'][:, 1:, 0:1] - data['pose'][:, :-1, 0:1]
    abs_d_y = data['pose'][:, 1:, 1:2] - data['pose'][:, :-1, 1:2]
    d_theta = wrap_angle(data['pose'][:, 1:, 2:3] - data['pose'][:, :-1, 2:3])
    s = torch.sin(data['pose'][:, :-1, 2:3])
    c = torch.cos(data['pose'][:, :-1, 2:3])
    rel_d_x = c * abs_d_x + s * abs_d_y
    rel_d_y = s * abs_d_x - c * abs_d_y

    return {'o': data['rgbd'][:, 1:, :, :, :3],
            's': data['pose'][:, 1:, :],
            'a': torch.cat([rel_d_x, rel_d_y, d_theta], dim=-1)}


def compute_staticstics(data):
    means = {}
    stds = {}
    state_step_sizes = []
    state_mins = []
    state_maxs = []

    for key in 'osa':
        # Compute means over the first two dimensions (episodes and time).
        means[key] = torch.mean(data[key], dim=(0, 1), keepdim=True)
        if key == 's':
            means[key][:, :, 2] = 0  # leave orientation mean at 0
        if key == 'a':
            means[key] = means[key] * 0  # set to zero
        # Compute std over all but the last dimension.
        dims = tuple(range(len(data[key].shape) - 1))
        stds[key] = torch.std(data[key] - means[key], dim=dims, keepdim=True)
        if key == 's':
            stds[key][:, :, :2] = torch.mean(stds[key][:, :, :2])
        if key == 'a':
            stds[key][:, :, :2] = torch.mean(stds[key][:, :, :2])
    for i in range(3):
        steps = (data['s'][:, 1:, i] - data['s'][:, :-1, i]).view(-1)
        if i == 2:
            steps = wrap_angle(steps)
        state_step_sizes.append(torch.mean(torch.abs(steps)))
    avg = (state_step_sizes[0] + state_step_sizes[1]) / 2
    state_step_sizes[0] = state_step_sizes[1] = avg
    state_step_sizes = torch.stack(state_step_sizes)
    for i in range(3):
        state_mins.append(torch.min(data['s'][:, :, i]))
        state_maxs.append(torch.max(data['s'][:, :, i]))
    state_mins = torch.stack(state_mins)
    state_maxs = torch.stack(state_maxs)
    return means, stds, state_step_sizes, state_mins, state_maxs


def split_data(data, ratio=0.8, categories=['train', 'val']):
    print('SPLIT {}'.format(data['s'].shape))
    split_data_dict = {categories[0]: {}, categories[1]: {}}
    for key in data.keys():
        split_point = int(data[key].shape[0] * ratio)
        split_data_dict[categories[0]][key] = data[key][:split_point]
        split_data_dict[categories[1]][key] = data[key][split_point:]
    for cat in split_data_dict:
        print('SPLIT --> {}: {}'.format(cat, len(split_data_dict[cat]['s'])))
    return split_data_dict


def reduce_data(data, num_episodes):
    new_data = {}
    for key in 'osa':
        new_data[key] = data[key][:num_episodes]
    return new_data


def shuffle_data(data):
    new_data = {}
    indices = torch.randperm(data['o'].shape[0])
    for key in 'osa':
        new_data[key] = data[key][indices]
    return new_data


def remove_state(data, provide_initial_state=False):
    new_data = {}
    new_data['o'] = data['o']
    new_data['a'] = data['a']
    if provide_initial_state:
        new_data['s'] = data['s'][..., :1, :]
    return new_data


def noisify_data_condition(data, condition):
    print('condition', condition)
    if condition == 'odom0_imgTG':
        return noisyfy_data(data, odom_noise_factor=0.0)
    elif condition == 'odom5_imgTG':
        return noisyfy_data(data, odom_noise_factor=0.5)
    elif condition == 'odom10_imgTG':
        return noisyfy_data(data)
    elif condition == 'odom20_imgTG':
        return noisyfy_data(data, odom_noise_factor=2.0)
    elif condition == 'odomX_imgTG':
        data = noisyfy_data(data, odom_noise_factor=0.0)
        shape = data['a'].shape
        a = data['a'].view(-1, shape[-1])
        # Shuffle actions using torch.randperm on indices.
        idx = torch.randperm(a.shape[0])
        a = a[idx]
        data['a'] = a.view(shape)
        return data
    elif condition == 'odom10_imgC':
        return noisyfy_data(data, img_noise_factor=0.0, img_random_shift=False)
    elif condition == 'odom10_imgG':
        return noisyfy_data(data, img_noise_factor=1.0, img_random_shift=False)
    elif condition == 'odom10_imgT':
        return noisyfy_data(data, img_noise_factor=0.0, img_random_shift=True)
    elif condition == 'odom10_imgX':
        data = noisyfy_data(data, img_noise_factor=0.0, img_random_shift=False)
        shape = data['o'].shape
        o = data['o'].view(-1, shape[-1])
        idx = torch.randperm(o.shape[0])
        o = o[idx]
        data['o'] = o.view(shape)
        return data


def noisyfy_data(data, odom_noise_factor=1.0, img_noise_factor=1.0, img_random_shift=True):
    print("noisyfying data ...")
    new_data = {}
    # Ensure keys exist and clone tensors to avoid modifying original data
    if 's' in data:
        new_data['s'] = data['s'].clone()
    if 'a' in data:
        device = data['a'].device
        noise_a = torch.normal(mean=1.0, std=0.1 * odom_noise_factor, size=data['a'].shape, device=device, dtype=data['a'].dtype)
        new_data['a'] = data['a'] * noise_a
    if 'o' in data:
        B, T, H, W, C = data['o'].shape # Original shape (e.g., B, T, 32, 32, 3)
        target_H, target_W = 24, 24     # Target cropped shape
        device_o = data['o'].device
        dtype_o = data['o'].dtype

        # Initialize new_o with the TARGET shape [B, T, 24, 24, C]
        new_o = torch.zeros(B, T, target_H, target_W, C, dtype=dtype_o, device=device_o)

        max_offset_H = H - target_H # e.g., 32 - 24 = 8
        max_offset_W = W - target_W # e.g., 32 - 24 = 8

        for i in range(B):
            for j in range(T):
                if img_random_shift:
                    # Ensure offset doesn't go out of bounds for cropping
                    offset_h = torch.randint(0, max_offset_H + 1, (1,), device=device_o).item()
                    offset_w = torch.randint(0, max_offset_W + 1, (1,), device=device_o).item()
                else:
                    # Center crop: start at (H-target_H)/2, (W-target_W)/2
                    offset_h = (H - target_H) // 2 # e.g., (32-24)//2 = 4
                    offset_w = (W - target_W) // 2 # e.g., (32-24)//2 = 4

                # Crop the original image and assign to the correctly sized new_o slice
                new_o[i, j] = data['o'][i, j, offset_h:offset_h + target_H, offset_w:offset_w + target_W, :]

        # Add noise to the cropped image tensor
        noise_o = torch.normal(mean=0.0, std=20 * img_noise_factor, size=new_o.shape, device=device_o, dtype=dtype_o)
        new_o = new_o + noise_o
        new_data['o'] = new_o
    else:
        print("Warning: Key 'o' not found in data during noisyfy_data.")

    return new_data

def make_batch_iterator(data, batch_size=32, seq_len=10):
    # Generator that yields a batch from random episodes and start steps.
    while True:
        episodes = torch.randint(0, data['s'].shape[0], (batch_size,))
        start_steps = torch.randint(0, data['s'].shape[1] - seq_len - 1, (batch_size,))
        batches = {}
        for k in data.keys():
            batch_list = []
            for i, j in zip(episodes, start_steps):
                batch_list.append(data[k][i:i+1, j:j+seq_len])
            batches[k] = torch.cat(batch_list, dim=0)
        yield batches


def make_repeating_batch_iterator(data, epoch_len, batch_size=32, seq_len=10):
    repeating_episodes = torch.randint(0, data['s'].shape[0], (epoch_len, batch_size))
    repeating_start_steps = torch.randint(0, data['s'].shape[1] - seq_len - 1, (epoch_len, batch_size))
    while True:
        for episodes, start_steps in zip(repeating_episodes, repeating_start_steps):
            batches = {}
            for k in data.keys():
                batch_list = []
                for i, j in zip(episodes, start_steps):
                    batch_list.append(data[k][i:i+1, j:j+seq_len])
                batches[k] = torch.cat(batch_list, dim=0)
            yield batches


def make_complete_batch_iterator(data, batch_size=1000, seq_len=10):
    num_episodes = data['s'].shape[0]
    num_start_steps = data['s'].shape[1] - seq_len
    batch_indices = [(i, j) for i in range(num_episodes) for j in range(num_start_steps)]
    while batch_indices:
        current = batch_indices[:batch_size]
        batch_indices = batch_indices[batch_size:]
        batches = {}
        for k in data.keys():
            batch_list = []
            for (i, j) in current:
                batch_list.append(data[k][i:i+1, j:j+seq_len])
            batches[k] = torch.cat(batch_list, dim=0)
        yield batches


def compare_data_coverage():
    task = 'nav02'
    data = load_data(filename=task + '_train', data_path='../data/100s_mix', steps_per_episode=100, num_episodes=100)
    means, stds, state_step_sizes, state_mins, state_maxs = compute_staticstics(data)
    states = {}
    states['ab'] = data['s']
    data = load_data(filename=task + '_train', data_path='../data/100s_astar', steps_per_episode=100, num_episodes=100)
    states['b'] = data['s']
    data = load_data(filename=task + '_train', data_path='../data/100s', steps_per_episode=100, num_episodes=100)
    states['a'] = data['s']
    for f in ['a', 'b']:
        for t in ['a', 'b', 'ab']:
            d = average_nn(states_from=states[f], states_to=states[t], step_sizes=state_step_sizes, num_from=10000, num_to=10000)
            print('{} <- {}: {}'.format(f, t, d))
            plt.pause(0.01)


if __name__ == '__main__':
    # Uncomment to mix or compare data coverage.
    # mix_data('../data/100s/nav02_test.npz',
    #          '../data/100s_astar/nav02_test.npz',
    #          '../data/100s_mix/nav02_test.pt')
    # compare_data_coverage()

    task = 'nav03'
    data = load_data(filename=task + '_train', data_path='../data/100s', steps_per_episode=100, num_episodes=1000)
    data = split_data(data, ratio=0.5)
    scaling = 0.5
    if task == 'nav01':
        plt.figure(figsize=[10*scaling, 5*scaling])
    elif task == 'nav02':
        plt.figure(figsize=[15*scaling, 9*scaling])
    elif task == 'nav03':
        plt.figure(figsize=[20*scaling, 13*scaling])
    i = 108
    torch.manual_seed(i)
    dat = shuffle_data(data['train'])
    dat = reduce_data(dat, 1)
    dat = noisyfy_data(dat)
    plot_trajectory(dat, figure_name=None, emphasize=0, mincolor=0.0, linewidth=0.5)
    plot_maze(task)
    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    plt.tight_layout()
    plt.savefig("../plots/" + task + ".pdf",
                bbox_inches='tight',
                transparent=False,
                pad_inches=0)
    plt.figure()
    plot_observations(dat, n=5)
    plt.savefig("../plots/" + task + "_noisy_obs.pdf",
                bbox_inches='tight',
                transparent=False,
                pad_inches=0,
                dpi=200)
    plt.show()
