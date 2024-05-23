from torch.utils.data import Dataset
from stork.generators import DataGenerator
import torch
import numpy as np

def get_random_signal(rng, period=10.0, alpha=1.5, nb_repeats=11, dt=10e-3, cutoff=128):
    """
    Generates a random signal based on a superposition of sinusoidal components. 
    Each component has a random amplitude and phase shift. The signal is designed 
    to model instantaneous firing rates, such as those in neurons.

    Parameters:
    - rng (np.random.Generator): Random number generator object.
    - period (float): Period of the sinusoidal components.
    - alpha (float): Damping factor for the amplitude of higher frequency components.
    - nb_repeats (int): Number of times the period is repeated in the signal.
    - dt (float): Time step for the signal generation.
    - cutoff (int): Number of sinusoidal components to generate.
    - seed (int): Seed for random number generation to ensure reproducibility.

    Returns:
    - times (ndarray): Array of time points for which the signal is computed.
    - signal (ndarray): The generated signal as an array of values corresponding to 'times'.
    """

    # # Set the random seed for reproducibility
    # np.random.seed(seed)

    # Randomly generate parameters for each sinusoid (amplitude and phase shift)
    theta = np.random.rand(cutoff, 2)
    
    # Calculate the total duration of the signal
    duration = period * nb_repeats
    
    # Generate a time array from 0 to 'duration' with a step of 'dt'
    times = np.arange(0, duration, dt)
    
    # Initialize the signal array
    signal = 0.0
    
    # Add each sinusoidal component to the signal
    for i, params in enumerate(theta):
        a, b = params  # amplitude and phase shift
        # Add the sinusoidal component, with amplitude decreasing with frequency
        signal += 1.0 / alpha**i * a * np.sin(2 * np.pi * i * (times + b) / period)

    # Normalize the signal to have zero mean and unit variance
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-4)
    
    # Return the times and the corresponding signal
    return times, signal




def split_data_by_time(data, split_duration):
    """
    Splits data into segments based on a specified time duration.
    
    Parameters:
    - data: numpy array of shape (n, m), where n is the number of data points and m is the number of features.
            It is assumed that the first column contains the timestamps.
    - split_duration: the duration for each split segment.
    
    Returns:
    - splits: a list of numpy arrays, each containing the data within a specific time segment.
    """
    
    # Find the maximum time in the data
    max_time = data[:, 0].max()
    
    # Initialize a list to hold the split data segments
    splits = []
    
    # Initialize the start time for the first segment
    start_time = 0
    
    # Loop to split the data until the start time exceeds the maximum time
    while start_time < max_time:
        # Define the end time for the current segment
        end_time = start_time + split_duration
        
        # Create a mask to select data within the current time segment
        mask = (data[:, 0] >= start_time) & (data[:, 0] < end_time)
        
        # Extract the data for the current time segment based on the mask
        split_data = data[mask]
        
        # Adjust the timestamps to be relative to the start time of the current segment
        split_data[:, 0] -= start_time
        
        # Add the current time segment data to the list of splits
        splits.append(split_data)
        
        # Update the start time to move to the next segment
        start_time = end_time
    
    return splits



        
class FileModulatedPoissonGroup():
    """
    A class to generate Poisson spike trains modulated by time-varying firing rates read from a file.
    """

    def __init__(self, time_step=2e-3, loop_mode=True, seed=123):
        """
        Initializes the Poisson group with specific simulation parameters.

        Parameters:
        - time_step (float): The simulation time step in seconds.
        - loop_mode (bool): If True, the firing rates from the file are looped continuously.
        - seed (int): Seed for the random number generator for reproducibility.
        """
        self.dt = time_step
        self.loop = loop_mode
        self.seed = seed
        np.random.seed(seed)
        self.init()

    def init(self, x_offset=0):
        """
        Initializes or resets the variables for spike generation.
        
        Parameters:
        - x_offset (int): Offset added to the neuron indices, useful when using multiple instances for different neuron groups.
        """
        self.ftime = 0  # Future time when the next rate change occurs
        self.ftime_offset = 0  # Offset for looping over signals
        self.ltime = 0  # Last time a rate change occurred
        self.rate_m = 0.0  # Gradient of rate change
        self.rate_n = 0.0  # Initial rate at the last change
        self.last_rate = 0.0  # The last rate applied
        self.lambda_ = 0.0  # Current rate of Poisson process
        self.x = 0  # Current neuron index or time counter for the next spike
        self.x_offset = x_offset  # Offset for neuron index
    
    def exponential_random(self, scale=1.0):
        """
        Generates a random number from an exponential distribution.
        
        Parameters:
        - scale (float): The scale of the exponential distribution, 1/lambda.
        
        Returns:
        - (float): A random number from the exponential distribution.
        """
        return np.random.exponential(scale)

    def set_rate(self, rate):
        """
        Sets the rate of the Poisson process and computes the next spike time.

        Parameters:
        - rate (float): The firing rate in Hz.
        """
        self.lambda_ = 1.0 / (1.0 / rate - self.dt)
        if rate > 0.0:
            r = self.exponential_random() / self.lambda_
            self.x = int(r / self.dt + 0.5)
        else:
            self.x = int(6e9)  # Use a very high number to effectively disable spiking

    def poisson_evolve(self, cur_clock):
        """
        Generates spikes according to Poisson statistics until the current simulation time.

        Parameters:
        - cur_clock (int): The current simulation clock or time step.
        """
        while self.x < self.neurons:
            self.spike_times.append(cur_clock * self.dt)
            self.spike_ids.append(self.x + self.x_offset)
            r = self.exponential_random() / self.lambda_
            self.x += int(r / self.dt + 1.5)
        
        self.x -= self.neurons

    def evolve(self, cur_clock):
        """
        Updates the firing rate based on the input signals file and generates spikes for the current time step.

        Parameters:
        - cur_clock (int): The current simulation clock or time step.
        """
        # Advance in time and update rates from the file
        while self.ftime < cur_clock and self.signals_idx < len(self.signals):
            self.ltime = self.ftime
            self.rate_n = self.lambda_
            
            t, r = self.signals[self.signals_idx]
            self.signals_idx += 1
            self.ftime = int(t / self.dt + 0.5) + self.ftime_offset

            if self.ftime < cur_clock or (self.signals_idx == len(self.signals) and not self.loop):
                self.rate_m = 0.0
                self.rate_n = r
                self.set_rate(r)
            else:
                self.rate_m = (r - self.rate_n) / (self.ftime - self.ltime)

        # Handle looping of the signal file
        if self.signals_idx == len(self.signals) and self.loop:
            self.ftime_offset = self.ftime
            self.signals_idx = 0

        # Update rate based on the current time and generate spikes
        rate = self.rate_m * (cur_clock - self.ltime) + self.rate_n
        if self.last_rate != rate:
            self.set_rate(rate)
        
        if rate > 0.0:
            self.poisson_evolve(cur_clock)
        
        self.last_rate = rate

    def mk_spikes(self, rates_file_path_list, spikes_file_path, time=3600, neurons=100):
        """
        Generates spikes for all neurons over a specified duration based on input rate files.

        Parameters:
        - rates_file_path_list (list): List of file paths containing the rate changes.
        - spikes_file_path (str): File path to save the generated spikes.
        - time (float): Total time in seconds for the simulation.
        - neurons (int): Number of neurons for which to generate spikes.
        """
        self.neurons = neurons
        self.clock = int(time / self.dt)
        self.spike_times = []
        self.spike_ids = []

        # Process each rate file
        for rates_file_path in rates_file_path_list:
            self.signals = np.loadtxt(rates_file_path)
            self.signals_idx = 0
            self.init(x_offset=self.neurons * rates_file_path_list.index(rates_file_path))

            for cur_clock in range(self.clock):
                self.evolve(cur_clock)
        
        # Write spikes to file
        spkfile = open(spikes_file_path, 'w')
        # Sort spikes for writing to file
        idx = np.argsort(np.array(self.spike_times))
        for t, i in zip(np.array(self.spike_times)[idx], np.array(self.spike_ids)[idx]):
            spkfile.write("%e %i\n" % (t, i))
        spkfile.close()

    
    
class LPLDataset(Dataset):
    """
    Dataset class to handle spike train data formatted in RAS (Raster) format,
    organizing the spikes into manageable segments based on specified time windows.
    """

    def __init__(self, data_np, time_step=2e-3, time_window=1.0, device=torch.device("cpu")):
        """
        Initializes the dataset with the given spike data, time step resolution, and time window size.

        Parameters:
        - data_np (np.ndarray): NumPy array containing the spike data. Each row is a spike event,
                                with the first column for timestamps and the second for neuron IDs.
        - time_step (float): Duration of each discrete step within the time window, used to
                             convert spike times from seconds to more granular time steps.
        - time_window (float): Duration of the time window over which the data is segmented.
        - device (torch.device): The computing device (CPU or GPU) where the data will be processed.
        """
        data = torch.from_numpy(data_np)
        data = data.to(device=device)
        self.data = self.split_by_time(data, time_step, time_window)
    
    def split_by_time(self, data, time_step, time_window):
        """
        Organizes data into groups based on the specified time window.

        Parameters:
        - data (torch.Tensor): The spike data as a PyTorch tensor.
        - time_step (float): The interval of each time step.
        - time_window (float): The size of each time window for grouping spikes.

        Returns:
        - list: A list of tensors, each representing spike data for a specific time window.
        """
        group_indices = (data[:, 0] / time_window).floor().int()
        groups = []
        for group_index in group_indices.unique():
            mask = (group_indices == group_index)
            temp = data[mask]
            # Adjust timestamps within each group relative to the start of the time window and convert to time steps.
            temp[:, 0] = (temp[:, 0] - time_window * group_index) / time_step
            groups.append(temp.to(torch.int64))
        return groups

    def __len__(self):
        """
        Returns the number of time window groups in the dataset.
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Retrieves the data for a specific time window group.

        Parameters:
        - idx (int): Index of the time window group.

        Returns:
        - torch.Tensor: The tensor containing spike data for the specified time window.
        """
        return self.data[idx]


class SparseGenerator(DataGenerator):
    """
    A custom data loader generator class that organizes and batches data for neural network training or inference.
    This class specifically handles sparse data representations, converting them into a dense format or maintaining
    them in sparse format as needed.

    Attributes:
        nb_workers (int): The number of subprocesses to use for data loading.
        batch_tensor (torch.Tensor): Temporary storage for the current batch's data.
        time_indices (torch.Tensor): Extracted time indices from the batch data.
        neuron_indices (torch.Tensor): Extracted neuron indices from the batch data.
        indices (torch.Tensor): Combined indices for constructing a sparse tensor.
        values (torch.Tensor): Values associated with each index in the sparse tensor.
    """

    def __init__(self, nb_workers=1):
        """
        Initializes the SparseGenerator with the specified number of workers.

        Parameters:
            nb_workers (int): Number of workers for loading data.
        """
        self.nb_workers = nb_workers

    def collate_function(self, data):
        """
        Processes the data by collating an n x 2 tensor into a dense format.
        This method assumes that the first column of the data tensor contains time indices,
        and the second column contains neuron indices, both essential for constructing
        the sparse representation.

        Parameters:
            data (list of torch.Tensor): List containing batch data from a Dataset.

        Returns:
            torch.Tensor: A dense tensor representation of the input batch.
        """
        self.batch_tensor = data[0]
        self.time_indices = self.batch_tensor[:, 0]
        self.neuron_indices = self.batch_tensor[:, 1]

        # Determine dimensions based on max indices, ensuring all events are included
        nb_steps = self.nb_steps
        nb_neurons = self.nb_units

        # Stack time and neuron indices to create a coordinate format for the sparse tensor
        self.indices = torch.stack([self.time_indices.flatten(), self.neuron_indices.flatten()], dim=0)
        self.values = torch.ones(self.indices.shape[1], dtype=torch.float, device=self.device)

        # Construct a sparse tensor and then optionally convert it to a dense format
        sparse_tensor = torch.sparse.FloatTensor(self.indices, self.values, torch.Size([nb_steps, nb_neurons]))
        dense_tensor = sparse_tensor.to_dense()

        return dense_tensor.view(1, nb_steps, nb_neurons)

    def __call__(self, dataset, shuffle=False):
        """
        Creates a DataLoader to handle the batching of the dataset.

        Parameters:
            dataset (Dataset): The dataset to load data from.
            shuffle (bool): Whether to shuffle the data at every epoch.

        Returns:
            DataLoader: A configured DataLoader ready to yield batches of data.
        """
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.nb_workers,
            collate_fn=self.collate_function,
        )
