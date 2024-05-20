import numpy as np

import torch
import torch.nn as nn

import stork.nodes.base
import stork.generators as generators
import stork.utils as utils

class LPLModel(nn.Module):
    """
    Represents a large-scale neural network model where individual components (groups and connections)
    can be added and configured dynamically. This model supports both dense and sparse input modalities,
    and can handle data generation, state management, and execution of dynamics across multiple devices.

    Attributes:
        batch_size (int): Number of samples per batch.
        nb_time_steps (int): Number of time steps to simulate.
        nb_inputs (int): Number of input features or neurons.
        device (torch.device): The device (CPU, GPU) on which to perform computations.
        dtype (torch.dtype): Data type for tensors (e.g., torch.float).
        sparse_input (bool): If True, expects sparse input data.
        groups (list): Holds instances of neuron groups in the model.
        connections (list): Holds the connections between neuron groups.
        devices (list): Devices used for computation.
        input_group (CellGroup): Special group to manage input data.
        stored (bool): Flag indicating if outputs should be stored.
        store_out (dict): Dictionary to store outputs of specified groups.
        store_weight (dict): Dictionary to store weights of specified connections.
        map (dict): Maps names to groups or connections for quick reference.
        filename (str): Base filename for storing outputs.
    """
    
    def __init__(
        self, 
        batch_size,
        nb_time_steps,
        nb_inputs,
        device=torch.device("cpu"),
        dtype=torch.float,
        sparse_input=False,
    ):
        super(LPLModel, self).__init__()
        self.batch_size = batch_size
        self.nb_time_steps = nb_time_steps
        self.nb_inputs = nb_inputs

        self.device = device
        self.dtype = dtype

        self.groups = []
        self.connections = []
        self.devices = []

        self.input_group = None
        self.sparse_input = sparse_input

        self.stored = False
        self.store_out = {}
        self.store_weight = {}        
        self.map = {}
        self.filename = ""

    def configure(
        self,
        input,
        generator=None,
        time_step=2e-3,
    ):
        """
        Configures the model for simulation by setting up the input group, data generator, and time step.
        
        Parameters:
            input (CellGroup): The input group to which input data will be fed.
            generator (DataGenerator, optional): The data generator to use for creating input data.
            time_step (float): The simulation time step.
        """
        self.input_group = input
        self.time_step = time_step

        if generator is None:
            self.data_generator_ = generators.StandardGenerator()
        else:
            self.data_generator_ = generator

        self.data_generator_.configure(
            self.batch_size,
            self.nb_time_steps,
            self.nb_inputs,
            self.time_step,
            device=self.device,
            dtype=self.dtype,
        )

        for o in self.groups + self.connections:
            o.configure(
                self.batch_size,
                self.nb_time_steps,
                self.time_step,
                self.device,
                self.dtype,
            )

        self.to(self.device)


    def configure_stored(self, list_neurons, list_cons, filename=""):
        """
        Configures storage for specified neuron groups and connections.

        Parameters:
            list_neurons (list): Names of neuron groups whose outputs should be stored.
            list_cons (list): Names of connections whose weights should be stored.
            filename (str): Base filename for storing the outputs.
        """
        self.stored = True

        for g in self.groups:
            for t in list_neurons:
                if g.name == t:
                    self.map[t] = g
                    self.store_out[t] = []
        
        for c in self.connections:
            for t in list_cons:
                if c.name == t:
                    self.map[t] = c
                    self.store_weight[t] = []
        self.filename = filename


    def mk_store(self):
        """
        Stores outputs specified groups if storage is enabled.
        """
        if not self.stored:
            return
        
        for key in self.store_out.keys():
            out = self.map[key].out.detach().to(torch.device("cpu"))
            self.store_out[key].append(out)


    def prepare_data(self, dataset):
        return self.data_generator_.prepare_data(dataset)

    def data_generator(self, dataset, shuffle=False):
        return self.data_generator_(dataset, shuffle=shuffle)

    def add_group(self, group):
        """
        Adds a neuron group to the model.

        Parameters:
            group (CellGroup): The neuron group to add.
        
        Returns:
            CellGroup: The added neuron group.
        """
        self.groups.append(group)
        self.add_module("group%i" % len(self.groups), group)
        return group

    def add_connection(self, con):
        """
        Adds a connection between neuron groups to the model.

        Parameters:
            con (Connection): The connection to add.
        
        Returns:
            Connection: The added connection.
        """
        self.connections.append(con)
        self.add_module("con%i" % len(self.connections), con)
        return con


    def reset_states(self, batch_size=None):
        """
        Resets the states of all groups in the model, typically done before starting a new simulation run.

        Parameters:
            batch_size (int, optional): The batch size for the reset, can differ from initial configuration.
        """
        for g in self.groups:
            g.reset_state(batch_size)

    def evolve_all(self):
        """
        Evolves the state of all groups in the model, typically by one time step.
        """
        for g in self.groups:
            g.evolve()
            g.clear_input()


    def propagate_all(self):
        """
        Propagates signals through all connections in the model.
        """
        for c in self.connections:
            c.propagate()
            
    def apply_constraints(self):
        """
        Applies constraints to all connections, typically after a simulation step.
        """
        for c in self.connections:
            c.apply_constraints()

    def update_all(self):
        for c in self.connections:
            c.update_weight()

    def execute_all(self):
        for d in self.devices:
            d.execute()
        

    def run(self, x_batch):
        """
        Runs a simulation for the given input batch.

        Parameters:
            x_batch: The input data batch to feed into the model.
        """
        self.input_group.feed_data(x_batch)
        for t in range(self.nb_time_steps):
            stork.nodes.base.CellGroup.clk = t
            self.evolve_all()
            self.propagate_all()
            self.execute_all()
            self.mk_store()


    def forward_pass(self, x_batch):
        """
        Conducts a forward pass through the model using the specified input batch.

        Parameters:
            x_batch: The input data batch.
        """
        self.run(x_batch)

    def get_example_batch(self, dataset, **kwargs):
        """
        Retrieves an example batch from the dataset using the data generator.

        Parameters:
            dataset: The dataset to sample from.
            **kwargs: Additional keyword arguments for the data generator.

        Returns:
            Batch: The generated data batch.
        """
        self.prepare_data(dataset)
        for batch in self.data_generator(dataset, **kwargs):
            return batch

    def stimulate(self, dataset, epoch=1, shuffle=False):
        """
        Stimulates the model with data from the dataset over a specified number of epochs.

        Parameters:
            dataset: The dataset to use for stimulation.
            epoch (int): Number of epochs to run the stimulation.
            shuffle (bool): Whether to shuffle the dataset.
        """
        print("stimulate start!")

        with torch.no_grad():
            self.prepare_data(dataset)
            for i in range(epoch):
                for local_X in self.data_generator(dataset, shuffle=shuffle):
                    self.forward_pass(local_X)
                    self.update_all()
                    self.apply_constraints()
                    
                print("   epoch %d end!"%(i))

        if self.stored:
            utils.write_to_file(self.store_out, "outs/out" + self.filename)
            print("   out%s have stored!"%(self.filename))
            for key in self.store_weight.keys():
                w = self.map[key].get_weights().data.detach().to(torch.device("cpu"))
                self.store_weight[key].append(w)
            utils.write_to_file(self.store_weight, "outs/weight" + self.filename)
            print("   weight%s have stored!"%(self.filename))

    def summary(self):
        """Print model summary"""

        print("\n# Model summary")
        print("\n## Groups")
        for group in self.groups:
            if group.name is None or group.name == "":
                print("no name, %s" % (group.shape,))
            else:
                print("%s, %s" % (group.name, group.shape))

        print("\n## Connections")
        for con in self.connections:
            print(con)
