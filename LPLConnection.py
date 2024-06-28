import torch
import torch.nn as nn

import numpy as np

from stork.connections import Connection as Con


class LPLConnection(Con):
    """
    LPLConnection class implementation.
    
    Parameters:
    - src: Source group of neurons.
    - dst: Destination group of neurons.
    - lamda_: Lambda parameter, used to determine the application of the Hebbian rule.
    - phi: Phi parameter, used to determine the application of the Predictive rule.
    - tau_mem: Membrane time constant.
    - tau_syn: Synaptic time constant.
    - tau_post_mean: Time constant for the post-synaptic mean.
    - tau_var: Variance time constant.
    - tau_el_rise: EL rise time constant.
    - tau_el_decay: EL decay time constant.
    - tau_rms: RMS time constant.
    - delta_time: Delta time, used in the Predictive rule.
    - delay_time: Delay in synaptic transmission
    - connection_prob: Probability of forming a connection between neurons.
    - initial_weight: Initial weight value for connections.
    - timestep_rmsprop_updates: Time step for RMSProp updates.
    - lr: Learning rate.
    - delta: Delta parameter for weight updates.
    - epsilon: Epsilon parameter for numerical stability.
    - operation: Neural network operation (default is nn.Linear).
    - evolved: Boolean indicating if the connection evolves over time.
    - target: Target layer or output.
    - bias: Boolean indicating if the operation includes a bias term.
    - requires_grad: Boolean indicating if gradients are required for the weights.
    - propagate_gradients: Boolean indicating if gradients should be propagated through the network.
    - flatten_input: Boolean indicating if the input should be flattened.
    - name: Name of the connection.
    - regularizers: Regularizers applied to the connection.
    - constraints: Constraints applied to the connection.
    - **kwargs: Additional arguments.
    """

    def __init__(
        self,
        src,
        dst,
        lamda_=1.0,
        phi=1.0,
        tau_mem=20e-3,
        tau_syn=5e-3,
        tau_post_mean=600,
        tau_var=30,
        tau_el_rise=2e-3,
        tau_el_decay=10e-3,
        tau_rms=100.0,
        delta_time=20e-3,
        connection_prob=0.1,
        initial_weight=0.15,
        timestep_rmsprop_updates=5000,
        lr=1e-2,
        beta=1.0/1e-3,
        delta=1e-5,
        delay_time=8e-4,
        epsilon=1e-3,
        gamma=1e-7,
        operation=nn.Linear,
        evolved=False,
        target=None,
        bias=False,
        requires_grad=False,
        propagate_gradients=False,
        flatten_input=False,
        name="LPLConnection",
        regularizers=None,
        constraints=None,
        **kwargs
    ):
        super(LPLConnection, self).__init__(
            src,
            dst,
            name=name,
            target=target,
            regularizers=regularizers,
            constraints=constraints,
        )

        self.lamda_ = lamda_
        self.phi = phi
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.tau_post_mean = tau_post_mean
        self.tau_var = tau_var

        self.connection_prob = connection_prob
        self.initial_weight = initial_weight
        self.tau_el_rise = tau_el_rise
        self.tau_el_decay = tau_el_decay
        self.tau_vrd_rise = self.tau_el_rise
        self.tau_vrd_decay = self.tau_el_decay
        self.timestep_rmsprop_updates = timestep_rmsprop_updates
        self.tau_rms = tau_rms


        self.lr = lr
        self.beta = beta
        self.delta = delta
        self.epsilon = epsilon
        self.delta_time = delta_time
        self.delay_time = delay_time
        self.gamma = gamma

        self.w_val = None
        self.el_val = None
        self.el_val_flt = None
        self.el_sum = None
        self.w_grad2 = None

        self.trace_post = None
        self.trace_post_mean = None
        self.trace_post_sigma2 = None
        self.trace_pre = None
        self.trace_pre_psp = None
        self.err = None
        self.trace_err = None
        self.trace_err_flt = None
        self.partial = None
        self.evolved = evolved

        self.requires_grad = requires_grad
        self.propagate_gradients = propagate_gradients
        self.flatten_input = flatten_input
        if flatten_input:
            self.op = operation(src.nb_units, dst.shape[0], bias=bias, **kwargs)
        else:
            self.op = operation(src.shape[0], dst.shape[0], bias=bias, **kwargs)
        for param in self.op.parameters():
            param.requires_grad = requires_grad

    def init_weight(self):
        prob_tensor = torch.rand(self.op.weight.shape)
        self.op.weight.data[prob_tensor < self.connection_prob] = self.initial_weight
        self.op.weight.data[prob_tensor >= self.connection_prob] = 0.0
    
    def configure(self, batch_size, nb_steps, time_step, device, dtype):
        super().configure(batch_size, nb_steps, time_step, device, dtype)
        self.init_weight()
        self.zero_mask = (self.op.weight.data == 0.0)

        self.delay_post_len = int(self.delta_time/time_step) + 1
        self.delay_post = []
        for i in range(self.delay_post_len):
            tmp = torch.zeros_like(self.dst.out, dtype=dtype, device=device)
            self.delay_post.append(tmp)
        self.store_idx = 0
        self.choose_idx = 1




        self.delay_list_len = int(self.delay_time / time_step) + 1
        self.delay_list = []
        for i in range(self.delay_list_len):
            tmp1 = torch.zeros_like(self.src.out, dtype=dtype, device=device)
            tmp2 = torch.zeros_like(self.dst.out, dtype=dtype, device=device)
            self.delay_list.append([tmp1, tmp2])
        self.delay_list_store_idx = 0
        self.delay_lsit_choose_idx = 1      

        self.delay_pre = []
        for i in range(self.delay_list_len):
            tmp = torch.zeros_like(self.src.out, dtype=dtype, device=device)
            self.delay_pre.append(tmp)
        self.delay_pre_store_idx = 0
        self.delay_pre_choose_idx = 1


        self.trace_pre = torch.zeros_like(self.src.out, dtype=dtype, device=device)
        self.trace_pre_psp = torch.zeros_like(self.src.out, dtype=dtype, device=device)
        self.trace_post = torch.zeros_like(self.dst.out, dtype=dtype, device=device)
        self.trace_post_mean = torch.zeros_like(self.dst.out, dtype=dtype, device=device)
        self.trace_post_sigma2 = torch.zeros_like(self.dst.out, dtype=dtype, device=device)

        self.err = torch.zeros_like(self.dst.out, dtype=dtype, device=device)
        self.trace_err = torch.zeros_like(self.dst.out, dtype=dtype, device=device)
        self.trace_err_flt = torch.zeros_like(self.dst.out, dtype=dtype, device=device)
        self.partial = torch.zeros_like(self.dst.out, dtype=dtype, device=device)

        self.w_val = torch.zeros_like(self.op.weight.data, dtype=dtype, device=device)
        self.el_val = torch.zeros_like(self.op.weight.data, dtype=dtype, device=device)
        self.el_val_flt = torch.zeros_like(self.op.weight.data, dtype=dtype, device=device)
        self.el_sum =  torch.zeros_like(self.op.weight.data, dtype=dtype, device=device)      
        self.w_grad2 = torch.zeros_like(self.op.weight.data, dtype=dtype, device=device)

        tau_mem = torch.tensor(self.tau_mem, device=device, dtype=dtype)
        tau_syn = torch.tensor(self.tau_syn, device=device, dtype=dtype)
        tau_var = torch.tensor(self.tau_var, device=device, dtype=dtype)
        tau_post = torch.tensor(100e-3, device=device, dtype=dtype)
        tau_post_mean = torch.tensor(self.tau_post_mean, device=device, dtype=dtype)
        tau_vrd_decay = torch.tensor(self.tau_vrd_decay, device=device, dtype=dtype)
        tau_vrd_rise = torch.tensor(self.tau_vrd_rise, device=device, dtype=dtype)
        tau_el_decay = torch.tensor(self.tau_el_decay, device=device, dtype=dtype)
        tau_el_rise = torch.tensor(self.tau_el_rise, device=device, dtype=dtype)
        tau_rms = torch.tensor(self.tau_rms, device=device, dtype=dtype)
        
        self.follow_el_val = time_step / tau_el_rise
        self.follow_el_val_decay = time_step / tau_el_decay
        self.follow_pre_psp = time_step / tau_mem
        self.follow_sigma2 = time_step / tau_var
        self.follow_post_mean = time_step / tau_post_mean
        self.follow_trace_err = time_step / tau_vrd_rise
        self.follow_trace_err_flt = time_step / tau_vrd_decay
        
        self.scale_el_val = torch.exp(-1.0 * time_step / tau_el_rise)
        self.scl_pre = torch.exp(-1.0 * time_step / tau_syn)
        self.scl_sigma2 = torch.exp(-1.0 * time_step / tau_var)
        self.scl_post = torch.exp(-1.0 * time_step / tau_post)
        self.scl_post_mean = torch.exp(-1.0 * time_step / tau_post_mean)

        self.rms_mul = torch.exp(-1.0 * time_step * self.timestep_rmsprop_updates / tau_rms)

        self.a = torch.ones_like(self.src.out, device=self.device, dtype=self.dtype)

    def set_evolve(self, evolved=True):
        self.evolved = evolved
    

    def compute_err(self):
        """
        Compute the error for the connection based on the destination output and other parameters.
        """
        err = self.dst.out - self.follow_post_mean * self.trace_post_mean 
        tmp = self.trace_post_sigma2 + self.epsilon
        err = self.lamda_ * (err.div(tmp)) + self.delta
        self.err = err - self.phi * self.dst.out + self.phi * self.delay_post[self.choose_idx]

        self.delay_post[self.store_idx] = self.dst.out
        self.store_idx = self.choose_idx
        self.choose_idx = (self.choose_idx + 1) % self.delay_post_len


    def instantaneous_partial(self):
        """
        Compute the instantaneous partial derivative of the connection.
        """
        h = (self.dst.mem - self.dst.thr_rest) * self.beta
        self.partial = self.beta / (1.0 + h.abs()).pow(2)
        idx = (self.dst.mem < -80e-3)
        self.partial[idx] = 0.0

    def forward(self):
        preact = self.delay_pre[self.delay_pre_choose_idx]
        out = self.op(preact)
        self.dst.add_to_state(self.target, out)
    
    def process_plasticity(self):
        """
        Process the plasticity of the connection.
        """
        psp = self.delay_list[self.delay_lsit_choose_idx][0]
        psp[psp <= self.gamma] = 0.0
        sigma_prime = self.delay_list[self.delay_lsit_choose_idx][1]
        self.el_val = self.el_val + sigma_prime.T @ psp
        self.el_val_flt = self.el_val_flt + self.follow_el_val_decay * (self.el_val - self.el_val_flt)
        self.el_val = self.scale_el_val * self.el_val

        self.instantaneous_partial()
        self.delay_list[self.delay_list_store_idx][0] = self.trace_pre_psp
        self.delay_list[self.delay_list_store_idx][1] = self.partial
        self.delay_list_store_idx = self.delay_lsit_choose_idx
        self.delay_lsit_choose_idx = (self.delay_lsit_choose_idx + 1) % self.delay_list_len

        self.trace_err_flt[self.trace_err_flt <= self.gamma] = 0.0
        self.el_sum += self.el_val_flt * (self.trace_err_flt.T @ self.a)

    def evolve(self):       
        """
        Evolve the state of the connection if the evolved flag is set.
        """
        if self.evolved:
        
            self.compute_err()

            self.process_plasticity()
            
            # evolve_trace
            self.trace_pre = self.scl_pre * (self.trace_pre + self.delay_pre[self.delay_pre_choose_idx])
            self.trace_post_mean = self.scl_post_mean * (self.trace_post_mean + self.dst.out)

            # evolve state vector
            self.trace_pre_psp = self.trace_pre_psp + self.follow_pre_psp * (self.trace_pre - self.trace_pre_psp)
            self.trace_err = self.trace_err + self.follow_trace_err * (self.err - self.trace_err)
            self.trace_err_flt = self.trace_err_flt + self.follow_trace_err_flt * (self.trace_err -  self.trace_err_flt)

            tmp = self.dst.out - self.follow_post_mean * self.trace_post_mean 
            tmp = torch.square(tmp)
            self.trace_post_sigma2 = self.scl_sigma2 * (self.trace_post_sigma2 + (tmp / self.tau_var))
            

    def update_weight(self):
        """
        Update the weights of the connection using RMSProp.
        """        
        grad = self.el_sum / self.timestep_rmsprop_updates
        self.w_grad2 = torch.max(torch.square(grad), self.rms_mul * self.w_grad2)
        gm = self.w_grad2
        rms_scale = 1.0 / (torch.sqrt(gm) + self.epsilon)
        self.op.weight.data += rms_scale * grad * self.lr
        self.op.weight.data[self.zero_mask] = 0.0
        self.el_sum = self.el_sum * 0.0


    def propagate(self):
        """
        Propagate the state through the connection by performing forward pass and evolving.
        """
        self.forward()
        self.evolve()
        self.delay_pre[self.delay_pre_store_idx] = self.src.out.detach()
        self.delay_pre_store_idx = self.delay_pre_choose_idx
        self.delay_pre_choose_idx = (self.delay_pre_choose_idx + 1) % self.delay_list_len           



class Connection(Con):

    """
    Defines a synaptic connection between two neuron groups with configurable properties,
    including random initialization with uniform or Gaussian distributions.

    Attributes:
    - src (CellGroup): Source neuron group providing the input.
    - dst (CellGroup): Destination neuron group receiving the input.
    - gaussian (bool): Flag to determine if Gaussian initialization is used.
    - connection_prob (float): Probability of connection between neurons.
    - initial_weight (float): Initial weight for connections that are established.
    - delay_time(float): Delay in synaptic transmission
    - sigma (float): Standard deviation for Gaussian distribution used in weight initialization.
    - operation (nn.Module): PyTorch module defining the transformation applied to input data.
    - target (str): Target state in the destination group that the operation affects.
    - bias (bool): Flag to include bias in the operation.
    """

    def __init__(
        self,
        src,
        dst,
        gaussian=False,
        connection_prob=0.5,
        initial_weight=0.1,
        sigma=20,
        delay_time=8e-4,
        operation=nn.Linear,
        target=None,
        bias=False,
        requires_grad=False,
        propagate_gradients=False,
        flatten_input=False,
        name="Connection",
        regularizers=None,
        constraints=None,
        **kwargs
    ):
        super(Connection, self).__init__(
            src,
            dst,
            name=name,
            target=target,
            regularizers=regularizers,
            constraints=constraints,
        )

        self.gaussian=gaussian
        self.sigma=sigma
        self.connection_prob = connection_prob
        self.initial_weight = initial_weight

        self.delay_time = delay_time

        self.requires_grad = requires_grad
        self.propagate_gradients = propagate_gradients
        self.flatten_input = flatten_input

        

        if flatten_input:
            self.op = operation(src.nb_units, dst.shape[0], bias=bias, **kwargs)
        else:
            self.op = operation(src.shape[0], dst.shape[0], bias=bias, **kwargs)
        for param in self.op.parameters():
            param.requires_grad = requires_grad

    def init_weight(self):
        """
        Initializes the weights of the connection randomly based on the connection probability
        and the specified initial weight.
        """        
        prob_tensor = torch.rand(self.op.weight.shape)
        self.op.weight.data[prob_tensor < self.connection_prob] = self.initial_weight
        self.op.weight.data[prob_tensor >= self.connection_prob] = 0.0

    def gaussian_con(self, sigma):
        """
        Initializes weights using a Gaussian distribution centered around each post-synaptic neuron.
        """        
        nb_post, nb_pre = self.op.weight.data.shape
        mat = torch.empty(nb_pre, nb_post, device=self.device, dtype=self.dtype)
        centers = torch.linspace(0, nb_pre, nb_post,  device=self.device, dtype=self.dtype)
        
        x = torch.arange(nb_pre, device=self.device, dtype=self.dtype)
        for i, c in enumerate(centers):
            mat[:, i] = torch.exp(-(x - c).pow(2) / sigma.pow(2))
 
        rnd = torch.rand_like(mat)
        con = torch.where(mat > rnd, torch.ones_like(mat), torch.zeros_like(mat))
        self.op.weight.data = con.T * self.initial_weight

    
    def configure(self, batch_size, nb_steps, time_step, device, dtype):
        super().configure(batch_size, nb_steps, time_step, device, dtype)
        
        if self.gaussian:
            sigma = torch.tensor(self.sigma, device=device, dtype=dtype)
            self.gaussian_con(sigma)
        else:
            self.init_weight()

        self.delay_list_len = int(self.delay_time / time_step)  + 1
        self.delay_pre = []
        for i in range(self.delay_list_len):
            tmp = torch.zeros_like(self.src.out, dtype=dtype, device=device)
            self.delay_pre.append(tmp)
        self.delay_pre_store_idx = 0
        self.delay_pre_choose_idx = 1

    def forward(self):
        preact = self.delay_pre[self.delay_pre_store_idx]
        out = self.op(preact)
        self.dst.add_to_state(self.target, out)

    def update_weight(self):
        pass

    def propagate(self):
        self.forward()
        self.delay_pre[self.delay_pre_store_idx] = self.src.out
        self.delay_pre_store_idx = self.delay_pre_choose_idx
        self.delay_pre_choose_idx = (self.delay_pre_choose_idx + 1) % self.delay_list_len      


class SymmetricSTDPConnection(Con):
    """
    Establishes a connection between neural groups with symmetric spike-timing-dependent plasticity (STDP).
    STDP modifies the synaptic strength based on the timing of pre- and post-synaptic spikes.

    Attributes:
        src (CellGroup): Source neuron group providing pre-synaptic spikes.
        dst (CellGroup): Destination neuron group receiving post-synaptic spikes.
        connection_prob (float): Probability of connection between neurons in the source and destination groups.
        initial_weight (float): Initial synaptic weight for connected neuron pairs.
        tau_stdp (float): Time constant of the STDP function, affecting the decay rate of synaptic changes.
        lr (float): Learning rate for STDP updates.
        kappa (float): Scaling factor for the target threshold in STDP calculations.
        delay_time: Delay in synaptic transmission
        operation (nn.Module): PyTorch module defining the transformation applied to input data.
        target (str, optional): Target state in the destination group that the operation affects.
        bias (bool): Flag to include bias in the operation.
        requires_grad (bool): Specifies if gradients should be calculated for this connection.
        propagate_gradients (bool): Allows toggling gradient propagation through the network.
        flatten_input (bool): Determines if input tensor should be flattened before applying the operation.
        name (str): Identifier for the connection.
        regularizers (list, optional): List of regularization functions applied to the connection weights.
        constraints (list, optional): List of constraints applied to the connection weights.
        **kwargs: Additional keyword arguments for the operation module.
    """

    def __init__(
        self,
        src,
        dst,
        connection_prob=0.1,
        initial_weight=0.15,
        tau_stdp=20e-3,
        lr=1e-2,
        kappa=10,
        delay_time=8e-4,
        operation=nn.Linear,
        target=None,
        bias=False,
        requires_grad=False,
        propagate_gradients=False,
        flatten_input=False,
        name="STDPConnection",
        regularizers=None,
        constraints=None,
        **kwargs
    ):
        super(SymmetricSTDPConnection, self).__init__(
            src,
            dst,
            name=name,
            target=target,
            regularizers=regularizers,
            constraints=constraints,
        )

        self.conneciton_prob = connection_prob
        self.initial_weight = initial_weight

        self.tau_stdp = tau_stdp

        self.lr = lr
        self.kappa = kappa
        self.trace_pre = None
        self.trace_post = None
        
        self.delay_time = delay_time

        self.requires_grad = requires_grad
        self.propagate_gradients = propagate_gradients
        self.flatten_input = flatten_input

        self.evolved = False

        if flatten_input:
            self.op = operation(src.nb_units, dst.shape[0], bias=bias, **kwargs)
        else:
            self.op = operation(src.shape[0], dst.shape[0], bias=bias, **kwargs)
        for param in self.op.parameters():
            param.requires_grad = requires_grad


    def init_weight(self):
        prob_tensor = torch.rand(self.op.weight.shape)
        self.op.weight.data[prob_tensor < self.conneciton_prob] = self.initial_weight
        self.op.weight.data[prob_tensor >= self.conneciton_prob] = 0.0

    
    def configure(self, batch_size, nb_steps, time_step, device, dtype):
        self.init_weight()
        self.zero_mask = (self.op.weight.data == 0.0)
        self.trace_pre = torch.zeros_like(self.src.out, device=device, dtype=dtype)
        self.trace_post = torch.zeros_like(self.dst.out, device=device, dtype=dtype)
        
        tau_stdp = torch.tensor(self.tau_stdp, device=device, dtype=dtype)
        target = torch.tensor(self.kappa, device=device, dtype=dtype)
        self.scl_stdp = torch.exp(-time_step/tau_stdp)
        self.dcy_stdp = 1.0 - self.scl_stdp
        self.kappa_fudge = 2 * target * tau_stdp

        self.delay_list_len = int(self.delay_time / time_step)  + 1
        self.delay_pre = []
        for i in range(self.delay_list_len):
            tmp = torch.zeros_like(self.src.out, dtype=dtype, device=device)
            self.delay_pre.append(tmp)
        self.delay_pre_store_idx = 0
        self.delay_pre_choose_idx = 1

        super().configure(batch_size, nb_steps, time_step, device, dtype)


    
    def set_evolve(self, evolved=True):
        """
        Toggles whether the connection's weight updates are active, allowing for runtime changes to plasticity.
        """
        self.evolved = evolved



    def evolve(self):
        """
        Updates synaptic weights based on the STDP rule, applying changes based on pre and post synaptic traces.
        """
        if self.evolved:
            dw_pre = self.trace_post - self.kappa_fudge
            dw_post = self.trace_pre
            self.op.weight.data += self.lr * (dw_pre.T @ self.src.out + self.dst.out.T @ dw_post)
            self.op.weight.data[self.zero_mask] = 0.0

            self.trace_pre = self.scl_stdp * (self.trace_pre + self.delay_pre[self.delay_pre_choose_idx])
            self.trace_post = self.scl_stdp * (self.trace_post + self.dst.out)

    def forward(self):
        preact = self.delay_pre[self.delay_pre_choose_idx]
        out = self.op(preact)
        self.dst.add_to_state(self.target, out)

    def update_weight(self):
        pass

    def propagate(self):
        self.forward()
        self.evolve()
        self.delay_pre[self.delay_pre_store_idx] = self.src.out
        self.delay_pre_store_idx = self.delay_pre_choose_idx
        self.delay_pre_choose_idx = (self.delay_pre_choose_idx + 1) % self.delay_list_len




