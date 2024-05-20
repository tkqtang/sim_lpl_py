import torch
import torch.nn as nn

import numpy as np

from stork.connections import Connection as Con


class LPLConnection(Con):
    """
    Implements a connection with learning properties between source and destination neural groups.
    
    Attributes:
        src (CellGroup): Source neuron group.
        dst (CellGroup): Destination neuron group.
        lamda_ (float): Lambda parameter influencing hebb rule.
        phi (float): Phi parameter influencing pred rule.
        tau_mem (float): Membrane time constant.
        tau_syn (float): Synaptic time constant.
        tau_post_mean (float): Time constant for the moving average of the post-synaptic activity.
        tau_var (float): Time constant for variability of synaptic strength.
        tau_el_rise (float): Rise time for the eligibility trace.
        tau_el_decay (float): Decay time for the eligibility trace.
        tau_avg_err (float): Time constant for the averaging of error signals.
        delta_time (float): Time interval for pred rule.
        connection_prob (float): Probability of creating a synaptic connection.
        initial_weight (float): Initial weight of the synaptic connection.
        lr (float): Learning rate for synaptic updates.
        beta (float): Render the learning rule voltage dependent.
        delta (float): Ensure that weights of quiescent neurons slowly potentiate in the absence of activity to ultimately render them active.
        epsilon (float): Small constant to prevent division by zero in calculations.
        operation (torch.nn.Module): PyTorch module defining the operation to be applied at the connection.
        evolved (bool): Indicates if the connection is subject to evolution during the simulation.
        target (str, optional): Target property in the destination neurons to be affected.
        bias (bool): If true, adds a bias term to the operation.
        requires_grad (bool): If true, allows backpropagation through this connection.
        propagate_gradients (bool): If true, allows gradients to propagate through this connection for learning.
        flatten_input (bool): If true, flattens the input before applying the operation.
        name (str): Name identifier for the connection.
        regularizers (list): List of regularization functions to apply to the connection weights.
        constraints (list): List of constraints to apply to the connection weights.
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
        tau_avg_err=10.0,
        tau_rms=300,
        delta_time=20e-3,
        connection_prob=0.1,
        initial_weight=0.15,
        timestep_rmsprop_updates=5000,
        lr=1e-2,
        beta=1.0/1e-3,
        delta=1e-5,
        delay_time=8e-3,
        # gamma=1e-7,
        epsilon=1e-3,
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
        self.tau_avg_err = tau_avg_err
        self.timestep_rmsprop_updates = timestep_rmsprop_updates
        self.tau_rms = tau_rms


        self.lr = lr
        self.beta = beta
        self.delta = delta
        self.epsilon = epsilon
        self.delta_time = delta_time
        self.delay_time = delay_time


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
        
        self.store_post_out = []
        self.store_post_out_len = int(self.delta_time/time_step) + 1
        for i in range(self.store_post_out_len):
            tmp = torch.zeros_like(self.dst.out, dtype=dtype, device=device)
            self.store_post_out.append(tmp)
        self.store_idx = 0
        self.choose_idx = 1

        # self.delay_list = []
        # self.delay_list_len = int (self.delay_time / time_step) + 1
        # for i in range(self.delay_list_len):
        #     tmp1 = torch.zeros_like(self.src.out, dtype=dtype, device=device)
        #     tmp2 = torch.zeros_like(self.dst.out, dtype=dtype, device=device)
        #     self.delay_list.append([tmp1, tmp2])
        # self.delay_store_idx = 0
        # self.delay_choose_idx = 1

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


    def set_evolve(self, evolved=True):
        self.evolved = evolved
    

    def compute_err(self):
        self.err = 0.0
        self.err = self.dst.out - self.follow_post_mean * self.trace_post_mean 
        tmp = self.trace_post_sigma2 + self.epsilon
        self.err = self.lamda_ * (self.err.div(tmp)) + self.delta
        self.err =self.err - self.phi * self.dst.out + self.phi * self.store_post_out[self.choose_idx]

    def instantaneous_partial(self):

        h = (self.dst.mem - self.dst.thr_rest) * self.beta
        self.partial = self.beta / (1.0 + h.abs()).pow(2)

    def forward(self):
        preact = self.src.out
        if not self.propagate_gradients:
            preact = preact.detach()
        if self.flatten_input:
            shp = preact.shape
            preact = preact.reshape(shp[:1] + (-1,))

        out = self.op(preact)
        self.dst.add_to_state(self.target, out)
    

    # def evolve(self):
        
    #     self.instantaneous_partial()
    #     self.trace_pre_psp = self.trace_pre_psp + self.follow_pre_psp * (self.trace_pre - self.trace_pre_psp)
    #     self.trace_pre = self.scl_pre * (self.trace_pre + self.src.out)

    #     tmp = self.dst.out - self.follow_post_mean * self.trace_post_mean 
    #     tmp = torch.square(tmp)
    #     self.trace_post_sigma2 = self.scl_sigma2 * (self.trace_post_sigma2 + (tmp / self.tau_var))
        
        
    #     self.trace_post = self.scl_post * (self.trace_post + self.dst.out)
    #     self.trace_post_mean = self.scl_post_mean * (self.trace_post_mean + self.dst.out)
        
    #     self.trace_err_flt = self.trace_err_flt + self.follow_trace_err_flt * (self.trace_err -  self.trace_err_flt)
    #     self.trace_err = self.trace_err + self.follow_trace_err * (self.err - self.trace_err)
        
    #     self.compute_err()
    #     self.store_post_out[self.store_idx] = self.dst.out * 1.0
    #     self.choose_idx = (self.choose_idx + 1) % self.store_post_out_len
    #     self.store_idx = (self.store_idx + 1) % self.store_post_out_len

    #     self.el_val_flt = self.el_val_flt + self.follow_el_val_decay * (self.el_val - self.el_val_flt)
    #     self.el_val = self.scale_el_val * (self.el_val + self.partial.T * self.trace_pre_psp)

    
    #     if self.evolved:
    #         a = torch.ones_like(self.src.out, device=self.device, dtype=self.dtype)
    #         self.op.weight.data += self.lr * self.el_val_flt * (self.trace_err_flt.T @ a)

    
    # def evolve(self):       

    #     self.compute_err()
    #     self.store_post_out[self.store_idx] = self.dst.out * 1.0
    #     self.choose_idx = (self.choose_idx + 1) % self.store_post_out_len
    #     self.store_idx = (self.store_idx + 1) % self.store_post_out_len
    
    #     # process_plasticity
    #     self.el_val_flt = self.el_val_flt + self.follow_el_val_decay * (self.el_val - self.el_val_flt)
    #     self.el_val = self.el_val + self.follow_el_val * (self.el_val - self.partial.T @ self.trace_pre_psp)
    
    #     if self.evolved:
    #         a = torch.ones_like(self.src.out, device=self.device, dtype=self.dtype)
    #         self.op.weight.data += self.lr * self.el_val_flt * (self.trace_err_flt.T @ a) 

    #     tmp = self.dst.out - self.follow_post_mean * self.trace_post_mean 
    #     tmp = torch.square(tmp)
    #     self.trace_post_sigma2 = self.scl_sigma2 * (self.trace_post_sigma2 + (tmp / self.tau_var))
        
    #     self.trace_err_flt = self.trace_err_flt + self.follow_trace_err_flt * (self.trace_err -  self.trace_err_flt)
    #     self.trace_err = self.trace_err + self.follow_trace_err * (self.err - self.trace_err)

    #     self.trace_pre_psp = self.trace_pre_psp + self.follow_pre_psp * (self.trace_pre - self.trace_pre_psp)

    #     # evolve_trace
    #     self.trace_pre = self.scl_pre * (self.trace_pre + self.src.out)
    #     self.trace_post = self.scl_post * (self.trace_post + self.dst.out)
    #     self.trace_post_mean = self.scl_post_mean * (self.trace_post_mean + self.dst.out)

    def evolve(self):       

        if self.evolved:

            self.compute_err()
            self.store_post_out[self.store_idx] = self.dst.out.detach()
            self.choose_idx = (self.choose_idx + 1) % self.store_post_out_len
            self.store_idx = (self.store_idx + 1) % self.store_post_out_len
        
            # process_plasticity
            # self.instantaneous_partial()
            # self.delay_list[self.delay_store_idx][0] = self.trace_pre_psp.detach()
            # self.delay_list[self.delay_store_idx][1] = self.partial.detach()

            # psp = self.delay_list[self.delay_choose_idx][0]
            # sigma_prime = self.delay_list[self.delay_choose_idx][1]
            
            # self.delay_choose_idx = (self.delay_choose_idx + 1) % self.delay_list_len
            # self.delay_store_idx = (self.delay_store_idx + 1) % self.delay_list_len
            psp = self.trace_pre_psp
            sigma_prime = self.partial
            self.el_val = self.el_val + sigma_prime.T @ psp
            self.el_val_flt = self.el_val_flt + self.follow_el_val_decay * (self.el_val - self.el_val_flt)
            self.el_val = self.scale_el_val * self.el_val
        
            a = torch.ones_like(self.src.out, device=self.device, dtype=self.dtype)
            self.el_sum += self.el_val_flt * (self.trace_err_flt.T @ a)

            self.instantaneous_partial()        
            self.trace_pre_psp = self.trace_pre_psp + self.follow_pre_psp * (self.trace_pre - self.trace_pre_psp)

            tmp = self.dst.out - self.follow_post_mean * self.trace_post_mean 
            tmp = torch.square(tmp)
            self.trace_post_sigma2 = self.scl_sigma2 * (self.trace_post_sigma2 + (tmp / self.tau_var))
            
            self.trace_err_flt = self.trace_err_flt + self.follow_trace_err_flt * (self.trace_err -  self.trace_err_flt)
            self.trace_err = self.trace_err + self.follow_trace_err * (self.err - self.trace_err)
            
            # evolve_trace
            self.trace_pre = self.scl_pre * (self.trace_pre + self.src.out)
            self.trace_post_mean = self.scl_post_mean * (self.trace_post_mean + self.dst.out)

    def update_weight(self):
        grad = self.el_sum / self.timestep_rmsprop_updates
        self.w_grad2 = torch.max(torch.square(grad), self.rms_mul * self.w_grad2)
        gm = self.w_grad2
        rms_scale = 1.0 / (torch.sqrt(gm) + self.epsilon)
        self.op.weight.data += rms_scale * grad * self.lr
        self.el_sum = self.el_sum * 0.0


    def propagate(self):
        self.forward()
        self.evolve()
        

class LPLConnection2(Con):
    """
    Compared to the previous one, this class does not consider filtering
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
        tau_avg_err=10.0,
        delta_time=20e-3,
        connection_prob=0.1,
        initial_weight=0.15,
        lr=1e-2,
        beta=1.0/1e-3,
        delta=1e-5,
        # gamma=1e-7,
        epsilon=1e-3,
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
        super(LPLConnection2, self).__init__(
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
        self.tau_avg_err = tau_avg_err

        self.lr = lr
        self.beta = beta
        self.delta = delta
        self.epsilon = epsilon
        
        self.delta_time = delta_time
        self.store_post_out = []

        self.el_val = None
        self.el_val_flt = None
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
        
        self.store_post_out_len = int(self.delta_time/time_step)
        for i in range(self.store_post_out_len):
            tmp = torch.zeros_like(self.dst.out, dtype=dtype, device=device)
            self.store_post_out.append(tmp)

        self.store_idx = 0
        self.choose_idx = 1

        self.trace_pre = torch.zeros_like(self.src.out, dtype=dtype, device=device)
        self.trace_pre_psp = torch.zeros_like(self.src.out, dtype=dtype, device=device)
        self.trace_post = torch.zeros_like(self.dst.out, dtype=dtype, device=device)
        self.trace_post_mean = torch.zeros_like(self.dst.out, dtype=dtype, device=device)
        self.trace_post_sigma2 = torch.zeros_like(self.dst.out, dtype=dtype, device=device)
        self.tmp = torch.zeros_like(self.dst.out, dtype=dtype, device=device)

        self.err = torch.zeros_like(self.dst.out, dtype=dtype, device=device)
        self.trace_err = torch.zeros_like(self.dst.out, dtype=dtype, device=device)
        self.trace_err_flt = torch.zeros_like(self.dst.out, dtype=dtype, device=device)
        self.partial = torch.zeros_like(self.dst.out, dtype=dtype, device=device)

        self.el_val = torch.zeros_like(self.op.weight.data, dtype=dtype, device=device)
        self.el_val_flt = torch.zeros_like(self.op.weight.data, dtype=dtype, device=device)
        

        tau_mem = torch.tensor(self.tau_mem, device=device, dtype=dtype)
        tau_syn = torch.tensor(self.tau_syn, device=device, dtype=dtype)
        tau_var = torch.tensor(self.tau_var, device=device, dtype=dtype)
        tau_post = torch.tensor(100e-3, device=device, dtype=dtype)
        tau_post_mean = torch.tensor(self.tau_post_mean, device=device, dtype=dtype)
        tau_vrd_decay = torch.tensor(self.tau_vrd_decay, device=device, dtype=dtype)
        tau_vrd_rise = torch.tensor(self.tau_vrd_rise, device=device, dtype=dtype)
        tau_el_decay = torch.tensor(self.tau_el_decay, device=device, dtype=dtype)
        tau_el_rise = torch.tensor(self.tau_el_rise, device=device, dtype=dtype)

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


    def set_evolve(self, evolved=True):
        self.evolved = evolved
    

    def compute_err(self):
       
        self.err = self.dst.out - self.follow_post_mean * self.trace_post_mean 
        tmp = self.trace_post_sigma2 + self.epsilon
        self.err = self.lamda_ * self.err.div(tmp) + self.delta
        self.err = self.err - self.phi * self.dst.out + self.phi * self.store_post_out[self.choose_idx]
    

    def instantaneous_partial(self):

        self.partial = self.beta / (1.0 + self.beta * (self.dst.mem - self.dst.thr_rest).abs()).pow(2)

    def forward(self):
        preact = self.src.out
        if not self.propagate_gradients:
            preact = preact.detach()
        if self.flatten_input:
            shp = preact.shape
            preact = preact.reshape(shp[:1] + (-1,))

        out = self.op(preact)
        self.dst.add_to_state(self.target, out)
    
    def update_weight(self):
        pass

    def evolve(self):
        
        if self.evolved:
            # a = torch.ones_like(self.src.out, device=self.device, dtype=self.dtype)
            # b = torch.ones_like(self.dst.out, device=self.device, dtype=self.dtype)
            # b = torch.ones_like(self.dst.out, device=self.device, dtype=self.dtype)
            # self.op.weight.data += self.lr * (self.partial.T @ a) * (self.err.T @ a) + self.lr * (self.delta * self.b.T * self.src.out)
            self.op.weight.data += self.lr * (self.trace_err_flt.T @ self.src.out)


        # self.instantaneous_partial()

        tmp = self.dst.out - self.follow_post_mean * self.trace_post_mean 
        tmp = torch.square(tmp)
        self.trace_post_sigma2 = self.scl_sigma2 * (self.trace_post_sigma2 + (tmp / self.tau_var))
        
        self.trace_post_mean = self.scl_post_mean * (self.trace_post_mean + self.dst.out)
        
        self.compute_err()
        self.store_post_out[self.store_idx] = self.dst.out * 1.0
        self.choose_idx = (self.choose_idx + 1) % self.store_post_out_len
        self.store_idx = (self.store_idx + 1) % self.store_post_out_len
        self.trace_err_flt = self.trace_err_flt + self.follow_trace_err_flt * (self.trace_err -  self.trace_err_flt)
        self.trace_err = self.trace_err + self.follow_trace_err * (self.err - self.trace_err)

    def propagate(self):
        self.forward()
        self.evolve()



class Connection(Con):

    """
    Defines a synaptic connection between two neuron groups with configurable properties,
    including random initialization with uniform or Gaussian distributions.

    Attributes:
        src (CellGroup): Source neuron group providing the input.
        dst (CellGroup): Destination neuron group receiving the input.
        gaussian (bool): Flag to determine if Gaussian initialization is used.
        connection_prob (float): Probability of connection between neurons.
        initial_weight (float): Initial weight for connections that are established.
        sigma (float): Standard deviation for Gaussian distribution used in weight initialization.
        operation (nn.Module): PyTorch module defining the transformation applied to input data.
        target (str): Target state in the destination group that the operation affects.
        bias (bool): Flag to include bias in the operation.
    """

    def __init__(
        self,
        src,
        dst,
        gaussian=False,
        connection_prob=0.5,
        initial_weight=0.1,
        sigma=20,
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

    def forward(self):
        preact = self.src.out
        if not self.propagate_gradients:
            preact = preact.detach()
        if self.flatten_input:
            shp = preact.shape
            preact = preact.reshape(shp[:1] + (-1,))

        out = self.op(preact)
        self.dst.add_to_state(self.target, out)

    def update_weight(self):
        pass

    def propagate(self):
        self.forward()


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
        
        self.trace_pre = torch.zeros_like(self.src.out, device=device, dtype=dtype)
        self.trace_post = torch.zeros_like(self.dst.out, device=device, dtype=dtype)
        
        tau_stdp = torch.tensor(self.tau_stdp, device=device, dtype=dtype)
        target = torch.tensor(self.kappa, device=device, dtype=dtype)
        self.scl_stdp = torch.exp(-time_step/tau_stdp)
        self.dcy_stdp = 1.0 - self.scl_stdp
        self.kappa_fudge = 2 * target * tau_stdp

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

        self.trace_pre = self.scl_stdp * (self.trace_pre + self.src.out)
        self.trace_post = self.scl_stdp * (self.trace_post + self.dst.out)


    def forward(self):
        preact = self.src.out
        if not self.propagate_gradients:
            preact = preact.detach()
        if self.flatten_input:
            shp = preact.shape
            preact = preact.reshape(shp[:1] + (-1,))

        out = self.op(preact)
        self.dst.add_to_state(self.target, out)

    def update_weight(self):
        pass

    def propagate(self):
        self.forward()
        self.evolve()




