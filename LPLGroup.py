import torch
from stork.nodes.base import CellGroup

class InputGroup(CellGroup):
    """A special group which is used to supply batched dense tensor input to the network via its feed_data function."""

    def __init__(self, shape, name="Input"):
        super(InputGroup, self).__init__(shape, name=name)

    def reset_state(self, batch_size=None):
        super().reset_state(batch_size)
        self.out = self.states["out"] = torch.zeros(self.int_shape, device=self.device, dtype=self.dtype)

    def feed_data(self, data):
        self.local_data = data.reshape((data.shape[:2] + self.shape)).to(self.device)

    def forward(self):
        self.out = self.states["out"] = self.local_data[:, self.clk]
    
    def evolve(self):
        self.forward()

    def clear_input(self):
        pass


    

class LPLGroup(CellGroup):
    """
    A neural group model with detailed synaptic dynamics based on physiological properties
    such as membrane time constants, synaptic time constants, and adaptive threshold mechanics.

    Attributes:
        shape (tuple): The number of neurons and possibly other dimensions defining the group's structure.
        tau_mem (float): The membrane time constant which affects the decay of the membrane potential.
        tau_ampa (float): Time constant for the decay of AMPA type excitatory postsynaptic currents.
        tau_nmda (float): Time constant for the decay of NMDA type excitatory postsynaptic currents, which are slower than AMPA.
        tau_gaba (float): Time constant for the decay of GABAergic inhibitory postsynaptic currents.
        tau_thr (float): Time constant for the adaptive threshold mechanism.
        thr_rest (float): Resting threshold potential, the baseline threshold.
        delta_thr (float): Increment of the threshold after a neuron fires.
        sigma_tau (float): Variability of the time constants, if non-zero applies a log-normal variation.
        urest (float): Resting membrane potential.
        uexc (float): Reversal potential for excitatory inputs.
        uinh (float): Reversal potential for inhibitory inputs.
        uleak (float): Leak potential, typically near the resting potential.
    """
    def __init__(
        self, 
        shape, 
        tau_mem   =20e-3, 
        tau_ampa  =5e-3, 
        tau_nmda  =100e-3, 
        tau_gaba  =10e-3, 
        tau_thr   =5e-3,
        thr_rest  =-50e-3, 
        delta_thr =100e-3, 
        sigma_tau =0.0, 
        urest     =-70e-3, 
        uexc      =0.0, 
        uinh      =-80e-3, 
        uleak     =-70e-3, 
        **kwargs
    ):
        super(LPLGroup, self).__init__(shape, **kwargs)
  
        self.tau_mem = tau_mem
        self.tau_ampa = tau_ampa
        self.tau_gaba = tau_gaba
        self.tau_nmda = tau_nmda
        self.thr_rest = thr_rest
        self.tau_thr  = tau_thr
        self.delta_thr = delta_thr
        self.urest = urest
        self.uexc = uexc
        self.uinh = uinh
        self.uleak = uleak
        self.sigma_tau = sigma_tau

        self.out = None
        self.mem = None
        self.ampa = None
        self.gaba = None
        self.nmda = None
        self.exc = None
        self.inh = None
        self.thr = None

        self.default_target = "exc"


    def configure(self, batch_size, nb_steps, time_step, device, dtype):
        super().configure(batch_size, nb_steps, time_step, device, dtype)
        tau_mem = torch.tensor(self.tau_mem, dtype=dtype).to(device)
        tau_ampa = torch.tensor(self.tau_ampa, dtype=dtype).to(device)
        tau_nmda = torch.tensor(self.tau_nmda, dtype=dtype).to(device)
        tau_gaba = torch.tensor(self.tau_gaba, dtype=dtype).to(device)
        tau_thr = torch.tensor(self.tau_thr, dtype=dtype).to(device)

        if self.sigma_tau:
            tau_mem, tau_ampa, tau_nmda, tau_gaba = [
                tau * torch.exp(self.sigma_tau * torch.randn(self.shape, dtype=dtype).to(device)) for tau in
                [tau_mem, tau_ampa, tau_nmda, tau_gaba]]
            
        self.scl_ampa = torch.exp(-1.0 * time_step / tau_ampa) 
        self.scl_gaba = torch.exp(-1.0 * time_step / tau_gaba) 
        self.scl_thr = torch.exp(-1.0 * time_step / tau_thr) 
        self.mul_nmda = 1.0 - torch.exp(-1.0 * time_step / tau_nmda) 
        self.mul_tau_mem = time_step/tau_mem
        self.dcy_ampa = 1 - self.scl_ampa
        self.dcy_gaba = 1 - self.scl_gaba

    def clear_input(self):
        # not passing previous state forces init during stateful
        self.exc = self.get_state_tensor("exc")
        self.inh = self.get_state_tensor("inh")


    def reset_state(self, batch_size=None):
        super().reset_state(batch_size)
        self.out = self.get_state_tensor("out", state=self.out)
        self.mem = self.get_state_tensor("mem", state=self.mem)
        self.ampa = self.get_state_tensor("ampa", state=self.ampa)
        self.gaba = self.get_state_tensor("gaba", state=self.gaba)
        self.nmda = self.get_state_tensor("nmda", state=self.nmda)
        self.exc = self.get_state_tensor("exc", state=self.exc)
        self.inh = self.get_state_tensor("inh", state=self.inh)
        self.thr = self.get_state_tensor("thr", state=self.thr)
        self.mem = self.mem + self.urest

    def get_spike_and_reset(self):
        out = torch.zeros_like(self.mem)
        out[self.mem > self.thr] = 1.0
        indices = (out == 1.0)
        return out, indices

    def forward(self):

         
        self.ampa = self.ampa + self.exc
        self.gaba = self.gaba + self.inh
        
        t_exc = -0.5 * (self.ampa + self.nmda) * self.mem
        t_inh = (self.mem - self.uinh) * self.gaba
        t_leak = self.mem - self.urest
        
        self.ampa = self.scl_ampa * self.ampa
        self.gaba = self.scl_gaba * self.gaba
        self.nmda = self.nmda + self.mul_nmda * (self.ampa - self.nmda) 
        
        
        self.thr = self.thr_rest + self.scl_thr * (self.thr - self.thr_rest)
        self.mem = self.mem + self.mul_tau_mem * (t_exc - t_inh - t_leak)
        new_out, indices = self.get_spike_and_reset()
        self.out = new_out
        
        self.mem[self.mem > 0.0] = 0.0
        self.mem[self.mem < self.urest] = self.urest
        self.mem[indices] = self.urest
        self.thr[indices] = self.delta_thr + self.thr_rest

        
    def evolve(self):
        self.forward()
    