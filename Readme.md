This code is intended to reproduce the results of the SNN section of the [paper](https://www.nature.com/articles/s41593-023-01460-y)(The combination of Hebbian and predictive plasticity learns invariant object representations in deep sensory networks). You can find the original code [here](https://github.com/fmi-basel/latent-predictive-learning). Before using this code, you need to:

1. install the [stork](https://github.com/fmi-basel/stork?tab=readme-ov-file) first, because this code is based on the stork.
2. run mk_spikes.ipynb to make spike train for stimulate
3. run stimulate.ipynb, this may take several days
4. run analyze_result.ipynb to analyze result

By default, the simulation duration is 50000 seconds with a time step of 2 milliseconds. The original paper is 100000 seconds and 0.1 milliseconds. Under the original conditions, the runtime is very long, so it is not recommended to do so.


