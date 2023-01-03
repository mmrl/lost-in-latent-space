# Lost in Latent Space: Examining Failures of Disentangled Models at Combinatorial Generalisation.

[NeurIPS version](https://openreview.net/forum?id=7yUxTNWyQGf)

**Authors**: [Milton L. Montero](https://github.com/miltonllera), [Jeffrey S. Bowers](https://jeffbowers.blogs.bristol.ac.uk/), [Rui Ponte Costa](https://neuralml.github.io/), [Casimir J.H. Ludwig](https://casludwig.github.io) and [Gaurav Malhotra](https://scholar.google.com/citations?user=SqX8yX4AAAAJ&hl=en).

**Abstract**: Recent research has shown that generative models with highly disentangled representations fail to generalise to unseen combination of generative factor values. These findings contradict earlier research which showed improved performance in out-of-training distribution settings when compared to entangled representations. Additionally, it is not clear if the reported failures are due to (a) encoders failing to map novel combinations to the proper regions of the latent space, or (b) novel combinations being mapped correctly but the decoder being unable to render the correct output for the unseen combinations. We investigate these alternatives by testing several models on a range of datasets and training settings. We find that (i) when models fail, their encoders also fail to map unseen combinations to correct regions of the latent space and (ii) when models succeed, it is either because the test conditions do not exclude enough examples, or because excluded cases involve combinations of object properties with its shape. We argue that to generalise properly, models not only need to capture factors of variation, but also understand how to invert the process that causes the visual input.

---

This repo contains the code necessary to run the experiments for the article. The code was tested on Python 3.9.12 and PyTorch 1.11. There are implementations for:

1. three models:
  - CompositionNet: Solves the composition task using a [variational autoencoder](http://proceedings.mlr.press/v32/rezende14.html) backbone.
  - [CascadeVAE](http://proceedings.mlr.press/v97/jeong19d/jeong19d.pdf): Uses continuous and discrete variables in its latent space.
  - [LieGroupVAE](https://arxiv.org/abs/1901.07017v2): Models interactions between latent variables using Group Theory.
2. losses to train them:
  - [VAE](https://arxiv.org/abs/1312.6114): Penalize conditional posterior.
  - [$\beta$-VAE](https://openreview.net/pdf?id=Sy2fzU9gl): Add capacity constrain.
  - [WAE](https://arxiv.org/abs/1711.01558): Penalize marginal posterior.
  - [Information Cascade](http://proceedings.mlr.press/v97/jeong19d/jeong19d.pdf): Progressively allows latent variables to become non-zero.
3. five datasets to test the models on:
  - [dSprites](https://github.com/deepmind/dsprites-dataset): Simple, uniform sprites on black background to which several transformations are applied.
  - [3DShapes](https://github.com/deepmind/3d-shapes): 3D scenes with one object in a room observerd from different prespectives.
  - [MPI3D](https://github.com/rr-learning/disentanglement_dataset): Different frames of objects being manipulated by a robot arm.
  - [Circles](https://arxiv.org/abs/2106.03375v1): Dataset consisting of a circle in different positions of an image.
  - Simple: Extension of the circles dataset containing two shapes instead of one.

Is is technically possible to train other common unsupervised models (like standard VAEs and $\beta$-VAEs, however we do not use them in our experimetns.

## Setting up the Conda environment

Running these experiments requires (among others) the following libraries installed:

* [PyTorch and Torchvision](https://pytorch.org/): Basic framework for Deep Learning models and training.
* [Ignite](https://github.com/pytorch/ignite): High-level framework to train models, eliminating the need for much boilerplate code.
* [Sacred](https://github.com/IDSIA/sacred): Libary used to define and run experiments in a systematic way.
* [Matplotlib](https://matplotlib.org/): For plotting.
* [Jupyter](https://jupyter.org/): To produce the plots.

We recommend using the provided [environment configuration file](https://gist.github.com/miltonllera/e0a6ca7f3283b029d0e333730b0ce980) and intalling using:

```
conda env create -f torchlab-env.yml
```

## Directory structure

The repository is organized as follows:

```
data/
├── raw/
    ├── dsprites/
    ├── shapes3d/
    ├── mpi/
    ├── ....
	├── mpi3d_real.npz
├── sims/
    ├── disent/  # Runs will be added here, Sacred will asign names as integers of increasing value
    ├── composition/
scripts/
├── configs/
    ├── vaes.py  # An example config file with VAE architectures.
├── ingredients/
    ├── models.py  # Example ingredient that wrapps model initalization
├── experiments/
    ├── composition.py  # Experiment script for training disentangled models
src/
├── analysis/  # These folders contain the actual datasets, losses, model classes etc.
├── dataset/
├── models/
├── training/
```

The data structure should be self explanatory for the most part. The main thing to note is that ``src`` contains code for models that are used throughout the experiments while the ingredients contain wrappers around these to initialize them from the configuration files. Simulation results will be saved in sims. The results of the analysis were stored in a new folder (``results``, not shown). We attempted to use models with the hightes disentanglement in our analysis.

Datasets should appear in a subfolder as shown above. Right now, there is not method for automatically downloading the data, but they can be found in their corresponding repos. Alternatively, altering the source file or passing the dataset root as a parameter can be used to look for the datasets in another location[^1].

The configuration folder has the different parameters combinations used in the experiments. Following these should allow someone to define new experiments easily. Just remember to add the configurations to the appropriate ingredient using ``ingredient.named_config(config_function/yaml_file)``.

## Running an experiment

To run an experiment you should execute one of the scripts from the scripts folder with the appropraite options. We use Sacred to run and track experimetns. You can check the online documentation to understand how it works. Below you is the general command used and more can be found in the ``bin`` folder.

```
cd ~/path/to/project/scripts/
python -m experiments.composition with dataset.<option> model.<option> training.<option>
```

Sacred allows passing parameters using keyword arguments. For example we can change the latent size and $\beta$ from the default values:

```
python -m experiments.composition with dataset.dsprites model.kim training.factor model.latent_size=50 training.loss.params.beta=10
```

## Acknowledgements

We would like to thank everyone who gave feedback on this research, especially the members of the [Mind and Machine Research Lab](https://mindandmachine.blogs.bristol.ac.uk/) and [Neural and Machine Learning Group](https://neuralml.github.io/).

This project has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (grant agreement No. 741134).


## Citation

If the code here helps with your research, please cite it as:
```
@article{montero2022lost,
  title={Lost in Latent Space: Disentangled Models and the Challenge of Combinatorial Generalisation},
  author={Montero, Milton L and Bowers, Jeffrey S and Costa, Rui Ponte and Ludwig, Casimir JH and Malhotra, Gaurav},
  journal={arXiv preprint arXiv:2204.02283},
  year={2022}
}
```

[^1]: I might add code to automatically download the datasets and create the folders, but only if I have time.
