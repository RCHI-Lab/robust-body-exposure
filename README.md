# Robust Body Exposure (RoBE): A Graph-based Dynamics Modeling Approach to Manipulating Blankets over People

This code accompanies the submission:  
["Robust Body Exposure (RoBE): A Graph-based Dynamics Modeling Approach to Manipulating Blankets over People"](https://arxiv.org/abs/2304.04822)

Kavya Puthuveetil, Sasha Wald, Atharva Pusalkar, Pratyusha Karnati, and Zackory Erickson

## Citation
##### ["Robust Body Exposure (RoBE): A Graph-based Dynamics Modeling Approach to Manipulating Blankets over People"](https://arxiv.org/abs/2304.04822)
K. Puthuveetil, Sasha Wald, Atharva Pusalkar, Pratyusha Karnati, and Z. Erickson, “Robust Body Exposure (RoBE): A Graph-based Dynamics Modeling Approach to Manipulating Blankets over People,” 2023.

```
@misc{puthuveetil2023robust,
      title={Bodies Uncovered: Learning to Manipulate Real Blankets Around People via Physics Simulations}, 
      author={Kavya Puthuveetil and Sasha Wald and Atharva Pusalkar and Pratyusha Karnati and Zackory Erickson},
      year={2023},
      eprint={2304.04822}, 
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

## Install

### General Packages

To run the RoBE framework, you will need install the following packages according to their respective installation instructions. We have provided the package versions that we used, though other versions may also work.
1. [pytorch](https://pytorch.org/get-started/previous-versions/#v1101)==1.10.1+cu113
2. [torch-scatter](https://github.com/rusty1s/pytorch_scatter)==2.0.9
3. [torch-sparse](https://github.com/rusty1s/pytorch_sparse)==0.6.12
4. [torch-geometric](https://pytorch-geometric.readthedocs.io/en/2.0.3/notes/installation.html)==2.0.3
5. [tensorboard](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html)==2.9.1
6. [cma](https://github.com/CMA-ES/pycma)==3.1.0


### Assistive Gym

This repository provides a version of [Assistive Gym](https://github.com/Healthcare-Robotics/assistive-gym) modified for this work, as well as additional task-specific functionality. **Although files for other assistive environments are included, ONLY the RoBE Bedding Manipulation enviornment is functional!**

For more details on installing the version of Assistive Gym contained in this repository, check out the [installation guide for Assistive Gym](https://github.com/Healthcare-Robotics/assistive-gym/wiki/1.-Install). Just replace the lines that say `git clone https://github.com/Healthcare-Robotics/assistive-gym.git` and `cd assistive-gym` with:
```
git clone https://github.com/RCHI-Lab/robust-body-exposure.git
cd robust-body-exposure/assistive-gym-fem
```
Generating the actuated human model in the RoBE Bedding Manipulation environment relies on SMPL-X human mesh models. In order to use these models, you will need to create an account at https://smpl-x.is.tue.mpg.de/index.html and [download](https://smpl-x.is.tue.mpg.de/download.php) the mesh models. Once downloaded, extract the file and move the entire `smplx` directory to `robust-body-exposure/assistive_gym/envs/assets/smpl_models/`. Once complete, you should have several files with this format: `robust-body-exposure/assistive_gym/envs/assets/smpl_models/smplx/SMPLX_FEMALE.npz`. This step is REQUIRED to run the RoBE Bedding Manipulation enviornment!

## Download Models
To run RoBE with our pre-trained dynamics models or to run our trained PPO policies for bedding manipulation, you will need to download them (15.5 GB) from the following link: [pre-trained dynamics models](https://drive.google.com/drive/folders/1pJbTdy3lsDDvSy7WUoEhFkFN9oaKVIUX?usp=sharing). 

Once downloaded, move the unzipped `trained_models` directory (with all sub-directories also unzipped) into the `robust-body-exposure` directory. The final path to the dynamics models, for example, should be `robust-body-exposure/trained_models/GNN` 


## Basics
The RoBE Bedding Manipulation environment, built in [Assistive Gym](https://github.com/Healthcare-Robotics/assistive-gym), can be visualized using the following command:
```
python3 -m assistive_gym --env RobustBodyExposure-v1
```

## Running RoBE in Simulation

Lets try running an evaluation of RoBE in simulation! All of the commands below assume that they are being run from the `robust-body-exposure` directory so please `cd` accordingly!

To optimize over a pre-trained RoBE dynamics model and uncover randomly selected target limbs over 100 simulation rollouts from the training distribution:
```
python3 code/run_robe_sim.py --model-path 'standard_2D_10k_epochs=250_batch=100_workers=4_1668718872' --graph-config 2D --env-var standard --num-rollouts 100
```

## Training New Dynamics Models

Given a new dataset of cloth interactions, we can train a new dynamics model that can be used in the RoBE framework. To do so, first ensure that the raw data (in the form of pickle files) is contained in the following directory structure `robust-body-exposure/DATASETS/<YOUR-DATASET-NAME>/raw`. Once the raw data is in the right location, we can process the data for training (if needed) and actually execute training using the following command, which, in this example, trains a model using 10k training samples:
```
python3 code/train_gnns.py --dataset-name <YOUR-DATASET-NAME> --dataset-desc <YOUR-DATASET-DESCRIPTION> --model-name <YOUR-MODEL-NAME> --num-train-samp 10000
```
Where the `--dataset-desc` flag is usually used to specify processing parameters (whether overhanging blanket points are rotated, if using a 3D representation of the cloth, etc.)




