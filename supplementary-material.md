---
layout: default
---

# Supplementary Material

The core findings of the work are detailed in the paper and can be understood independent of the content on this page. The following material is intended to expand on the methodology and results presented in the paper. 


## 2D vs. 3D Representations of the Cloth
Referenced on Page 2, Section III-A of the paper

In Sections III-A and III-B, we describe how input graphs to our dynamics models encode the 2D position of the cloth at each node instead of the 3D position, allowing our models to be invariant to changes in depth. To verify that this representation of the cloth does not negatively impact performance relative to a 3D representation, we train a GNN that takes in input graphs that encode the 3D position of a given cloth point at each node. When preparing a given cloth point cloud for composition of a graph, we do not rotate the overhanging points and instead only voxelize the point cloud. The dynamics model is trained identically to that trained on a 2D representation of a cloth. 

Evaluating both models on the training distribution described in Section III-B, we find that the 2D and 3D dynamics models achieve $$F_1 = 0.77\pm0.23$$ and $$F_1 = 0.80\pm0.19$$, respectively. On the combination distribution, both models have identical performance, both achieving 0.65$$\pm$$0.30. Ultimately, we see that encoding a 3D representation of the cloth provides no significant improvement in performance despite requiring tedious depth alignment procedures to transfer simulation-trained models to the real world.


## Preparing a Raw Point Cloud for Composition of a Graph: Rotating Overhanging Points
Referenced on Page 3, Section III-B of the paper

Given an overhanging point $$\boldsymbol{p}$$ on the raw cloth point cloud $$P$$, defined by whether the point's position along the $$z$$-axis is below the top of the bed, $$\boldsymbol{p} \in P: p_z < 0.575$$, we apply the following function to rotate the point to the 2D bed plane:

$$H(\boldsymbol{p}) = \begin{cases} p_x > 0 & T_R^{-1} R_{R} T_R \boldsymbol{p} \\ p_x < 0 & T_L^{-1} R_{L} T_L \boldsymbol{p} \end{cases}$$

where $$T_R$$ and $$T_L$$ are translation matrices representing the translation between the axes along the right ([0.44, 0, 0.58]) and left ([-0.44, 0, 0.58]) edges of the bed relative to the origin of the world (center of the bed), and where $$R_R$$ and $$R_L$$ are 90 and -90 degree rotation matrices around an axis along the length of the bed that passes through its center ($$y$$-axis).


## Performance using Dynamics Models Trained on Datasets of Various Size
Referenced on Page 3, Section III-B, of the paper

We train six dynamics models on datasets containing 100, 500, 1000, 5000, 7500, and 10,000 input graphs, respectively. The F-scores achieved by each model when used for control are shown in Figure \ref{fig:training_samps_v_fscore}. Performance plateaus once the number of training samples is greater than 5000. All further evaluations are conducted using the dynamics model trained on 10,000 input graphs.

## Optimization Hyperparameters
Referenced on Page 4, Section III-D, of the paper


## Formalization of the Geometric Baseline
Referenced on Page 6, Section IV-B, of the paper








[back to Homepage](./)
