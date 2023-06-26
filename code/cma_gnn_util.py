import os
import os.path as osp
import pickle
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import torch
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def compute_fscore(initial_covered_status, final_covered_status):
    targ_uncov = 0
    nontarg_uncov = 0
    targ_cov = 0
    total_nontarg = 0
    nontarg_initially_covered = 0
    nontarg_cov = 0

    for i in range(len(final_covered_status)):
        bod_point_type = final_covered_status[i][0]
        is_covered =  final_covered_status[i][1]
        is_initially_covered = initial_covered_status[i][1]
        if bod_point_type == 1:
            if is_covered:
                targ_cov += 1
            else:
                targ_uncov +=1
        elif bod_point_type == 0 and is_initially_covered:
            total_nontarg += 1
            if not is_covered:
                nontarg_uncov += 1
            else:
                nontarg_initially_covered += 1
    # print(total_nontarg)
    total_targ = targ_cov+targ_uncov
    weight = total_targ/(total_targ+total_nontarg)
    penalties = []
    for i in range(1,nontarg_uncov+1):
        penalty = i*weight
        penalties.append(penalty if penalty <= 1 else 1)
        # penalty = i*weight if i+weight <= 1 else 1
    
    # print(penalties)
        
    tp = targ_uncov
    fp = np.sum(penalties)
    fn = targ_cov
    f_score = tp/(tp + 0.5*(fp+fn))

    # print(targ_uncov + targ_cov, total_nontarg)

    return f_score

def set_x0_for_cmaes(target_limb_code):
    # these actions are unscaled!
    if target_limb_code in [0, 1, 2]:       # Top Right
        x0 = [0.5, -0.4, 0, 0]
        # x0 = [0.5, -0.4, -0.5, -0.5]
    elif target_limb_code in [3, 4, 5]:     # Bottom Right
        x0 = [0.5, 0.5, 0, 0]
    elif target_limb_code in [6, 7, 8]:     # Top Left
        x0 = [-0.5, -0.4, 0, 0]
        # x0 = [-0.5, -0.4, 0.5, -0.5]
    elif target_limb_code in [9, 10, 11]:   # Bottom Left
        x0 = [-0.5, 0.5, 0, 0]
    elif target_limb_code in [13, 15]:
        x0 = [0, 0, 0,  0.5]
    elif target_limb_code in [12, 14]:
        x0 = [0, 0, 0, -0.5]
    else:
        x0 = [0, 0, 0, -0.5]
    return x0


def save_data_to_pickle(idx, seed, action, human_pose, target_limb_code, sim_info, cma_info, iter_data_dir):
    #! when lines below are uncommented, will not save if no grasp on cloth found
    # if isinstance(covered_status, int) and covered_status == -1:
    #     return
    pid = os.getpid()
    filename = f"tl{target_limb_code}_c{idx}_{seed}_pid{pid}"

    raw_dir = osp.join(iter_data_dir, 'raw')
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    pkl_loc = raw_dir

    with open(os.path.join(pkl_loc, filename +".pkl"),"wb") as f:
        pickle.dump({
            "action":action,
            "human_pose":human_pose,
            'target_limb_code':target_limb_code,
            'sim_info':sim_info,
            'cma_info':cma_info,
            'observation':[sim_info['observation']], # exposed here for bm_dataset use later, [] necessary for correct unpacking by BMDataset
            'info':sim_info['info']}, f)

def save_dataset(idx, graph, data, sim_info, action, human_pose, covered_status):
    # ! function behavior is not correct at the moment
    if isinstance(covered_status, int) and covered_status == -1:
        return


    initial_blanket_state = sim_info['info']["cloth_initial_subsample"]
    final_blanket_state = sim_info['info']["cloth_final_subsample"]
    cloth_initial, cloth_final = graph.get_cloth_as_tensor(initial_blanket_state, final_blanket_state)

    data['cloth_initial'] = cloth_initial
    data['cloth_final'] = cloth_final
    data['action'] = torch.tensor(action, dtype=torch.float)
    data['human_pose'] = torch.tensor(human_pose, dtype=torch.float)
    
    proc_data_dir = graph.proc_data_dir
    data = graph.dict_to_Data(data)
    torch.save(data, osp.join(proc_data_dir, f'data_{idx}.pt'))
    
# def generate_figure_sim_results(cloth_initial, pred, all_body_points, covered_status, initial_covered_status, info, cma_reward, sim_reward, action):
    
#     # handle case if action was clipped - there is essentially no point in generating this figure since initial and final is the same
#     if isinstance(covered_status, int) and covered_status == -1:
#         fig = None # clipped
#         return fig

#     initial_gt = np.array(cloth_initial)
#     covered_status_sim = info["covered_status_sim"]
#     final_sim = np.array(info["cloth_final_subsample"])

#     cma_fscore = compute_fscore(initial_covered_status, covered_status)
#     sim_fscore = compute_fscore(initial_covered_status, covered_status_sim)

#     point_colors = []
#     point_colors_sim = []

#     #! consolidate these two loops
#     for i in range(len(covered_status)):
#         is_target = covered_status[i][0]
#         is_covered = covered_status[i][1]
#         is_initially_covered = initial_covered_status[i][1]
#         if is_target == 1:
#             color = 'purple' if is_covered else 'forestgreen'
#         elif is_target == -1: # head points
#             color = 'red' if is_covered else 'darkorange'
#         else:
#             color = 'darkorange' if is_covered or not is_initially_covered else 'red'
#         point_colors.append(color)

#     for i in range(len(covered_status_sim)):
#         is_target = covered_status_sim[i][0]
#         is_covered = covered_status_sim[i][1]
#         is_initially_covered = initial_covered_status[i][1]

#         if is_target == 1:
#             color = 'purple' if is_covered else 'forestgreen'
#         elif is_target == -1: # head points
#             color = 'red' if is_covered else 'darkorange'
#         else:
#             color = 'darkorange' if is_covered or not is_initially_covered else 'red'
#         point_colors_sim.append(color)

#     # aspect = (4, 6)
#     aspect = (12, 10)

#     fig, (ax2, ax1) = plt.subplots(1, 2,figsize=aspect)
#     fig.patch.set_facecolor('white')
#     fig.patch.set_alpha(1.0)

#     ax1.scatter(all_body_points[:,0], all_body_points[:,1], c=point_colors)

#     s1 = ax1.scatter(initial_gt[:,0], initial_gt[:,1], alpha=0.2, edgecolors='none', label='cloth initial')
#     s3 = ax1.scatter(all_body_points[:,0], all_body_points[:,1], c=point_colors)
#     s2 = ax1.scatter(pred[:,0], pred[:,1], alpha=0.6, color='mediumvioletred', label='cloth final')
#     ax1.scatter(action[0], action[1], color='k')
#     ax1.scatter(action[2], action[3], color='b')
#     ax1.set_xlim([-0.7, 0.7])
#     ax1.set_ylim([-0.9, 1.05])
#     ax1.set_xlabel('x position')
#     ax1.set_ylabel('y position')
#     ax1.invert_yaxis()
#     ax1.set_title(f"Predicted: Rew = {cma_reward:.2f}, $F_1$ = {cma_fscore:.2f}", fontsize=15)

#     final_sim = np.array(info["cloth_final_subsample"])

#     ax2.scatter(initial_gt[:,0], initial_gt[:,1], alpha=0.2, edgecolors='none')
#     # s3 = ax2.scatter(pred.detach()[:,0], pred.detach()[:,1], color='red', alpha=0.6)
#     ax2.scatter(all_body_points[:,0], all_body_points[:,1], c=point_colors_sim)
#     s4 = ax2.scatter(final_sim[:,0], final_sim[:,1],  color='mediumvioletred', alpha=0.6)
#     ax2.scatter(action[0], action[1], color='k')
#     ax2.scatter(action[2], action[3], color='b')
    
#     ntarg = mlines.Line2D([], [], color='darkorange', marker='o', linestyle='None', label='uncovered points')
#     targ = mlines.Line2D([], [], color='forestgreen', marker='o', linestyle='None', label='target points')


#     ax2.set_xlim([-0.7, 0.7])
#     ax2.set_ylim([-0.9, 1.05])
#     ax2.set_xlabel('x position')
#     ax2.set_ylabel('y position')
#     ax2.set_title(f"Ground Truth: Rew = {sim_reward:.2f}, $F_1$ = {sim_fscore:.2f}", fontsize=15)
#     ax2.invert_yaxis()
#     # plt.show()

#     # plt.show()

#     # fig.legend((s1,s2,s3,s4), ('Initial GT', 'Final Predicted', 'Human', 'Final Sim'), 'lower center', ncol=4, borderaxespad=0.3)
#     fig.legend(loc='lower center', handles=[ntarg, targ, s1, s2], ncol=4, borderaxespad=2)

#     return fig
target_names = [
    '','','Right Arm',
    '','Right Lower Leg','Right Leg',
    '','','Left Arm',
    '','Left Lower Leg','Left Leg',
    'Both Lower Legs','Upper Body','Lower Body',
    'Whole Body'
]
def get_body_point_colors(initial_covered_status, covered_status):
    point_colors = []
    for i in range(len(covered_status)):
        is_target = covered_status[i][0]
        is_covered = covered_status[i][1]
        is_initially_covered = initial_covered_status[i][1]
        if is_target == 1:
            # infill = 'rgba(168, 102, 39, 1)' if is_covered else 'forestgreen'
            infill = 'rgba(184, 33, 166, 1)' if is_covered else 'forestgreen'
        elif is_target == -1: # head points
            # infill = 'red' if is_covered and not is_initially_covered else 'rgba(255,186,71,1)'
            infill = 'rgba(255, 186, 71, 1)' if not is_covered or is_initially_covered else 'red'
        else:
            infill = 'rgba(255, 186, 71, 1)' if is_covered or not is_initially_covered else 'red'
        point_colors.append(infill)
    
    return point_colors

# TODO: take in initial cloth state and both final cloth states, all body points, action
def generate_figure(action, all_body_points, cloth_initial, cloth_intermediate, cloth_final, plot_initial=False, compare_subplots=False, transparent=False, draw_axes=False):  
    scale = 4
    num_subplots = 3 if compare_subplots else 1
    bg_color = 'rgba(0,0,0,0)' if transparent else 'rgba(255,255,255,1)'
    
    # fig = go.Figure()
    fig = make_subplots(rows=1, cols=num_subplots)
    arrows = []
    
    for i in range(num_subplots):

        # print(i)
        #* For bed plotting
        fig.add_shape(type="rect",
            x0=0.44, y0=1.05, x1=-0.44, y1=-1.05,
            line=dict(color='rgb(163, 163, 163)'), fillcolor = 'rgb(163, 163, 163)', opacity=0.2, layer='below', row=1, col=i+1)

        #Light Blue
        if plot_initial:
            fig.add_trace(
                go.Scatter(mode='markers',
                           x = cloth_initial[:,0],
                           y = cloth_initial[:,1],
                           showlegend = False,
                           marker=dict(color = 'rgba(99, 190, 242, 0.3)', size = 9)), row=1, col=1)

        # TODO: don't need to change point colors based on covered/uncovered
        fig.add_trace(
            go.Scatter(mode='markers',
                       x = all_body_points[:,0],
                       y = all_body_points[:,1],
                       showlegend=False,
                       marker=dict(color = 'rgba(240, 30, 200, 1)')), row=1, col=i+1)

        #Dark Blue
        fig.add_trace(
            go.Scatter(mode='markers',
                x = cloth_intermediate[:,0],
                y = cloth_intermediate[:,1],
                showlegend = False,
                marker=dict(color = 'rgba(99, 190, 242, 0.3)', size = 9)), row=1, col=2)

        fig.add_trace(
            go.Scatter(mode='markers',
                       x = cloth_final[:,0],
                       y = cloth_final[:,1],
                       showlegend = False,
                       marker=dict(color = 'rgba(99, 190, 242, 0.3)', size = 9)), row=1, col=3)
        fig.add_trace(
            go.Scatter(mode='markers',
                       x = [action[0]], y = [action[1]], showlegend = False, marker=dict(color = 'rgba(0,0,0,1)', size = 12)),
                       row=1, col=[1])
        fig.add_trace(
            go.Scatter(mode='markers',
                       x = [action[2]], y = [action[3]], showlegend = False, marker=dict(color = 'rgba(0,0,0,1)', size = 12)),
                       row=1, col=[2])
        # if plotting more than one action arrow turns out to be annoying don't worry about it
        action_arrow_initial = go.layout.Annotation(dict(
                        ax=action[0],
                        ay=action[1],
                        xref=f"x{1}", yref=f"y{1}",
                        text="",
                        showarrow=True,
                        axref=f"x{1}", ayref=f"y{1}",
                        x=action[2],
                        y=action[3],
                        arrowhead=3,
                        arrowwidth=4,
                        arrowcolor='rgb(0,0,0)'))
        action_arrow_intermediate = go.layout.Annotation(dict(
                        ax=action[2],
                        ay=action[3],
                        xref=f"x{2}", yref=f"y{2}",
                        text="",
                        showarrow=True,
                        axref=f"x{2}", ayref=f"y{2}",
                        x=action[0],
                        y=action[1],
                        arrowhead=3,
                        arrowwidth=4,
                        arrowcolor='rgb(0,0,0)'))
        arrows += [action_arrow_initial, action_arrow_intermediate]
        fig.update_xaxes(autorange="reversed", visible=draw_axes, row=1, col=i+1)
        fig.update_yaxes(autorange="reversed", visible=draw_axes, row=1, col=i+1)
    
    fig.update_layout(width=140*3*scale, height=195*scale,plot_bgcolor=bg_color, paper_bgcolor=bg_color,annotations=arrows,
                         title={'text': f"", 'y':0.05,'x':0.5,'xanchor': 'center','yanchor': 'bottom'})

    # fig.show()
    return fig

    