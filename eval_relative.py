
import torch
from torch.cuda.amp import autocast
from anchors import sample_anchors
import pandas as pd
import numpy as np
import os
import tqdm
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from models import relative
from measures import sample_fld, class_fld
from data import Latent
from utils import highlight_anchors
import argparse

parser = argparse.ArgumentParser(description='Relative representation evaluation')

pio.templates.default = 'plotly_white'

parser.add_argument('--model', type=str, default='VGG16', metavar='MODEL',
                    help='model name (default: VGG16)')
parser.add_argument('--data_set', type=str, default='CIFAR10', metavar='DATASET',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='input batch size (default: 128)')
parser.add_argument('--n_anchors', type=int, default=512, metavar='N',
                    help='number of anchors (default: 512)')
parser.add_argument('--lda_dim', type=int, default=8, metavar='N',
                    help='LDA dimension (default: 2)')
parser.add_argument('--dtype', type=str, default='float32', metavar='TYPE',
                    help='data type (default: float32)')

args = parser.parse_args()

model_name = args.model
data_set = args.data_set
batch_size = args.batch_size
n_anchors = args.n_anchors
lda_dim = args.lda_dim
dtype = args.dtype

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
latent_dataset = torch.load(f'./data/{data_set}_{model_name}_latent_space.pt')
latent_loader = torch.utils.data.DataLoader(latent_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
fig_path = os.path.join('figures', data_set, model_name)
os.makedirs(fig_path, exist_ok=True)


# Load Data
latent_space = []
targets = []
for i, (data, target) in enumerate(latent_loader):
    data.requires_grad = False
    latent_space.append(data.to(dtype=getattr(torch, dtype)))
    targets.append(target.to(torch.long))

latent_space = torch.cat(latent_space, dim=0)
latent_space_mean = latent_space.mean(dim=0, keepdim=True)
latent_space -= latent_space_mean
latent_space_0 = latent_space[:,:,0]
targets = torch.cat(targets, dim=0)

print(f'Latent space shape: {latent_space.shape}')
print(f'Latent space mean shape: {latent_space_mean.shape}')
print(f'Latent space_0 shape: {latent_space_0.shape}')
print(f'Targets shape: {targets.shape}')

anchor_types = ['rand', 'furth', 'furth_sum', 'furth_cos', 'furth_sum_cos']
anchors_idx = {}
n_anchors = latent_space_0.shape[-1]
for anchor_type in anchor_types:
    anchors_idx[anchor_type] = sample_anchors(latent_space_0, procedure=anchor_type, n_anchors=n_anchors)

for anchor_type, idx in anchors_idx.items():
    print(f'First 10 {anchor_type} anchors: {idx[:10].tolist()}')


# Statistics
df = []
mean = latent_space_0.mean().item()
std = latent_space_0.std().item()
min_norm, max_norm = torch.norm(latent_space_0, dim=1).min().item(), torch.norm(latent_space_0, dim=1).max().item()
rank = torch.linalg.matrix_rank(latent_space_0).item()
df.append({'model': model_name, 'mean': mean, 'std': std, 'min norm': min_norm, 'max norm': max_norm, 'rank': rank})

for anchor_type, idx in anchors_idx.items():
    anchors = latent_space_0[idx]
    mean = anchors.mean().item()
    std = anchors.std().item()
    min_norm, max_norm = torch.norm(anchors, dim=1).min().item(), torch.norm(anchors, dim=1).max().item()
    rank = torch.linalg.matrix_rank(anchors).item()
    df.append({'model': f'Anchors {anchor_type}', 'mean': mean, 'std': std, 'min norm': min_norm, 'max norm': max_norm, 'rank': rank})

df = pd.DataFrame(df)
df = df.round({'mean': 2, 'std': 2, 'min norm': 2, 'max norm': 2})

fig = go.Figure(data=go.Table(
header=dict(values=list(df.columns)),
cells=dict(values=[df[col] for col in df.columns])))
fig.update_layout(height=350, width=800, font_family='Open-Sherif')
fig.update_layout(title_text=f"{model_name} statistics", )
fig.write_image(os.path.join(fig_path, 'statistics.png'), scale=3)

# LDA absolute space
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(latent_space.mean(dim=-1), targets)
df = []
sequence = [1, 10, 20, 30, 40, 50]
for snapshot in sequence:
    X_lda = lda.transform(latent_space[:,:,snapshot-1])
    for i in range(X_lda.shape[0]):
        df.append({'snapshot': snapshot, 'z0': X_lda[i,0], 'z1': X_lda[i,1], 'target': str(targets[i].item())})

df = pd.DataFrame(df).sort_values(by=['snapshot', 'target'])

fig = px.scatter(pd.DataFrame(df), x='z0', y='z1', color='target', facet_col='snapshot',
                 title=f"LDA - {model_name} - absolute space")
fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_traces(marker=dict(size=3))
fig.update_layout(height=400, width=1000, font_family='Open-Sherif')
fig.write_image(os.path.join(fig_path, 'lda_absolute.png'), scale=3)

# Relative transformations
rel_models = {}
rel_latent_space = {}

projections = ['cosine', 'basis_norm', 'euclidean']
max_value = latent_space_0.shape[1]
anchor_steps = (2**np.arange(1, np.log2(max_value).astype(int)+1)).astype(int)
if anchor_steps[-1] != max_value:
    anchor_steps = np.concatenate([anchor_steps, [max_value]])


for anchor_type, idx in anchors_idx.items():
    rel_models[anchor_type] = {}
    rel_latent_space[anchor_type] = {}
    for projection in projections:
        anchor_space = latent_space[idx]
        if projection == 'cosine':
            rel_model = relative.RelCos(anchors=anchor_space)
        elif projection == 'basis':
            rel_model = relative.RelBasis(anchors=anchor_space)
        elif projection == 'basis_norm':
            rel_model = relative.RelBasis(anchors=anchor_space, norm=True)
        elif projection == 'euclidean':
            rel_model = relative.RelDist(anchors=anchor_space, p=2)
        else:
            raise ValueError(f'Unknown projection: {projection}')

        rel_models[anchor_type][projection] = rel_model.to(device)
        rel_latent_space[anchor_type][projection] = {}

        for step in anchor_steps:
            rel_latent_space[anchor_type][projection][step] = []


with torch.no_grad():
    for i, (data, target) in enumerate(tqdm.tqdm(latent_loader)):
        data.requires_grad = False
        data = data.to(device, dtype=getattr(torch, dtype))
        data -= latent_space_mean.to(device)
        for anchor_type in anchor_types:
            for projection in projections:
                rel_model = rel_models[anchor_type][projection]
                for step in anchor_steps:
                    with autocast():
                        out = rel_model(data, n_anchors=step).detach()
                    rel_latent_space[anchor_type][projection][step].append(out.cpu())

print(f'Conecatenating relative latent spaces')
for anchor_type in anchor_types:
    print(f'\tAnchor type: {anchor_type}')
    for projection in projections:
        print(f'\t\tProjection: {projection}')
        for step in anchor_steps:
            print(f'\t\t\tStep: {step}')
            rel_latent_space[anchor_type][projection][step] = torch.stack(rel_latent_space[anchor_type][projection][step], dim=0).flatten(0, 1)
            # rel_latent_space[anchor_type][projection][step] = torch.cat(rel_latent_space[anchor_type][projection][step], dim=0)
            # print(f'\t\t\tShape: {rel_latent_space[anchor_type][projection][step].shape}')


# Evaluation
print(f'Evaluation')
estimator = 'var'
df_rel = []
for anchor_type in anchor_types:
    for projection in projections:
        for step in anchor_steps:

            space = rel_latent_space[anchor_type][projection][step]

            fld, between, within = sample_fld(space, estimator=estimator)
            df_rel.append({'model': model_name, 'anchor_type': anchor_type, 'n_anchors': step, 'projection': projection, 'type': 'sample', 'fld': fld, 'between': between, 'within': within})
            
            fld, between, within = class_fld(space, targets, estimator=estimator)
            df_rel.append({'model': model_name, 'anchor_type': anchor_type, 'n_anchors': step, 'projection': projection, 'type': 'class', 'fld': fld, 'between': between, 'within': within})
        


df_abs = []
for snapshot in range(latent_space.shape[-1]):

    space = latent_space[:,:, snapshot]

    fld, between, within = sample_fld(space.unsqueeze(-1), estimator=estimator)
    df_abs.append({'model': model_name, 'snapshot': snapshot, 'type': 'sample', 'fld': fld, 'between': between, 'within': within})

    fld, between, within = class_fld(space.unsqueeze(-1), targets, estimator=estimator)
    df_abs.append({'model': model_name, 'snapshot': snapshot, 'type': 'class', 'fld': fld, 'between': between, 'within': within})


fld, between, within = sample_fld(latent_space, estimator=estimator)
df_abs.append({'model': model_name, 'snapshot': 'all', 'type': 'sample', 'fld': fld, 'between': between, 'within': within})

fld, between, within = class_fld(latent_space, targets, estimator=estimator)
df_abs.append({'model': model_name, 'snapshot': 'all', 'type': 'class', 'fld': fld, 'between': between, 'within': within})

df_rel = pd.DataFrame(df_rel)
df_abs = pd.DataFrame(df_abs)

for plot_type in ['sample', 'class']:
    for measure in ['fld']:
        
        fig = px.line(df_rel[(df_rel['type'] == plot_type) & (df_rel['model'] == model_name)], x='n_anchors', y=f'{measure}',
                    color='anchor_type', facet_col='projection',
                    title=f'{model_name} - {plot_type} - {measure} - relative space',
                    facet_col_spacing = 0.1,
                    markers=True,
                    log_x=True)
        
        fig.update_traces(marker=dict(size=4))
        fig.update_xaxes(showgrid=True, showline=False, zeroline=False)
        fig.update_yaxes(showgrid=True, showline=False, zeroline=False)

        for i in range(0, len(projections)):
            abs_fld_lower = df_abs[(df_abs['model'] == model_name) & (df_abs['type'] == plot_type) & (df_abs['snapshot'] == 'all')][f'{measure}'].values[0]
            abs_fld_upper = df_abs[(df_abs['model'] == model_name) & (df_abs['type'] == plot_type) & (df_abs['snapshot'] != 'all')][f'{measure}'].mean()

            fig.add_hline(y=abs_fld_lower, line_dash='dot',
                        line_color='black',
                        annotation_text=round(abs_fld_lower, 2) if i == 0 else '',
                        col=i)
            fig.add_hline(y=abs_fld_upper, line_dash='dot',
                        line_color='black',
                        annotation_text=round(abs_fld_upper, 2) if i == 0 else '',
                        col=i)
        fig.update_layout(height=400, width=1000, font_family='Open-Sherif')
        fig.write_image(os.path.join(fig_path, f'fld_{plot_type}.png'), scale=3)


# LDA relative space
df = []
for anchor_type in anchor_types:
    for projection in projections:
        space = rel_latent_space[anchor_type][projection][lda_dim]
        lda = LinearDiscriminantAnalysis(n_components=2)
        lda.fit(space.mean(dim=-1), targets)

        for snapshot in sequence:
            X_lda = lda.transform(space[:,:,snapshot-1])
            for i in range(X_lda.shape[0]):
                df.append({'model': model_name, 'anchor_type': anchor_type, 'f_proj': projection, 'snapshot': snapshot, 'target': str(targets[i].item()), 'z0': X_lda[i, 0], 'z1': X_lda[i, 1]})
            for i in range(lda_dim):
                idx_a = anchors_idx[anchor_type][i]
                df.append({'model': model_name, 'anchor_type': anchor_type, 'f_proj': projection, 'snapshot': snapshot, 'target': 'anchor', 'z0': X_lda[idx_a, 0], 'z1': X_lda[idx_a, 1]})

df = pd.DataFrame(df)
df = df.sort_values(by=['anchor_type', 'f_proj', 'snapshot', 'target'])

for projection in projections:
    df_ = df[(df['f_proj'] == projection)]  
    fig = px.scatter(df_, x='z0', y='z1', color='target', facet_col='snapshot', facet_row='anchor_type',
                    facet_col_spacing=0.01, facet_row_spacing=.02,
                    title=f'LDA - {model_name} - {projection} projection')
    fig.update_traces(marker=dict(size=3))
    fig = highlight_anchors(fig)
    fig.update_layout(height=250*len(anchor_types), width=1000, font_family='Open-Sherif')
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(font_family='Open-Sherif')
    fig.write_image(os.path.join(fig_path, f'lda_{projection}.png'), scale=3)

