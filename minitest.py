from ska_sdp_func.visibility import flagger
import casacore.tables as tables
import numpy as np
import ms_operations
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import sys
from matplotlib.colors import ListedColormap

table_address = "~/twosm_summer/data/1672727931_sdp_l0.ms"
measurementSet = ms_operations.ReadMS(table_address)
#ms = tables.table(table_address)


th0 = 0.8
th1 = 0.8
th2 = 8
th3 = 0.9
th4 = 0
th5 = 2
th6 = 0.2
th7 = 1
th8 = 0.8
parameters = np.array([th0, th1, th2, th3, th4, th5, th6, th7, th8], dtype=np.float32)

vis_ms = measurementSet.GetMainTableData('DATA')
flags_ms = measurementSet.GetMainTableData('FLAG')

chan_freq = measurementSet.GetFreqTableData('CHAN_FREQ')[0]

baseline1 = measurementSet.GetMainTableData('ANTENNA1').astype(np.int32)
baseline2 = measurementSet.GetMainTableData('ANTENNA2').astype(np.int32)
antennas = np.unique(baseline1)

ant_name = measurementSet.GetAntennaTableData('NAME')

num_ants = len(ant_name)
num_baselines = int(num_ants * (num_ants + 1)/2)
num_freqs = len(chan_freq)
num_times = int(vis_ms.shape[0]/num_baselines)
vis_ms_dims = vis_ms.shape
num_pols = vis_ms_dims[2]


vis_all = vis_ms.reshape([num_times, num_baselines, num_freqs, num_pols])

vis = np.ascontiguousarray(np.squeeze(vis_all[17:30, :, :, :]))
flags = np.zeros(vis.shape, dtype=np.int32)

flagger(vis, parameters, flags, antennas, baseline1, baseline2)


cmap = sns.cubehelix_palette(start=1.8, rot=1.1, light=0.7, n_colors=2)
ticks = np.array([0, 1])
ax = sns.heatmap(np.squeeze(flags[:, 0, :, 0]), cmap=ListedColormap(cmap), cbar_kws={"ticks": [0, 1]})

plt.title('flags')
plt.xlabel('channels')
plt.ylabel('time')
plt.savefig('figs/fig1.png')
