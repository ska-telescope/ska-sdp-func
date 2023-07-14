from ska_sdp_func import twosm_rfi_flagger
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
th2 = 1
thresholds = np.array([th0, th1, th2], dtype=np.float32)

vis_ms = measurementSet.GetMainTableData('DATA')
flags_ms = measurementSet.GetMainTableData('FLAG')

chan_freq = measurementSet.GetFreqTableData('CHAN_FREQ')[0]

antenna1 = measurementSet.GetMainTableData('ANTENNA1')
antenna2 = measurementSet.GetMainTableData('ANTENNA2')
ant_name = measurementSet.GetAntennaTableData('NAME')

num_ants = len(ant_name)
num_baselines = int(num_ants * (num_ants + 1)/2)
num_freqs = len(chan_freq)
num_times = int(vis_ms.shape[0]/num_baselines)
vis_ms_dims = vis_ms.shape
num_pols = vis_ms_dims[2]


vis_all = vis_ms.reshape([num_times, num_baselines, num_freqs, num_pols])

vis = np.ascontiguousarray(np.squeeze(vis_all[17:30, 0, :, 0]))
flags = np.zeros(vis.shape, dtype=np.int32)

twosm_rfi_flagger(vis, thresholds, flags)


cmap = sns.cubehelix_palette(start=1.8, rot=1.1, light=0.7, n_colors=2)
ticks = np.array([0, 1])
ax = sns.heatmap(flags, cmap=ListedColormap(cmap), cbar_kws={"ticks": [0, 1]})

plt.title('flags')
plt.xlabel('channels')
plt.ylabel('time')
plt.show()
