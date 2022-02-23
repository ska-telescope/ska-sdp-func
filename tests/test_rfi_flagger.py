from ska.sdp.func import rfi_flagger
import casacore.tables as tbl
import numpy as np
try:
    import cupy
except importerror:
    print("cupy not available, exiting..")
    exit(-1)

def threshold_calc(initial_value, rho, seq_lengths):
    thresholds = np.zeros(len(seq_lengths), dtype=np.float32)
    for i in range(len(seq_lengths)):
        m = pow(rho, np.log2(seq_lengths[i]))
        thresholds[i] = initial_value / m
    return thresholds

def test_rfi_flagger():
    vis = tbl.table("/home/ajay/work/gpu/data/20_sources_with_screen/aa0.5_ms.MS")
    rfi = tbl.table("/home/ajay/work/gpu/data/aa05_low_rfi_84chans.ms")
    num_freqs=200
    num_polarisations=4
    num_seq_len = 6
    sequence_lengths = np.array([1, 2, 4, 8, 16, 32], dtype=np.int32)
    rho1 = 1.5
    initial_threshold=20
    thresholds = threshold_calc(initial_threshold, rho1, sequence_lengths)
    start_row=0


    num_rows = vis.nrows()
    vis_data = vis.getcol('DATA', start_row,num_rows )
    rfi_data = rfi.getcol('DATA', start_row,num_rows)
    spectrogram = abs(vis_data + rfi_data)
    tmpspec=np.zeros((np.shape(spectrogram)[0],np.shape(spectrogram)[2],np.shape(spectrogram)[1]),dtype=np.float32)
    flags=np.zeros((np.shape(spectrogram)[0],np.shape(spectrogram)[2],np.shape(spectrogram)[1]),dtype=np.int32)

    for i in range(np.shape(spectrogram)[0]):
        tmpspec[i]=np.transpose(spectrogram[i])

    spectrogram_gpu=cupy.asarray(tmpspec)
    sequence_gpu=cupy.asarray(sequence_lengths)
    threshold_gpu=cupy.asarray(thresholds)
    flags_gpu=cupy.asarray(flags)

    rfi_flagger(spectrogram_gpu,sequence_gpu,threshold_gpu,flags_gpu)
    flags=cupy.asnumpy(flags_gpu)
