from ska.sdp.func import rfi_flagger
import numpy as np
try:
    import cupy
except importerror:
    cupy = None

def threshold_calc(initial_value, rho, seq_lengths):
    thresholds = np.zeros(len(seq_lengths), dtype=np.float32)
    for i in range(len(seq_lengths)):
        m = pow(rho, np.log2(seq_lengths[i]))
        thresholds[i] = initial_value / m
    return thresholds

def test_rfi_flagger():
    num_freqs=200
    num_baselines=21
    num_times=5040
    num_polarisations=4
    num_seq_len = 6
    sequence_lengths = np.array([1, 2, 4, 8, 16, 32], dtype=np.int32)
    rho1 = 1.5

    spectrogram = np.random.random_sample(
        [num_times,num_baselines,num_freqs, num_polarisations]) + 0j 
    initial_threshold=20
    thresholds = threshold_calc(initial_threshold, rho1, sequence_lengths)
    flags=np.zeros(spectrogram.shape,dtype=np.int32)
    rfi_flagger(spectrogram,sequence_lengths,thresholds,flags)

    flags=np.zeros(spectrogram.shape,dtype=np.int32)
    #GPU testing
    if cupy:
        spectrogram_gpu=cupy.asarray(spectrogram)
        sequence_gpu=cupy.asarray(sequence_lengths)
        threshold_gpu=cupy.asarray(thresholds)
        flags_gpu=cupy.asarray(flags)

        rfi_flagger(spectrogram_gpu,sequence_gpu,threshold_gpu,flags_gpu)
        flags=cupy.asnumpy(flags_gpu)

if __name__== "__main__":
    test_rfi_flagger()
