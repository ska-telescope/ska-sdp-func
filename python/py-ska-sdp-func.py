import ctypes
import numpy
try:
    import cupy
except ImportError:
    cupy = None

class sdp_Mem(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("location", ctypes.c_int),
        ("num_elements", ctypes.c_size_t),
        ("owner", ctypes.c_int),
        ("ref_count", ctypes.c_int),
        ("data", ctypes.POINTER(ctypes.c_void_p))
    ]

# Load the shared library.
# This might be the hardest bit to get right, if using ctypes - in general,
# where should we look for the library?
libska_sdp_func = numpy.ctypeslib.load_library("libska_sdp_func", "./")


def sdp_mem_from_python(obj):
    class MemType:
        SDP_MEM_FLOAT = 4
        SDP_MEM_DOUBLE = 8
        SDP_MEM_COMPLEX_FLOAT = 36
        SDP_MEM_COMPLEX_DOUBLE = 40

    mem = sdp_Mem()
    mem.owner = 0
    mem.ref_count = 1

    if type(obj) == numpy.ndarray:
        mem.location = 0  # SDP_MEM_CPU
        mem.num_elements = obj.size
        mem.data = obj.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))
        if obj.dtype == numpy.float32:
            mem.type = ctypes.c_int(MemType.SDP_MEM_FLOAT)
        elif obj.dtype == numpy.float64:
            mem.type = ctypes.c_int(MemType.SDP_MEM_DOUBLE)
        elif obj.dtype == numpy.complex64:
            mem.type = ctypes.c_int(MemType.SDP_MEM_COMPLEX_FLOAT)
        elif obj.dtype == numpy.complex128:
            mem.type = ctypes.c_int(MemType.SDP_MEM_COMPLEX_DOUBLE)
        else:
            raise TypeError("Unsupported type of numpy array")

    elif type(obj) == cupy.ndarray:
        mem.location = 1  # SDP_MEM_GPU
        mem.num_elements = obj.size
        mem.data = ctypes.cast(obj.data.ptr, ctypes.POINTER(ctypes.c_void_p))
        if obj.dtype == cupy.float32:
            mem.type = ctypes.c_int(MemType.SDP_MEM_FLOAT)
        elif obj.dtype == cupy.float64:
            mem.type = ctypes.c_int(MemType.SDP_MEM_DOUBLE)
        elif obj.dtype == cupy.complex64:
            mem.type = ctypes.c_int(MemType.SDP_MEM_COMPLEX_FLOAT)
        elif obj.dtype == cupy.complex128:
            mem.type = ctypes.c_int(MemType.SDP_MEM_COMPLEX_DOUBLE)
        else:
            raise TypeError("Unsupported type of cupy array")

    return mem


def check_error(error):
    if error == 0:  # SDP_SUCCESS
        return
    if error == 1:  # SDP_ERR_RUNTIME
        raise RuntimeError("Generic runtime error")
    if error == 2:  # SDP_ERR_INVALID_ARGUMENT:
        raise RuntimeError("Invalid function argument")
    if error == 3:  # SDP_ERR_DATA_TYPE:
        raise TypeError("Unsupported data type")
    if error == 4:  # SDP_ERR_MEM_ALLOC_FAILURE:
        raise RuntimeError("Memory allocation failure")
    if error == 5:  # SDP_ERR_MEM_COPY_FAILURE:
        raise RuntimeError("Memory copy failure")
    if error == 6:  # SDP_ERR_MEM_LOCATION:
        raise RuntimeError("Unsupported memory location")


def vector_add(input_a, input_b, output_vector):
    if input_a.ndim != 1 or input_b.ndim != 1 or output_vector.ndim != 1:
        raise RuntimeError("All vectors must be one-dimensional")
    if input_a.size != input_b.size or input_a.size != output_vector.size:
        raise RuntimeError("All vectors must be of the same size")

    py_vectoradd = libska_sdp_func.sdp_vector_add
    py_vectoradd.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(sdp_Mem),
        ctypes.POINTER(sdp_Mem),
        ctypes.POINTER(sdp_Mem),
        ctypes.POINTER(ctypes.c_int)
    ]
    error_status = ctypes.c_int(0)
    py_vectoradd(
        input_a.size,
        sdp_mem_from_python(input_a),
        sdp_mem_from_python(input_b),
        sdp_mem_from_python(output_vector),
        ctypes.byref(error_status)
    )
    check_error(error_status.value)


def dft(source_directions, source_fluxes, uvw_lambda, vis):
    py_dft = libska_sdp_func.sdp_dft_point_v00
    py_dft.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(sdp_Mem),
        ctypes.POINTER(sdp_Mem),
        ctypes.POINTER(sdp_Mem),
        ctypes.POINTER(sdp_Mem),
        ctypes.POINTER(ctypes.c_int)
    ]
    (num_times, num_baselines, num_channels, _) = uvw_lambda.shape
    (num_components, _, num_pols) = source_fluxes.shape
    error_status = ctypes.c_int(0)
    py_dft(
        num_components,
        num_pols,
        num_channels,
        num_baselines,
        num_times,
        sdp_mem_from_python(source_fluxes),
        sdp_mem_from_python(source_directions),
        sdp_mem_from_python(uvw_lambda),
        sdp_mem_from_python(vis),
        ctypes.byref(error_status)
    )
    check_error(error_status.value)


def main():
    # Run vector add test on CPU, using numpy arrays.
    input_a = numpy.random.random_sample([1000])
    input_b = numpy.random.random_sample(input_a.shape)
    output_vector = numpy.zeros(input_a.shape, dtype=input_a.dtype)
    print("Adding vectors on CPU using ska-sdp-func...")
    vector_add(input_a, input_b, output_vector)
    numpy.testing.assert_array_almost_equal(output_vector, input_a + input_b)
    print("Vector addition on CPU PASSED")

    # Run vector add test on GPU, using cupy arrays.
    if cupy:
        input_a_gpu = cupy.asarray(input_a)
        input_b_gpu = cupy.asarray(input_b)
        output_vector_gpu = cupy.zeros(input_a.shape, dtype=input_a.dtype)
        print("Adding vectors on GPU using ska-sdp-func...")
        vector_add(input_a_gpu, input_b_gpu, output_vector_gpu)
        output_gpu_check = cupy.asnumpy(output_vector_gpu)
        numpy.testing.assert_array_almost_equal(
            output_gpu_check, input_a + input_b)
        print("Vector addition on GPU PASSED")

    # Run DFT test on CPU, using numpy arrays.
    num_components = 20
    num_pols = 4
    num_channels = 10
    num_baselines = (128 * 127) // 2
    num_times = 10
    fluxes = numpy.random.random_sample(
        [num_components, num_channels, num_pols]) + 0j
    directions = numpy.random.random_sample([num_components, 3])
    uvw_lambda = numpy.random.random_sample(
        [num_times, num_baselines, num_channels, 3])
    vis = numpy.zeros(
        [num_times, num_baselines, num_channels, num_pols],
        dtype=numpy.complex128)
    print("Testing DFT on CPU from ska-sdp-func...")
    dft(directions, fluxes, uvw_lambda, vis)
    print("done!")

    # Run DFT test on GPU, using cupy arrays.
    if cupy:
        fluxes_gpu = cupy.asarray(fluxes)
        directions_gpu = cupy.asarray(directions)
        uvw_lambda_gpu = cupy.asarray(uvw_lambda)
        vis_gpu = cupy.zeros(
            [num_times, num_baselines, num_channels, num_pols],
            dtype=numpy.complex128)
        dft(directions_gpu, fluxes_gpu, uvw_lambda_gpu, vis_gpu)
        output_gpu_check = cupy.asnumpy(vis_gpu)
        numpy.testing.assert_array_almost_equal(output_gpu_check, vis)
        print("DFT on GPU PASSED")


if __name__ == "__main__":
    main()
