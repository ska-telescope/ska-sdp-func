#ifndef SKA_SDP_PROC_FUNC_RFI_FLAGGER_CONFIG_H_
#define SKA_SDP_PROC_FUNC_RFI_FLAGGER_CONFIG_H_


class RFI_flagger_params {
public:
	static const int nThreads = 256;
	static const int warp = 32;
};

#endif /* include guard */