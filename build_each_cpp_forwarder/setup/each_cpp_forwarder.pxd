from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string

cdef extern from "each_cpp_forwarder.h":
    bool c_initialize "initialize" (
        string yukarin_s_forwarder_path,
        string yukarin_sa_forwarder_path,
        string decode_forwarder_path,
        bool use_gpu
    )

    bool c_yukarin_s_forward "yukarin_s_forward" (
        int length,
        long *phoneme_list,
        long *speaker_id,
        vector[float] *output,
        vector[long] *output_size
    )

    bool c_yukarin_sa_forward "yukarin_sa_forward" (
        int length,
        long *vowel_phoneme_list,
        long *consonant_phoneme_list,
        long *start_accent_list,
        long *end_accent_list,
        long *start_accent_phrase_list,
        long *end_accent_phrase_list,
        long *speaker_id,
        vector[float] *output,
        vector[long] *output_size
    )

    bool c_decode_forward "decode_forward" (
        int length,
        int phoneme_size,
        vector[float *] f0_list,
        vector[float *] phoneme_list,
        long *speaker_id,
        vector[float] *output,
        vector[long] *output_size
    )
