cimport numpy
import numpy

from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string

cpdef initialize(
    str yukarin_s_forwarder_path,
    str yukarin_sa_forwarder_path,
    str decode_forwarder_path,
    bool use_gpu
):
    cdef bool success = c_initialize(
        <string> yukarin_s_forwarder_path.encode(),
        <string> yukarin_sa_forwarder_path.encode(),
        <string> decode_forwarder_path.encode(),
        use_gpu
    )
    assert success

cpdef yukarin_s_forward(
    int length,
    numpy.ndarray[numpy.int64_t, ndim=1] phoneme_list,
    numpy.ndarray[numpy.int64_t, ndim=1] speaker_id,
):
    cdef vector[float] output
    cdef vector[long] output_size
    cdef bool success = c_yukarin_s_forward(
        length,
        <long*> phoneme_list.data,
        <long*> speaker_id.data,
        &output,
        &output_size,
    )
    assert success
    return numpy.asarray(output).reshape(output_size)


cpdef yukarin_sa_forward(
    int length,
    numpy.ndarray[numpy.int64_t, ndim=2] vowel_phoneme_list,
    numpy.ndarray[numpy.int64_t, ndim=2] consonant_phoneme_list,
    numpy.ndarray[numpy.int64_t, ndim=2] start_accent_list,
    numpy.ndarray[numpy.int64_t, ndim=2] end_accent_list,
    numpy.ndarray[numpy.int64_t, ndim=2] start_accent_phrase_list,
    numpy.ndarray[numpy.int64_t, ndim=2] end_accent_phrase_list,
    numpy.ndarray[numpy.int64_t, ndim=1] speaker_id,
):
    cdef vector[float] output
    cdef vector[long] output_size
    cdef bool success = c_yukarin_sa_forward(
        length,
        <long*> vowel_phoneme_list.data,
        <long*> consonant_phoneme_list.data,
        <long*> start_accent_list.data,
        <long*> end_accent_list.data,
        <long*> start_accent_phrase_list.data,
        <long*> end_accent_phrase_list.data,
        <long*> speaker_id.data,
        &output,
        &output_size,
    )
    assert success
    return numpy.asarray(output, numpy.float32).reshape(output_size)

cpdef decode_forward(
    int length,
    int phoneme_size,
    list f0_list,
    list phoneme_list,
    numpy.ndarray[numpy.int64_t, ndim=1] speaker_id,
):
    cdef int i
    cdef numpy.ndarray[numpy.float32_t, ndim=2] tmp
    cdef vector[float *] f0_vector
    cdef vector[float *] phoneme_vector
    for i in range(len(f0_list)):
        tmp = f0_list[i]
        f0_vector.push_back(<float *> tmp.data)
    for i in range(len(phoneme_list)):
        tmp = phoneme_list[i]
        phoneme_vector.push_back(<float *> tmp.data)

    cdef vector[float] output
    cdef vector[long] output_size
    cdef bool success = c_decode_forward(
        length,
        phoneme_size,
        f0_vector,
        phoneme_vector,
        <long*> speaker_id.data,
        &output,
        &output_size,
    )
    assert success
    return numpy.asarray(output, numpy.float32).reshape(output_size)
