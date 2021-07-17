#pragma once

#include <utility>

#if defined(_WIN32) || defined(_WIN64)
#define DllExport __declspec(dllexport)
#else
#define DllExport
#endif

DllExport bool initialize(char *yukarin_s_forwarder_path, char *yukarin_sa_forwarder_path, char *decode_forwarder_path,
                          bool use_gpu);

DllExport bool yukarin_s_forward(int length, long *phoneme_list, long *speaker_id, float *output);

DllExport bool yukarin_sa_forward(int length, long *vowel_phoneme_list, long *consonant_phoneme_list,
                                  long *start_accent_list, long *end_accent_list, long *start_accent_phrase_list,
                                  long *end_accent_phrase_list, long *speaker_id, float *output);

DllExport bool decode_forward(int length, int phoneme_size, float *f0, float *phoneme, long *speaker_id, float *output);
