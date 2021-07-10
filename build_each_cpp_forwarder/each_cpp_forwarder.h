#pragma once

#include <string>
#include <vector>

#if defined(_WIN32) || defined(_WIN64)
// extern "C" DllExport
#define DllExport __declspec(dllexport)
#else
#define DllExport
#endif

DllExport bool initialize(std::string yukarin_s_forwarder_path, std::string yukarin_sa_forwarder_path,
                          std::string decode_forwarder_path);

DllExport bool yukarin_s_forward(int length, long *phoneme_list, long *speaker_id, std::vector<float> *output,
                                 std::vector<long> *output_size);

DllExport bool yukarin_sa_forward(int length, long *vowel_phoneme_list, long *consonant_phoneme_list,
                                  long *start_accent_list, long *end_accent_list, long *start_accent_phrase_list,
                                  long *end_accent_phrase_list, long *speaker_id, std::vector<float> *output,
                                  std::vector<long> *output_size);

DllExport bool decode_forward(int length, int phoneme_size, std::vector<float *> f0_list,
                              std::vector<float *> phoneme_list, long *speaker_id, std::vector<float> *output,
                              std::vector<long> *output_size);
