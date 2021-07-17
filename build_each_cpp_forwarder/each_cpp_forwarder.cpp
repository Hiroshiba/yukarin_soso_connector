#include "each_cpp_forwarder.h"

#include <torch/script.h>
#include <torch/torch.h>

#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#define HIHO_DEBUG

#ifdef HIHO_DEBUG
#define tryOrPassStart try
#define tryOrPassEnd                               \
    catch (const std::exception &e) {              \
        std::cout << e.what() << std::endl;        \
        return false;                              \
    }                                              \
    catch (...) {                                  \
        std::cout << "unknown error" << std::endl; \
        return false;                              \
    }
#else
#define tryOrPassStart try
#define tryOrPassEnd  \
    catch (...) {     \
        return false; \
    }
#endif

torch::Device *device;
torch::jit::script::Module yukarin_s_forwarder;
torch::jit::script::Module yukarin_sa_forwarder;
torch::jit::script::Module decode_forwarder;

bool initialize(char *yukarin_s_forwarder_path, char *yukarin_sa_forwarder_path, char *decode_forwarder_path,
                bool use_gpu) {
    tryOrPassStart {
        yukarin_s_forwarder = torch::jit::load(yukarin_s_forwarder_path);
        yukarin_sa_forwarder = torch::jit::load(yukarin_sa_forwarder_path);
        decode_forwarder = torch::jit::load(decode_forwarder_path);
        if (!use_gpu) {
            device = new torch::Device(torch::kCPU);
        } else {
            assert(torch::cuda::is_available());
            device = new torch::Device(torch::kCUDA);
        }
        return true;
    }
    tryOrPassEnd;
}

at::Tensor array_to_tensor(void *data, at::IntArrayRef sizes, at::ScalarType dtype) {
    return torch::from_blob(data, sizes, torch::TensorOptions().dtype(dtype)).to(*device);
}

bool yukarin_s_forward(int length, long *phoneme_list, long *speaker_id, float *output) {
    tryOrPassStart {
        std::vector<torch::jit::IValue> inputs = {
            array_to_tensor(phoneme_list, {length}, torch::kInt64),
            array_to_tensor(speaker_id, {1}, torch::kInt64),
        };
        auto output_tensor = yukarin_s_forwarder.forward(inputs).toTensor().cpu().contiguous();
        std::memcpy(output, output_tensor.data_ptr<float>(), sizeof(float) * output_tensor.numel());
        return true;
    }
    tryOrPassEnd;
}

bool yukarin_sa_forward(int length, long *vowel_phoneme_list, long *consonant_phoneme_list, long *start_accent_list,
                        long *end_accent_list, long *start_accent_phrase_list, long *end_accent_phrase_list,
                        long *speaker_id, float *output) {
    tryOrPassStart {
        std::vector<torch::jit::IValue> inputs = {
            array_to_tensor(vowel_phoneme_list, {1, length}, torch::kInt64),
            array_to_tensor(consonant_phoneme_list, {1, length}, torch::kInt64),
            array_to_tensor(start_accent_list, {1, length}, torch::kInt64),
            array_to_tensor(end_accent_list, {1, length}, torch::kInt64),
            array_to_tensor(start_accent_phrase_list, {1, length}, torch::kInt64),
            array_to_tensor(end_accent_phrase_list, {1, length}, torch::kInt64),
            array_to_tensor(speaker_id, {1}, torch::kInt64),
        };
        auto output_tensor = yukarin_sa_forwarder.forward(inputs).toTensor().cpu().contiguous();
        std::memcpy(output, output_tensor.data_ptr<float>(), sizeof(float) * output_tensor.numel());
        return true;
    }
    tryOrPassEnd;
}

bool decode_forward(int length, int phoneme_size, float *f0, float *phoneme, long *speaker_id, float *output) {
    tryOrPassStart {
        std::vector<torch::jit::IValue> inputs = {
            std::vector<at::Tensor>({array_to_tensor(f0, {length, 1}, torch::kFloat32)}),
            std::vector<at::Tensor>({array_to_tensor(phoneme, {length, phoneme_size}, torch::kFloat32)}),
            array_to_tensor(speaker_id, {1}, torch::kInt64),
        };
        auto output_tensor = decode_forwarder.forward(inputs).toTensor().cpu().contiguous();
        std::memcpy(output, output_tensor.data_ptr<float>(), sizeof(float) * output_tensor.numel());
        return true;
    }
    tryOrPassEnd;
}
