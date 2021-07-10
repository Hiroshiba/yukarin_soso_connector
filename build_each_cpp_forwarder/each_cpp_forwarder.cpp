#include <torch/script.h>

#include <iostream>
#include <memory>
#include <string>

torch::jit::script::Module yukarin_s_forwarder;
torch::jit::script::Module yukarin_sa_forwarder;
torch::jit::script::Module decode_forwarder;

bool initialize(std::string yukarin_s_forwarder_path, std::string yukarin_sa_forwarder_path,
                std::string decode_forwarder_path) {
    try {
        yukarin_s_forwarder = torch::jit::load(yukarin_s_forwarder_path);
        yukarin_sa_forwarder = torch::jit::load(yukarin_sa_forwarder_path);
        decode_forwarder = torch::jit::load(decode_forwarder_path);
        return true;
    } catch (...) {
        return false;
    }
}

at::Tensor array_to_tensor(void *data, at::IntArrayRef sizes, at::ScalarType dtype) {
    return torch::from_blob(data, sizes, torch::TensorOptions().dtype(dtype));
}

template <class T>
std::vector<at::Tensor> arrays_to_tensors(std::vector<T *> datas, at::IntArrayRef sizes, at::ScalarType dtype) {
    std::vector<at::Tensor> tensors(datas.size());
    std::transform(datas.begin(), datas.end(), tensors.begin(),
                   [sizes, dtype](T *data) { return array_to_tensor(data, sizes, dtype); });
    return tensors;
}

bool yukarin_s_forward(int length, long *phoneme_list, long *speaker_id, std::vector<float> *output,
                       std::vector<long> *output_size) {
    try {
        std::vector<torch::jit::IValue> inputs = {
            array_to_tensor(phoneme_list, {length}, torch::kInt64),
            array_to_tensor(speaker_id, {1}, torch::kInt64),
        };
        auto output_tensor = yukarin_s_forwarder.forward(inputs).toTensor().contiguous();
        *output = std::vector<float>(output_tensor.data_ptr<float>(),
                                     output_tensor.data_ptr<float>() + output_tensor.numel());
        *output_size = std::vector<long>(output_tensor.sizes().data(),
                                         output_tensor.sizes().data() + output_tensor.sizes().size());
        return true;
    } catch (...) {
        return false;
    }
}

bool yukarin_sa_forward(int length, long *vowel_phoneme_list, long *consonant_phoneme_list, long *start_accent_list,
                        long *end_accent_list, long *start_accent_phrase_list, long *end_accent_phrase_list,
                        long *speaker_id, std::vector<float> *output, std::vector<long> *output_size) {
    try {
        std::vector<torch::jit::IValue> inputs = {
            array_to_tensor(vowel_phoneme_list, {1, length}, torch::kInt64),
            array_to_tensor(consonant_phoneme_list, {1, length}, torch::kInt64),
            array_to_tensor(start_accent_list, {1, length}, torch::kInt64),
            array_to_tensor(end_accent_list, {1, length}, torch::kInt64),
            array_to_tensor(start_accent_phrase_list, {1, length}, torch::kInt64),
            array_to_tensor(end_accent_phrase_list, {1, length}, torch::kInt64),
            array_to_tensor(speaker_id, {1}, torch::kInt64),
        };
        auto output_tensor = yukarin_sa_forwarder.forward(inputs).toTensor().contiguous();
        *output = std::vector<float>(output_tensor.data_ptr<float>(),
                                     output_tensor.data_ptr<float>() + output_tensor.numel());
        *output_size = std::vector<long>(output_tensor.sizes().data(),
                                         output_tensor.sizes().data() + output_tensor.sizes().size());
        return true;
    } catch (...) {
        return false;
    }
}

bool decode_forward(int length, int phoneme_size, std::vector<float *> f0_list, std::vector<float *> phoneme_list,
                    long *speaker_id, std::vector<float> *output, std::vector<long> *output_size) {
    try {
        std::vector<torch::jit::IValue> inputs = {
            arrays_to_tensors(f0_list, {length, 1}, torch::kFloat32),
            arrays_to_tensors(phoneme_list, {length, phoneme_size}, torch::kFloat32),
            array_to_tensor(speaker_id, {1}, torch::kInt64),
        };
        auto output_tensor = decode_forwarder.forward(inputs).toTensor().contiguous();
        *output = std::vector<float>(output_tensor.data_ptr<float>(),
                                     output_tensor.data_ptr<float>() + output_tensor.numel());
        *output_size = std::vector<long>(output_tensor.sizes().data(),
                                         output_tensor.sizes().data() + output_tensor.sizes().size());
        return true;
    } catch (...) {
        return false;
    }
}
