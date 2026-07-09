#pragma once
#include <vector>
#include "model_loader.hpp"
namespace mt {

// Whisper log-mel front-end. Produces the HF `input_features` tensor for a
// single 30 s chunk: row-major [n_mels, n_frames] (n_frames = 3000).
//
// Exact Whisper mel: periodic Hann window, center STFT (reflect-pad by
// n_fft/2), power spectrum, mel-filterbank projection, log10, then per-chunk
// normalization ((max-8 floor), (v+4)/4).
class WhisperMel {
public:
    explicit WhisperMel(const ModelLoader& m);
    void compute(const std::vector<float>& samples, std::vector<float>& out,
                 int& n_mels, int& n_frames) const;
private:
    int n_fft_, hop_, n_mels_, n_bins_, n_samples_, nb_max_frames_;
    std::vector<float> window_;      // periodic Hann, length n_fft
    std::vector<float> fb_;          // [n_mels * n_bins], row-major (mel*n_bins + bin)
    // Real-DFT twiddle tables (n_fft is 400 -> not a power of 2, so we use a
    // direct DFT for the n_bins outputs). cos_/sin_ are [n_bins * n_fft].
    std::vector<float> cos_;         // cos(2*pi*bin*n / n_fft)
    std::vector<float> sin_;         // sin(2*pi*bin*n / n_fft)
};
} // namespace mt
