#include "mel.hpp"
#include "fft.hpp"  // mt::rfft (vendored; power-of-2 only, kept for future use)
#include "ggml-backend.h"
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace mt {

WhisperMel::WhisperMel(const ModelLoader& m) {
    const auto& c = m.config();
    n_fft_ = c.feat_n_fft;
    hop_   = c.feat_hop;
    n_mels_ = c.feat_size;
    n_bins_ = n_fft_ / 2 + 1;
    n_samples_ = c.feat_n_samples;
    nb_max_frames_ = c.feat_nb_max_frames;

    // Periodic Hann window (divide by n_fft, not n_fft-1).
    window_.resize(n_fft_);
    for (int n = 0; n < n_fft_; ++n)
        window_[n] = 0.5f * (1.0f - std::cos(2.0 * M_PI * n / (double)n_fft_));

    // Real-DFT twiddle tables. Whisper's n_fft (400) is not a power of two, so
    // the radix-2 mt::rfft cannot be used; a direct DFT over the n_bins
    // outputs is exact and cheap enough for the parity harness.
    // X[k] = sum_n x[n] * exp(-2*pi*i * k*n / N)
    cos_.resize((size_t)n_bins_ * n_fft_);
    sin_.resize((size_t)n_bins_ * n_fft_);
    for (int k = 0; k < n_bins_; ++k) {
        for (int n = 0; n < n_fft_; ++n) {
            double ang = 2.0 * M_PI * (double)k * (double)n / (double)n_fft_;
            cos_[(size_t)k * n_fft_ + n] = (float)std::cos(ang);
            sin_[(size_t)k * n_fft_ + n] = (float)std::sin(ang);
        }
    }

    // mel_filters: conceptually (n_mels, n_bins) row-major. Read the raw f32
    // bytes off the backend buffer and reorder into fb_[mel*n_bins + bin],
    // handling either ggml `ne` orientation.
    fb_.assign((size_t)n_mels_ * n_bins_, 0.0f);
    ggml_tensor* t = m.tensor("mel_filters");
    if (t) {
        std::vector<float> raw((size_t)ggml_nelements(t));
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size() * sizeof(float));
        const int64_t ne0 = t->ne[0];
        const int64_t ne1 = t->ne[1];
        if (ne0 == n_bins_ && ne1 == n_mels_) {
            // ne[0]=bins (fastest), ne[1]=mels -> flat[mel*n_bins + bin]
            for (int mel = 0; mel < n_mels_; ++mel)
                for (int bin = 0; bin < n_bins_; ++bin)
                    fb_[(size_t)mel * n_bins_ + bin] = raw[(size_t)mel * n_bins_ + bin];
        } else if (ne0 == n_mels_ && ne1 == n_bins_) {
            // ne[0]=mels (fastest), ne[1]=bins -> flat[bin*n_mels + mel]
            for (int mel = 0; mel < n_mels_; ++mel)
                for (int bin = 0; bin < n_bins_; ++bin)
                    fb_[(size_t)mel * n_bins_ + bin] = raw[(size_t)bin * n_mels_ + mel];
        }
    }
}

void WhisperMel::compute(const std::vector<float>& samples, std::vector<float>& out,
                         int& n_mels, int& n_frames) const {
    n_mels = n_mels_;
    n_frames = nb_max_frames_;

    // Center STFT: reflect-pad by n_fft/2 on both ends (numpy 'reflect', which
    // excludes the edge sample). This makes frame t = t*hop the centered frame
    // and yields exactly nb_max_frames frames (dropping the trailing frame).
    const int pad = n_fft_ / 2;
    const int64_t N = (int64_t)samples.size();
    std::vector<float> x((size_t)N + 2 * pad);
    for (int i = 0; i < pad; ++i) x[i] = samples[pad - i];
    for (int64_t i = 0; i < N; ++i) x[(size_t)pad + i] = samples[(size_t)i];
    for (int i = 0; i < pad; ++i)
        x[(size_t)pad + N + i] = samples[(size_t)(N - 2 - i)];

    out.assign((size_t)n_mels_ * n_frames, 0.0f);
    std::vector<float> frame(n_fft_);
    std::vector<double> power(n_bins_);

    for (int t = 0; t < n_frames; ++t) {
        const int start = t * hop_;
        for (int n = 0; n < n_fft_; ++n) frame[n] = x[(size_t)start + n] * window_[n];

        // Power spectrum |rfft(frame)|^2 over n_bins bins (computed once/frame).
        for (int b = 0; b < n_bins_; ++b) {
            const float* cr = &cos_[(size_t)b * n_fft_];
            const float* sr = &sin_[(size_t)b * n_fft_];
            double re = 0.0, im = 0.0;
            for (int n = 0; n < n_fft_; ++n) {
                re += (double)frame[n] * cr[n];
                im -= (double)frame[n] * sr[n];
            }
            power[b] = re * re + im * im;
        }

        // Mel projection: fb (n_mels x n_bins) @ power, then log10.
        for (int mel = 0; mel < n_mels_; ++mel) {
            const float* fbrow = &fb_[(size_t)mel * n_bins_];
            double acc = 0.0;
            for (int b = 0; b < n_bins_; ++b) acc += (double)fbrow[b] * power[b];
            out[(size_t)mel * n_frames + t] =
                (float)std::log10(std::max(acc, 1e-10));
        }
    }

    // Per-chunk normalization over the full (n_mels, n_frames):
    // mx = max(all); floor = mx - 8; v = max(v, floor); v = (v + 4) / 4.
    float mx = out[0];
    for (float v : out) mx = std::max(mx, v);
    const float floor = mx - 8.0f;
    for (float& v : out) {
        v = std::max(v, floor);
        v = (v + 4.0f) / 4.0f;
    }
}

} // namespace mt
