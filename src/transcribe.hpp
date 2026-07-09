#ifndef MT_TRANSCRIBE_HPP
#define MT_TRANSCRIBE_HPP

// Top-level transcription: wav -> text.
//
// Chains audio_io -> AudioEncoder -> build_input_ids -> fuse_embeds ->
// greedy_generate -> tokenizer.decode, reproducing the reference pipeline.

#include "model_loader.hpp"

#include <string>

namespace mt {

// Transcribe a wav file. Loads and resamples the audio to 16 kHz mono, runs the
// full pipeline, and returns the decoded transcript text (trailing EOS dropped,
// whitespace stripped). Returns an empty string on error.
std::string transcribe_wav(ModelLoader& m, const std::string& wav_path, int max_new);

}  // namespace mt

#endif  // MT_TRANSCRIBE_HPP
