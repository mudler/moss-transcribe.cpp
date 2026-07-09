#pragma once
// 1:1 C++ port of moss_transcribe_diarize/transcript_parser.py
// Parses compact MOSS transcript output of the form:
//     [start][Sxx]text[end][start][Sxx]text[end]...
// into TranscriptSegment records. A character-scanning state machine (no regex),
// identical in behavior to the reference TranscriptStreamParser.
#include <string>
#include <vector>
#include <functional>

namespace mt {

struct TranscriptSegment {
    double start = 0.0;
    double end = 0.0;
    std::string speaker;
    std::string text;

    bool operator==(const TranscriptSegment& o) const {
        return start == o.start && end == o.end &&
               speaker == o.speaker && text == o.text;
    }
    bool operator!=(const TranscriptSegment& o) const { return !(*this == o); }
};

// Streaming parser (optional but mirrors the reference for per-chunk parity).
class TranscriptStreamParser {
public:
    explicit TranscriptStreamParser(bool strip_text = true, bool skip_empty = true)
        : strip_text_(strip_text), skip_empty_(skip_empty) {}

    void reset();
    // Consume a chunk (bytes/UTF-8), return newly completed segments.
    std::vector<TranscriptSegment> feed(const std::string& chunk);
    void feed_into(const std::string& chunk,
                   const std::function<void(const TranscriptSegment&)>& emit);
    // Finish the stream and return a final segment if one is complete.
    std::vector<TranscriptSegment> close();
    void close_into(const std::function<void(const TranscriptSegment&)>& emit);

private:
    enum State {
        SEEK_START = 0, READ_START = 1, EXPECT_SPEAKER_OPEN = 2,
        READ_SPEAKER = 3, READ_TEXT = 4, READ_END = 5, AFTER_END = 6
    };

    void seek_start(char ch);
    void read_start(char ch);
    void expect_speaker_open(char ch);
    void read_speaker(char ch);
    void read_text(char ch);
    void read_end(char ch, const std::function<void(const TranscriptSegment&)>& emit);
    void after_end(char ch, const std::function<void(const TranscriptSegment&)>& emit);
    void emit_segment(const std::function<void(const TranscriptSegment&)>& emit);

    bool strip_text_;
    bool skip_empty_;
    State state_ = SEEK_START;
    std::string token_;
    std::string text_;
    std::string pending_after_end_;
    bool has_start_ = false;
    double start_ = 0.0;
    bool has_end_ = false;
    double end_ = 0.0;
    std::string end_token_;
    bool has_speaker_ = false;
    std::string speaker_;
};

// Batch convenience: parse an entire transcript string.
std::vector<TranscriptSegment> parse_transcript(const std::string& text,
                                                bool strip_text = true,
                                                bool skip_empty = true);

} // namespace mt
