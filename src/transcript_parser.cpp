#include "transcript_parser.hpp"

#include <cctype>
#include <cstdlib>
#include <string>

namespace mt {

namespace {

bool is_timestamp_char(char ch) {
    return (ch >= '0' && ch <= '9') || ch == '.';
}

bool is_speaker_char(char ch) {
    return ch == 'S' || (ch >= '0' && ch <= '9');
}

bool is_space(char ch) {
    // Matches the ASCII whitespace the reference relies on (str.isspace()).
    return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' ||
           ch == '\f' || ch == '\v';
}

// Returns true and sets out on success; mirrors reference _parse_timestamp
// (only digits and at most one '.', at least one digit).
bool parse_timestamp(const std::string& chars, double& out) {
    if (chars.empty()) return false;
    int dot_count = 0;
    int digit_count = 0;
    for (char ch : chars) {
        if (ch >= '0' && ch <= '9') {
            ++digit_count;
        } else if (ch == '.') {
            ++dot_count;
            if (dot_count > 1) return false;
        } else {
            return false;
        }
    }
    if (digit_count == 0) return false;
    out = std::strtod(chars.c_str(), nullptr);
    return true;
}

// Mirrors reference _parse_speaker: 'S' followed by >=1 digit(s).
bool parse_speaker(const std::string& chars, std::string& out) {
    if (chars.size() < 2 || chars[0] != 'S') return false;
    for (size_t i = 1; i < chars.size(); ++i) {
        if (!(chars[i] >= '0' && chars[i] <= '9')) return false;
    }
    out = chars;
    return true;
}

} // namespace

void TranscriptStreamParser::reset() {
    state_ = SEEK_START;
    token_.clear();
    text_.clear();
    pending_after_end_.clear();
    has_start_ = false; start_ = 0.0;
    has_end_ = false; end_ = 0.0;
    end_token_.clear();
    has_speaker_ = false; speaker_.clear();
}

std::vector<TranscriptSegment> TranscriptStreamParser::feed(const std::string& chunk) {
    std::vector<TranscriptSegment> segments;
    feed_into(chunk, [&](const TranscriptSegment& s) { segments.push_back(s); });
    return segments;
}

void TranscriptStreamParser::feed_into(
    const std::string& chunk,
    const std::function<void(const TranscriptSegment&)>& emit) {
    for (char ch : chunk) {
        switch (state_) {
            case SEEK_START:          seek_start(ch); break;
            case READ_START:          read_start(ch); break;
            case EXPECT_SPEAKER_OPEN: expect_speaker_open(ch); break;
            case READ_SPEAKER:        read_speaker(ch); break;
            case READ_TEXT:           read_text(ch); break;
            case READ_END:            read_end(ch, emit); break;
            case AFTER_END:           after_end(ch, emit); break;
        }
    }
}

std::vector<TranscriptSegment> TranscriptStreamParser::close() {
    std::vector<TranscriptSegment> segments;
    close_into([&](const TranscriptSegment& s) { segments.push_back(s); });
    return segments;
}

void TranscriptStreamParser::close_into(
    const std::function<void(const TranscriptSegment&)>& emit) {
    if (state_ == AFTER_END) {
        emit_segment(emit);
    }
    reset();
}

void TranscriptStreamParser::seek_start(char ch) {
    if (ch == '[') {
        token_.clear();
        state_ = READ_START;
    }
}

void TranscriptStreamParser::read_start(char ch) {
    if (ch == ']') {
        double start;
        if (!parse_timestamp(token_, start)) {
            reset();
            return;
        }
        start_ = start;
        has_start_ = true;
        state_ = EXPECT_SPEAKER_OPEN;
        token_.clear();
        return;
    }
    if (is_timestamp_char(ch)) {
        token_.push_back(ch);
        if (token_.size() <= 32) return;
    }
    reset();
    if (ch == '[') state_ = READ_START;
}

void TranscriptStreamParser::expect_speaker_open(char ch) {
    if (ch == '[') {
        token_.clear();
        state_ = READ_SPEAKER;
    } else if (!is_space(ch)) {
        reset();
    }
}

void TranscriptStreamParser::read_speaker(char ch) {
    if (ch == ']') {
        std::string speaker;
        if (!parse_speaker(token_, speaker)) {
            reset();
            return;
        }
        speaker_ = speaker;
        has_speaker_ = true;
        text_.clear();
        state_ = READ_TEXT;
        token_.clear();
        return;
    }
    if (is_speaker_char(ch)) {
        token_.push_back(ch);
        if (token_.size() <= 16) return;
    }
    reset();
    if (ch == '[') state_ = READ_START;
}

void TranscriptStreamParser::read_text(char ch) {
    if (ch == '[') {
        token_.clear();
        state_ = READ_END;
    } else {
        text_.push_back(ch);
    }
}

void TranscriptStreamParser::read_end(
    char ch, const std::function<void(const TranscriptSegment&)>& /*emit*/) {
    if (ch == ']') {
        double end;
        if (parse_timestamp(token_, end) && has_start_ && end >= start_) {
            end_ = end;
            has_end_ = true;
            end_token_ = token_;
            pending_after_end_.clear();
            state_ = AFTER_END;
        } else {
            text_.push_back('[');
            text_.append(token_);
            text_.push_back(']');
            state_ = READ_TEXT;
        }
        token_.clear();
        return;
    }
    if (is_timestamp_char(ch)) {
        token_.push_back(ch);
        if (token_.size() <= 32) return;
    }
    text_.push_back('[');
    text_.append(token_);
    text_.push_back(ch);
    token_.clear();
    state_ = READ_TEXT;
}

void TranscriptStreamParser::after_end(
    char ch, const std::function<void(const TranscriptSegment&)>& emit) {
    if (ch == '[') {
        emit_segment(emit);
        token_.clear();
        state_ = READ_START;
        return;
    }
    if (is_space(ch)) {
        pending_after_end_.push_back(ch);
        return;
    }
    text_.push_back('[');
    text_.append(end_token_);
    text_.push_back(']');
    text_.append(pending_after_end_);
    text_.push_back(ch);
    pending_after_end_.clear();
    has_end_ = false;
    end_token_.clear();
    state_ = READ_TEXT;
}

void TranscriptStreamParser::emit_segment(
    const std::function<void(const TranscriptSegment&)>& emit) {
    if (!has_start_ || !has_end_ || !has_speaker_) {
        reset();
        return;
    }
    std::string text = text_;
    if (strip_text_) {
        size_t a = text.find_first_not_of(" \t\n\r\f\v");
        if (a == std::string::npos) {
            text.clear();
        } else {
            size_t b = text.find_last_not_of(" \t\n\r\f\v");
            text = text.substr(a, b - a + 1);
        }
    }
    if (!text.empty() || !skip_empty_) {
        TranscriptSegment seg;
        seg.start = start_;
        seg.end = end_;
        seg.speaker = speaker_;
        seg.text = text;
        emit(seg);
    }
    token_.clear();
    text_.clear();
    pending_after_end_.clear();
    has_start_ = false; start_ = 0.0;
    has_end_ = false; end_ = 0.0;
    end_token_.clear();
    has_speaker_ = false; speaker_.clear();
    state_ = SEEK_START;
}

std::vector<TranscriptSegment> parse_transcript(const std::string& text,
                                                bool strip_text, bool skip_empty) {
    TranscriptStreamParser parser(strip_text, skip_empty);
    std::vector<TranscriptSegment> segments = parser.feed(text);
    std::vector<TranscriptSegment> tail = parser.close();
    segments.insert(segments.end(), tail.begin(), tail.end());
    return segments;
}

} // namespace mt
