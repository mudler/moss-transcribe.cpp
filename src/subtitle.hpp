#pragma once
// 1:1 C++ port of moss_transcribe_diarize/subtitle/{models,postprocess,export}.py
// Speaker-aware subtitle normalization + SRT/ASS/JSON export. Pure string/
// interval logic. UTF-8 aware where the reference counts Unicode codepoints.
#include "transcript_parser.hpp"

#include <map>
#include <string>
#include <vector>

namespace mt {

struct SubtitleSegment {
    std::string id;
    double start = 0.0;
    double end = 0.0;
    std::string speaker;
    std::string text;

    SubtitleSegment() = default;
    SubtitleSegment(std::string id_, double start_, double end_,
                    std::string speaker_, std::string text_)
        : id(std::move(id_)), start(start_), end(end_),
          speaker(std::move(speaker_)), text(std::move(text_)) {}
};

struct SubtitleStyle {
    std::string font_name = "Noto Sans CJK SC";
    int font_size = 0;            // 0 == "None" (auto from video height)
    int alignment = 2;
    int margin_v = 56;
    bool show_speaker = true;
    bool speaker_colors = true;
    std::string primary_color = "&H00FFFFFF";
    std::string outline_color = "&H00000000";
    std::string back_color = "&H64000000";
    int outline = 3;
    int shadow = 1;
    std::map<std::string, std::string> speaker_names;  // empty == None
};

// Postprocess defaults (mirror the reference module constants).
constexpr double DEFAULT_MIN_DURATION = 1.0;
constexpr double DEFAULT_MAX_DURATION = 6.0;
constexpr int    DEFAULT_MAX_CHARS = 24;
constexpr double DEFAULT_MERGE_GAP = 0.3;

// _prepare -> _fix_overlaps -> _merge_adjacent -> _split_long -> _fix_overlaps.
std::vector<SubtitleSegment> normalize_segments(
    const std::vector<SubtitleSegment>& segments,
    double min_duration = DEFAULT_MIN_DURATION,
    double max_duration = DEFAULT_MAX_DURATION,
    int max_chars = DEFAULT_MAX_CHARS,
    double merge_gap = DEFAULT_MERGE_GAP,
    bool regenerate_ids = false);

// Build subtitle segments (ids seg_0001..) from a raw transcript string.
std::vector<SubtitleSegment> subtitle_segments_from_transcript(
    const std::string& transcript,
    bool postprocess = true,
    double min_duration = DEFAULT_MIN_DURATION,
    double max_duration = DEFAULT_MAX_DURATION,
    int max_chars = DEFAULT_MAX_CHARS,
    double merge_gap = DEFAULT_MERGE_GAP);

std::vector<SubtitleSegment> subtitle_segments_from_transcript_segments(
    const std::vector<TranscriptSegment>& segments,
    bool postprocess = true,
    double min_duration = DEFAULT_MIN_DURATION,
    double max_duration = DEFAULT_MAX_DURATION,
    int max_chars = DEFAULT_MAX_CHARS,
    double merge_gap = DEFAULT_MERGE_GAP);

// Convert payloads without timing edits (parity with coerce_subtitle_segments).
std::vector<SubtitleSegment> coerce_subtitle_segments(
    const std::vector<SubtitleSegment>& segments);

// ---- Export ----------------------------------------------------------------
std::string to_srt(const std::vector<SubtitleSegment>& segments,
                   bool show_speaker = true,
                   const std::map<std::string, std::string>& speaker_names = {});

std::string to_ass(const std::vector<SubtitleSegment>& segments,
                   const SubtitleStyle& style = SubtitleStyle(),
                   int video_width = 1920,
                   int video_height = 1080);

std::string to_json(const std::vector<SubtitleSegment>& segments, int indent = 2);

std::string format_srt_time(double seconds);
std::string format_ass_time(double seconds);

} // namespace mt
