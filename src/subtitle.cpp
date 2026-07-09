#include "subtitle.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

namespace mt {

namespace {

// ---- UTF-8 helpers ---------------------------------------------------------
// Split a UTF-8 string into a vector of codepoint substrings.
std::vector<std::string> split_codepoints(const std::string& s) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        size_t len = 1;
        if (c < 0x80) len = 1;
        else if ((c >> 5) == 0x6) len = 2;
        else if ((c >> 4) == 0xE) len = 3;
        else if ((c >> 3) == 0x1E) len = 4;
        else len = 1;  // invalid lead byte: treat as single
        if (i + len > s.size()) len = 1;
        out.push_back(s.substr(i, len));
        i += len;
    }
    return out;
}

size_t cp_len(const std::string& s) {
    return split_codepoints(s).size();
}

bool cp_is_ascii(const std::string& cp) {
    return cp.size() == 1 && static_cast<unsigned char>(cp[0]) < 0x80;
}

bool is_ws(char ch) {
    return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' ||
           ch == '\f' || ch == '\v';
}

std::string strip(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && is_ws(s[a])) ++a;
    while (b > a && is_ws(s[b - 1])) --b;
    return s.substr(a, b - a);
}

std::string seg_id(int index) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "seg_%04d", index);
    return std::string(buf);
}

const std::set<std::string>& punctuation() {
    // "。！？!?；;，,、 " -- CJK + ASCII punctuation plus a trailing space.
    static const std::set<std::string> p = {
        "\xE3\x80\x82",  // 。 U+3002
        "\xEF\xBC\x81",  // ！ U+FF01
        "\xEF\xBC\x9F",  // ？ U+FF1F
        "!",
        "?",
        "\xEF\xBC\x9B",  // ； U+FF1B
        ";",
        "\xEF\xBC\x8C",  // ， U+FF0C
        ",",
        "\xE3\x80\x81",  // 、 U+3001
        " ",
    };
    return p;
}

// ---- _join_text ------------------------------------------------------------
std::string join_text(const std::string& left, const std::string& right) {
    if (left.empty()) return right;
    if (right.empty()) return left;
    // left[-1].isascii() and right[0].isascii()
    std::vector<std::string> lcp = split_codepoints(left);
    std::vector<std::string> rcp = split_codepoints(right);
    if (cp_is_ascii(lcp.back()) && cp_is_ascii(rcp.front())) {
        return left + " " + right;
    }
    return left + right;
}

// ---- _split_text -----------------------------------------------------------
std::vector<std::string> split_text(const std::string& text_in, int max_chars) {
    std::string text = strip(text_in);
    std::vector<std::string> cps = split_codepoints(text);
    if ((int)cps.size() <= max_chars) {
        return {text};
    }

    const std::set<std::string>& punct = punctuation();
    std::vector<std::string> chunks;
    std::string current;
    int current_len = 0;
    for (const std::string& ch : cps) {
        current += ch;
        ++current_len;
        bool should_cut = current_len >= max_chars ||
                          (punct.count(ch) && current_len >= max_chars / 2);
        if (should_cut) {
            chunks.push_back(strip(current));
            current.clear();
            current_len = 0;
        }
    }
    if (!current.empty()) {
        chunks.push_back(strip(current));
    }

    std::vector<std::string> compact;
    for (const std::string& chunk : chunks) {
        if (chunk.empty()) continue;
        if (!compact.empty() &&
            (int)(cp_len(compact.back()) + cp_len(chunk)) <= max_chars) {
            compact.back() = join_text(compact.back(), chunk);
        } else {
            compact.push_back(chunk);
        }
    }
    return compact;
}

// ---- normalize pipeline stages --------------------------------------------
std::vector<SubtitleSegment> prepare_segments(const std::vector<SubtitleSegment>& segments) {
    std::vector<SubtitleSegment> prepared;
    int index = 0;
    for (const SubtitleSegment& item : segments) {
        ++index;
        std::string text = strip(item.text);
        if (text.empty()) continue;
        double start = std::max(0.0, item.start);
        double end = std::max(start, item.end);
        prepared.emplace_back(
            item.id.empty() ? seg_id(index) : item.id,
            start, end,
            item.speaker.empty() ? "S00" : item.speaker,
            text);
    }
    std::stable_sort(prepared.begin(), prepared.end(),
                     [](const SubtitleSegment& a, const SubtitleSegment& b) {
                         if (a.start != b.start) return a.start < b.start;
                         return a.end < b.end;
                     });
    return prepared;
}

std::vector<SubtitleSegment> fix_overlaps(const std::vector<SubtitleSegment>& segments,
                                          double min_duration) {
    double cursor = 0.0;
    std::vector<SubtitleSegment> fixed;
    for (const SubtitleSegment& segment : segments) {
        double start = std::max(segment.start, cursor);
        double end = std::max(segment.end, start + min_duration);
        fixed.emplace_back(segment.id, start, end, segment.speaker, segment.text);
        cursor = end;
    }
    return fixed;
}

std::vector<SubtitleSegment> merge_adjacent(const std::vector<SubtitleSegment>& segments,
                                            double merge_gap, int max_chars) {
    if (segments.empty()) return {};
    std::vector<SubtitleSegment> merged;
    merged.push_back(segments[0]);
    for (size_t i = 1; i < segments.size(); ++i) {
        const SubtitleSegment& segment = segments[i];
        SubtitleSegment& previous = merged.back();
        double gap = segment.start - previous.end;
        std::string combined_text = join_text(previous.text, segment.text);
        bool can_merge = previous.speaker == segment.speaker &&
                         gap >= 0 && gap <= merge_gap &&
                         (int)cp_len(combined_text) <= max_chars * 2;
        if (can_merge) {
            merged.back() = SubtitleSegment(
                previous.id, previous.start,
                std::max(previous.end, segment.end),
                previous.speaker, combined_text);
        } else {
            merged.push_back(segment);
        }
    }
    return merged;
}

std::vector<SubtitleSegment> split_long_segments(const std::vector<SubtitleSegment>& segments,
                                                 double min_duration, double max_duration,
                                                 int max_chars) {
    std::vector<SubtitleSegment> output;
    for (const SubtitleSegment& segment : segments) {
        double duration = segment.end - segment.start;
        if (duration <= max_duration && (int)cp_len(segment.text) <= max_chars) {
            output.push_back(segment);
            continue;
        }
        std::vector<std::string> chunks = split_text(segment.text, max_chars);
        if (chunks.size() <= 1) {
            output.push_back(segment);
            continue;
        }
        int total_chars = 0;
        for (const std::string& chunk : chunks) {
            total_chars += std::max((int)cp_len(chunk), 1);
        }
        double cursor = segment.start;
        for (size_t index = 0; index < chunks.size(); ++index) {
            const std::string& chunk = chunks[index];
            double end;
            if (index == chunks.size() - 1) {
                end = segment.end;
            } else {
                double ratio = (double)std::max((int)cp_len(chunk), 1) / (double)total_chars;
                end = cursor + std::max(min_duration, duration * ratio);
                end = std::min(end,
                    segment.end - min_duration * (double)(chunks.size() - index - 1));
            }
            char idbuf[64];
            std::snprintf(idbuf, sizeof(idbuf), "%s_%zu", segment.id.c_str(), index + 1);
            output.emplace_back(std::string(idbuf), cursor,
                                std::max(end, cursor + min_duration),
                                segment.speaker, chunk);
            cursor = output.back().end;
        }
    }
    return output;
}

// ---- number / time formatting ---------------------------------------------
std::string fmt_json_number(double v) {
    // Mirror Python json.dumps(float) == repr(float): shortest round-trip.
    char buf[64];
    for (int prec = 1; prec <= 17; ++prec) {
        std::snprintf(buf, sizeof(buf), "%.*g", prec, v);
        if (std::strtod(buf, nullptr) == v) break;
    }
    std::string s(buf);
    // Ensure it reads as a float (json.dumps renders 2.0 not 2).
    if (s.find_first_of(".eEnN") == std::string::npos) {
        s += ".0";
    }
    return s;
}

std::string json_escape(const std::string& s) {
    std::string out;
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += (char)c;  // ensure_ascii=False: pass UTF-8 through
                }
        }
    }
    return out;
}

// ---- ASS helpers -----------------------------------------------------------
std::string speaker_style_name(const std::string& speaker) {
    std::string out = "Speaker_";
    for (char ch : speaker) {
        bool alnum = (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') ||
                     (ch >= '0' && ch <= '9');
        out += alnum ? ch : '_';
    }
    return out;
}

std::string display_text(const SubtitleSegment& segment, bool show_speaker,
                         const std::map<std::string, std::string>& speaker_names) {
    if (!show_speaker || segment.speaker.empty()) {
        return segment.text;
    }
    std::string speaker = segment.speaker;
    auto it = speaker_names.find(segment.speaker);
    if (it != speaker_names.end() && !it->second.empty()) {
        speaker = it->second;
    }
    return speaker + ": " + segment.text;
}

std::string ass_escape(const std::string& text) {
    std::string out;
    for (size_t i = 0; i < text.size(); ++i) {
        char c = text[i];
        if (c == '\\') out += "\\\\";
        else if (c == '{') out += '(';
        else if (c == '}') out += ')';
        else if (c == '\n') out += "\\N";
        else out += c;
    }
    return out;
}

std::string ass_style_line(const std::string& name, const SubtitleStyle& style,
                           int font_size, const std::string& primary_color) {
    char buf[512];
    std::snprintf(buf, sizeof(buf),
        "Style: %s,%s,%d,%s,&H000000FF,%s,%s,0,0,0,0,100,100,0,0,1,%d,%d,%d,48,48,%d,1",
        name.c_str(), style.font_name.c_str(), font_size, primary_color.c_str(),
        style.outline_color.c_str(), style.back_color.c_str(),
        style.outline, style.shadow, style.alignment, style.margin_v);
    return std::string(buf);
}

const char* SPEAKER_COLORS[] = {
    "&H00FFFFFF", "&H005BE7FF", "&H0086F28F", "&H00BBA7FF",
    "&H0000D7FF", "&H00FFB56B", "&H00FF8EDB", "&H00D8D8D8",
};
constexpr int NUM_SPEAKER_COLORS = 8;

} // namespace

// ---- public API ------------------------------------------------------------
std::vector<SubtitleSegment> normalize_segments(
    const std::vector<SubtitleSegment>& segments,
    double min_duration, double max_duration, int max_chars,
    double merge_gap, bool regenerate_ids) {
    std::vector<SubtitleSegment> prepared = prepare_segments(segments);
    prepared = fix_overlaps(prepared, min_duration);
    prepared = merge_adjacent(prepared, merge_gap, max_chars);
    prepared = split_long_segments(prepared, min_duration, max_duration, max_chars);
    prepared = fix_overlaps(prepared, min_duration);
    if (regenerate_ids) {
        for (size_t i = 0; i < prepared.size(); ++i) {
            prepared[i].id = seg_id((int)i + 1);
        }
    }
    return prepared;
}

std::vector<SubtitleSegment> subtitle_segments_from_transcript_segments(
    const std::vector<TranscriptSegment>& segments, bool postprocess,
    double min_duration, double max_duration, int max_chars, double merge_gap) {
    std::vector<SubtitleSegment> subtitle_segments;
    int index = 0;
    for (const TranscriptSegment& segment : segments) {
        ++index;
        subtitle_segments.emplace_back(seg_id(index), segment.start, segment.end,
                                       segment.speaker, segment.text);
    }
    if (!postprocess) return subtitle_segments;
    return normalize_segments(subtitle_segments, min_duration, max_duration,
                              max_chars, merge_gap, /*regenerate_ids=*/true);
}

std::vector<SubtitleSegment> subtitle_segments_from_transcript(
    const std::string& transcript, bool postprocess,
    double min_duration, double max_duration, int max_chars, double merge_gap) {
    return subtitle_segments_from_transcript_segments(
        parse_transcript(transcript), postprocess,
        min_duration, max_duration, max_chars, merge_gap);
}

std::vector<SubtitleSegment> coerce_subtitle_segments(
    const std::vector<SubtitleSegment>& segments) {
    std::vector<SubtitleSegment> coerced;
    int index = 0;
    for (const SubtitleSegment& segment : segments) {
        ++index;
        coerced.emplace_back(
            segment.id.empty() ? seg_id(index) : segment.id,
            segment.start, segment.end,
            segment.speaker.empty() ? "S00" : segment.speaker,
            segment.text);
    }
    return coerced;
}

std::string format_srt_time(double seconds) {
    long long ms = std::max<long long>(0, (long long)std::llrint(seconds * 1000.0));
    long long hours = ms / 3600000; ms %= 3600000;
    long long minutes = ms / 60000; ms %= 60000;
    long long secs = ms / 1000; ms %= 1000;
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%02lld:%02lld:%02lld,%03lld",
                  hours, minutes, secs, ms);
    return std::string(buf);
}

std::string format_ass_time(double seconds) {
    long long cs = std::max<long long>(0, (long long)std::llrint(seconds * 100.0));
    long long hours = cs / 360000; cs %= 360000;
    long long minutes = cs / 6000; cs %= 6000;
    long long secs = cs / 100; cs %= 100;
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%lld:%02lld:%02lld.%02lld",
                  hours, minutes, secs, cs);
    return std::string(buf);
}

std::string to_srt(const std::vector<SubtitleSegment>& segments, bool show_speaker,
                   const std::map<std::string, std::string>& speaker_names) {
    std::vector<std::string> blocks;
    int index = 0;
    for (const SubtitleSegment& segment : segments) {
        ++index;
        std::string text = display_text(segment, show_speaker, speaker_names);
        char idx[16];
        std::snprintf(idx, sizeof(idx), "%d", index);
        std::string block = std::string(idx) + "\n" +
            format_srt_time(segment.start) + " --> " + format_srt_time(segment.end) +
            "\n" + text;
        blocks.push_back(block);
    }
    std::string out;
    for (size_t i = 0; i < blocks.size(); ++i) {
        if (i) out += "\n\n";
        out += blocks[i];
    }
    if (!blocks.empty()) out += "\n";
    return out;
}

std::string to_ass(const std::vector<SubtitleSegment>& segments,
                   const SubtitleStyle& style, int video_width, int video_height) {
    int font_size = style.font_size != 0
                        ? style.font_size
                        : std::max<long long>(24, std::llrint(video_height * 0.045));

    std::set<std::string> speaker_set;
    for (const SubtitleSegment& s : segments) speaker_set.insert(s.speaker);
    std::vector<std::string> speakers(speaker_set.begin(), speaker_set.end());

    std::vector<std::string> style_lines;
    style_lines.push_back(ass_style_line("Default", style, font_size, style.primary_color));
    if (style.speaker_colors) {
        for (size_t i = 0; i < speakers.size(); ++i) {
            const char* color = SPEAKER_COLORS[i % NUM_SPEAKER_COLORS];
            style_lines.push_back(ass_style_line(speaker_style_name(speakers[i]),
                                                 style, font_size, color));
        }
    }

    std::vector<std::string> dialogue_lines;
    for (const SubtitleSegment& segment : segments) {
        std::string style_name = style.speaker_colors
                                     ? speaker_style_name(segment.speaker)
                                     : "Default";
        std::string text = ass_escape(
            display_text(segment, style.show_speaker, style.speaker_names));
        dialogue_lines.push_back(
            "Dialogue: 0," + format_ass_time(segment.start) + "," +
            format_ass_time(segment.end) + "," + style_name + ",,0,0,0,," + text);
    }

    std::vector<std::string> lines;
    lines.push_back("[Script Info]");
    lines.push_back("ScriptType: v4.00+");
    lines.push_back("WrapStyle: 2");
    lines.push_back("ScaledBorderAndShadow: yes");
    lines.push_back("PlayResX: " + std::to_string(video_width));
    lines.push_back("PlayResY: " + std::to_string(video_height));
    lines.push_back("");
    lines.push_back("[V4+ Styles]");
    lines.push_back(
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding");
    for (const std::string& l : style_lines) lines.push_back(l);
    lines.push_back("");
    lines.push_back("[Events]");
    lines.push_back("Format: Layer, Start, End, Style, Name, MarginL, MarginR, "
                    "MarginV, Effect, Text");
    for (const std::string& l : dialogue_lines) lines.push_back(l);
    lines.push_back("");

    std::string out;
    for (size_t i = 0; i < lines.size(); ++i) {
        if (i) out += "\n";
        out += lines[i];
    }
    return out;
}

std::string to_json(const std::vector<SubtitleSegment>& segments, int indent) {
    std::string pad(indent, ' ');
    if (segments.empty()) return "[]\n";
    std::string out = "[\n";
    for (size_t i = 0; i < segments.size(); ++i) {
        const SubtitleSegment& s = segments[i];
        out += pad + "{\n";
        out += pad + pad + "\"id\": \"" + json_escape(s.id) + "\",\n";
        out += pad + pad + "\"start\": " + fmt_json_number(s.start) + ",\n";
        out += pad + pad + "\"end\": " + fmt_json_number(s.end) + ",\n";
        out += pad + pad + "\"speaker\": \"" + json_escape(s.speaker) + "\",\n";
        out += pad + pad + "\"text\": \"" + json_escape(s.text) + "\"\n";
        out += pad + "}";
        if (i + 1 < segments.size()) out += ",";
        out += "\n";
    }
    out += "]\n";
    return out;
}

} // namespace mt
