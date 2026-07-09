// Ports moss_transcribe_diarize/tests/test_subtitle_export.py 1:1.
#include "subtitle.hpp"

#include <cstdio>
#include <map>
#include <string>
#include <vector>

using mt::SubtitleSegment;
using mt::SubtitleStyle;

static int g_fail = 0;

static void expect(bool cond, const char* name) {
    if (!cond) { ++g_fail; std::fprintf(stderr, "FAIL %s\n", name); }
    else std::printf("ok %s\n", name);
}

static bool has(const std::string& hay, const std::string& needle) {
    return hay.find(needle) != std::string::npos;
}

int main() {
    // test_time_formatters
    expect(mt::format_srt_time(3661.234) == "01:01:01,234" &&
           mt::format_ass_time(3661.23) == "1:01:01.23", "time_formatters");

    // test_export_srt
    {
        std::string t = mt::to_srt({{"seg_0001", 0.5, 2.0, "S01", "hello"}});
        expect(has(t, "00:00:00,500 --> 00:00:02,000") && has(t, "S01: hello"),
               "export_srt");
    }

    // test_export_srt_with_speaker_names
    {
        std::map<std::string, std::string> names = {{"S01", "Alice"}};
        std::string t = mt::to_srt({{"seg_0001", 0.5, 2.0, "S01", "hello"}},
                                   /*show_speaker=*/true, names);
        expect(has(t, "Alice: hello"), "export_srt_with_speaker_names");
    }

    // test_export_ass
    {
        SubtitleStyle style;
        style.font_size = 42;
        style.show_speaker = false;
        std::string t = mt::to_ass({{"seg_0001", 0.5, 2.0, "S01", "hello"}},
                                   style, /*video_width=*/1280, /*video_height=*/720);
        expect(has(t, "PlayResX: 1280") &&
               has(t, "Style: Speaker_S01,Noto Sans CJK SC,42") &&
               has(t, "Dialogue: 0,0:00:00.50,0:00:02.00,Speaker_S01") &&
               has(t, "hello"),
               "export_ass");
    }

    // test_export_ass_with_speaker_names
    {
        SubtitleStyle style;
        style.font_size = 42;
        style.speaker_names = {{"S01", "Alice"}};
        std::string t = mt::to_ass({{"seg_0001", 0.5, 2.0, "S01", "hello"}},
                                   style, 1280, 720);
        expect(has(t, "Alice: hello"), "export_ass_with_speaker_names");
    }

    // test_export_json
    {
        std::string t = mt::to_json({{"seg_0001", 0, 1, "S01", "hello"}});
        expect(has(t, "\"id\": \"seg_0001\"") && has(t, "\"text\": \"hello\""),
               "export_json");
    }

    // Integer-valued timestamps >= 10 must render like Python json.dumps
    // (60.0 / 132.0, never 6e+01) -- regression lock for fmt_json_number.
    {
        std::string t = mt::to_json({{"seg_0001", 60, 132, "S01", "x"}});
        expect(has(t, "\"start\": 60.0") && has(t, "\"end\": 132.0") &&
                   !has(t, "e+0") && !has(t, "E+0"),
               "export_json_integer_timestamps");
    }

    if (g_fail) { std::fprintf(stderr, "%d test(s) failed\n", g_fail); return 1; }
    std::printf("all subtitle_export tests passed\n");
    return 0;
}
