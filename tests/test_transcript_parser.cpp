// Ports moss_transcribe_diarize/tests/test_transcript_parser.py 1:1,
// plus a JFK baseline.text parse case.
#include "transcript_parser.hpp"

#include <cstdio>
#include <string>
#include <vector>

using mt::TranscriptSegment;

static int g_fail = 0;

static void check_segments(const char* name,
                           const std::vector<TranscriptSegment>& got,
                           const std::vector<TranscriptSegment>& want) {
    bool ok = got.size() == want.size();
    if (ok) {
        for (size_t i = 0; i < got.size(); ++i) {
            if (got[i] != want[i]) { ok = false; break; }
        }
    }
    if (!ok) {
        ++g_fail;
        std::fprintf(stderr, "FAIL %s: got %zu segments, want %zu\n",
                     name, got.size(), want.size());
        for (size_t i = 0; i < got.size(); ++i) {
            std::fprintf(stderr, "  got[%zu] = {%.6g, %.6g, %s, \"%s\"}\n", i,
                         got[i].start, got[i].end, got[i].speaker.c_str(),
                         got[i].text.c_str());
        }
    } else {
        std::printf("ok %s\n", name);
    }
}

int main() {
    // test_parse_compact_transcript
    check_segments("parse_compact_transcript",
        mt::parse_transcript("[0.48][S01]Welcome[1.66][12.26][S02]Ready[13.81]"),
        {{0.48, 1.66, "S01", "Welcome"}, {12.26, 13.81, "S02", "Ready"}});

    // test_streaming_with_single_character_chunks
    {
        std::string text = "[0.48][S01]\xE4\xBD\xA0\xE5\xA5\xBD[1.66]"
                           "[12.26][S02]\xE5\x8F\xAF\xE4\xBB\xA5\xE5\xBC\x80\xE5\xA7\x8B[13.81]";
        mt::TranscriptStreamParser parser;
        std::vector<TranscriptSegment> segs;
        for (char ch : text) {
            auto s = parser.feed(std::string(1, ch));
            segs.insert(segs.end(), s.begin(), s.end());
        }
        auto tail = parser.close();
        segs.insert(segs.end(), tail.begin(), tail.end());
        check_segments("streaming_single_char_chunks", segs,
            {{0.48, 1.66, "S01", "\xE4\xBD\xA0\xE5\xA5\xBD"},
             {12.26, 13.81, "S02", "\xE5\x8F\xAF\xE4\xBB\xA5\xE5\xBC\x80\xE5\xA7\x8B"}});
    }

    // test_iter_transcript_segments_accepts_arbitrary_chunks
    {
        std::vector<std::string> chunks = {"[0", ".48][", "S01]", "hello[",
                                           "1.66]", "[2.0][S02]", "bye", "[3.0]"};
        mt::TranscriptStreamParser parser;
        std::vector<TranscriptSegment> segs;
        for (const auto& c : chunks) {
            auto s = parser.feed(c);
            segs.insert(segs.end(), s.begin(), s.end());
        }
        auto tail = parser.close();
        segs.insert(segs.end(), tail.begin(), tail.end());
        check_segments("iter_arbitrary_chunks", segs,
            {{0.48, 1.66, "S01", "hello"}, {2.0, 3.0, "S02", "bye"}});
    }

    // test_numeric_brackets_inside_text_are_preserved
    check_segments("numeric_brackets_preserved",
        mt::parse_transcript(
            "[0][S01]\xE7\xAC\xAC[2024]\xE5\xB9\xB4\xEF\xBC\x8C\xE7\xBC\x96\xE5\x8F\xB7[001]\xE7\xBB\xA7\xE7\xBB\xAD[4]"),
        {{0.0, 4.0, "S01",
          "\xE7\xAC\xAC[2024]\xE5\xB9\xB4\xEF\xBC\x8C\xE7\xBC\x96\xE5\x8F\xB7[001]\xE7\xBB\xA7\xE7\xBB\xAD"}});

    // test_noise_before_first_segment_is_ignored
    check_segments("noise_before_first_segment",
        mt::parse_transcript("noise [bad][0.1][S01]hello[0.9]"),
        {{0.1, 0.9, "S01", "hello"}});

    // test_whitespace_between_segments_is_ignored
    check_segments("whitespace_between_segments",
        mt::parse_transcript("[0][S01]a[1]\n [2][S02]b[3]"),
        {{0.0, 1.0, "S01", "a"}, {2.0, 3.0, "S02", "b"}});

    // JFK baseline.text
    check_segments("jfk_baseline_text",
        mt::parse_transcript(
            "[0.28][S01] And so, my fellow Americans,[2.32][3.22][S01] ask not...[10.59]"),
        {{0.28, 2.32, "S01", "And so, my fellow Americans,"},
         {3.22, 10.59, "S01", "ask not..."}});

    if (g_fail) { std::fprintf(stderr, "%d test(s) failed\n", g_fail); return 1; }
    std::printf("all transcript_parser tests passed\n");
    return 0;
}
