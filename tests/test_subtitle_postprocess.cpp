// Ports moss_transcribe_diarize/tests/test_subtitle_postprocess.py 1:1.
#include "subtitle.hpp"

#include <cstdio>
#include <string>
#include <vector>

using mt::SubtitleSegment;

static int g_fail = 0;

static void expect(bool cond, const char* name) {
    if (!cond) { ++g_fail; std::fprintf(stderr, "FAIL %s\n", name); }
    else std::printf("ok %s\n", name);
}

int main() {
    // test_builds_subtitle_segments_from_transcript
    {
        auto segs = mt::subtitle_segments_from_transcript(
            "[0][S01]\xE4\xBD\xA0\xE5\xA5\xBD[1.5][2][S02]\xE5\xBC\x80\xE5\xA7\x8B[3.5]");
        bool ok = segs.size() == 2 &&
                  segs[0].speaker == "S01" && segs[1].speaker == "S02" &&
                  segs[0].text == "\xE4\xBD\xA0\xE5\xA5\xBD" &&
                  segs[1].text == "\xE5\xBC\x80\xE5\xA7\x8B" &&
                  segs[0].id == "seg_0001";
        expect(ok, "builds_subtitle_segments_from_transcript");
    }

    // test_can_build_raw_subtitle_segments_without_postprocess
    {
        auto segs = mt::subtitle_segments_from_transcript(
            "[0][S01]\xE7\x9F\xAD[0.4][0.2][S01]\xE9\x87\x8D\xE5\x8F\xA0\xE4\xBD\x86\xE4\xBF\x9D\xE7\x95\x99[0.8]",
            /*postprocess=*/false);
        bool ok = segs.size() == 2 &&
                  segs[0].start == 0.0 && segs[0].end == 0.4 &&
                  segs[0].text == "\xE7\x9F\xAD" &&
                  segs[1].start == 0.2 && segs[1].end == 0.8 &&
                  segs[1].text == "\xE9\x87\x8D\xE5\x8F\xA0\xE4\xBD\x86\xE4\xBF\x9D\xE7\x95\x99";
        expect(ok, "raw_subtitle_segments_without_postprocess");
    }

    // test_coerce_payload_does_not_reorder_or_fix_times
    {
        std::vector<SubtitleSegment> in = {
            {"b", 3, 2, "S02", ""},
            {"a", 0.2, 0.1, "S01", "x"},
        };
        auto segs = mt::coerce_subtitle_segments(in);
        bool ok = segs.size() == 2 &&
                  segs[0].id == "b" && segs[1].id == "a" &&
                  segs[0].start == 3.0 && segs[0].end == 2.0 &&
                  segs[1].start == 0.2 && segs[1].end == 0.1 &&
                  segs[0].text.empty();
        expect(ok, "coerce_payload_does_not_reorder_or_fix_times");
    }

    // test_merges_adjacent_same_speaker_short_gap
    {
        std::vector<SubtitleSegment> in = {
            {"a", 0, 1.2, "S01", "\xE4\xBD\xA0\xE5\xA5\xBD"},
            {"b", 1.3, 2.4, "S01", "\xE4\xB8\x96\xE7\x95\x8C"},
        };
        auto segs = mt::normalize_segments(in, mt::DEFAULT_MIN_DURATION,
            mt::DEFAULT_MAX_DURATION, /*max_chars=*/24, /*merge_gap=*/0.3);
        bool ok = segs.size() == 1 &&
                  segs[0].text == "\xE4\xBD\xA0\xE5\xA5\xBD\xE4\xB8\x96\xE7\x95\x8C" &&
                  segs[0].end == 2.4;
        expect(ok, "merges_adjacent_same_speaker_short_gap");
    }

    // test_fixes_overlaps_and_min_duration
    {
        std::vector<SubtitleSegment> in = {
            {"a", 0, 0.4, "S01", "a"},
            {"b", 0.2, 0.6, "S02", "b"},
        };
        auto segs = mt::normalize_segments(in, /*min_duration=*/1.0,
            mt::DEFAULT_MAX_DURATION, mt::DEFAULT_MAX_CHARS, /*merge_gap=*/0);
        bool ok = segs.size() == 2 &&
                  segs[0].start == 0.0 && segs[0].end == 1.0 &&
                  segs[1].start == 1.0 && segs[1].end == 2.0;
        expect(ok, "fixes_overlaps_and_min_duration");
    }

    // test_splits_long_segments
    {
        std::vector<SubtitleSegment> in = {
            {"a", 0, 12, "S01",
             "\xE7\xAC\xAC\xE4\xB8\x80\xE5\x8F\xA5\xE5\xBE\x88\xE9\x95\xBF\xEF\xBC\x8C"
             "\xE9\x9C\x80\xE8\xA6\x81\xE5\x88\x87\xE5\xBC\x80\xE3\x80\x82"
             "\xE7\xAC\xAC\xE4\xBA\x8C\xE5\x8F\xA5\xE4\xB9\x9F\xE5\xBE\x88\xE9\x95\xBF\xEF\xBC\x8C"
             "\xE9\x9C\x80\xE8\xA6\x81\xE7\xBB\xA7\xE7\xBB\xAD\xE5\x88\x87\xE5\xBC\x80\xE3\x80\x82"},
        };
        auto segs = mt::normalize_segments(in, mt::DEFAULT_MIN_DURATION,
            /*max_duration=*/6.0, /*max_chars=*/10, /*merge_gap=*/0);
        bool all_pos = true;
        for (auto& s : segs) if (!(s.end > s.start)) all_pos = false;
        bool ok = segs.size() > 1 && all_pos && segs[0].start == 0.0;
        expect(ok, "splits_long_segments");
    }

    if (g_fail) { std::fprintf(stderr, "%d test(s) failed\n", g_fail); return 1; }
    std::printf("all subtitle_postprocess tests passed\n");
    return 0;
}
