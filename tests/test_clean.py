"""Unit tests cho agent.data.clean — pure functions, không cần mạng/pandas."""

from agent.data.clean import clean_rows, pick_text


class TestPickText:
    def test_joins_in_prefer_order(self):
        row = {"input": "đau bụng", "instruction": "khác"}
        assert pick_text(row, ["input", "instruction"]) == "đau bụng\nkhác"
        assert pick_text(row, ["instruction", "input"]) == "khác\nđau bụng"

    def test_fallback_when_preferred_empty(self):
        row = {"input": "", "instruction": "câu hỏi"}
        assert pick_text(row, ["input", "instruction"]) == "câu hỏi"

    def test_drop_substring_duplicate(self):
        # instruction đóng khung, input đã nằm trong instruction -> giữ bản đầy đủ
        row = {
            "input": "đau đầu 3 ngày",
            "instruction": "Bạn là bác sĩ. Bệnh nhân: đau đầu 3 ngày. Hãy tư vấn.",
        }
        assert pick_text(row, ["input", "instruction"]) == row["instruction"]

    def test_join_distinct_parts(self):
        row = {"input": "đau bụng âm ỉ", "instruction": "Bạn là bác sĩ, hãy tư vấn."}
        out = pick_text(row, ["input", "instruction"])
        assert "đau bụng âm ỉ" in out and "hãy tư vấn" in out

    def test_all_empty(self):
        assert pick_text({}, ["input", "instruction"]) == ""


class TestCleanRows:
    def _rows(self):
        good = {
            "question": "bị sốt 3 ngày nên làm gì?",
            "answer": "bạn nên đi khám bác sĩ để được kiểm tra kỹ.",
        }
        long_q = {
            "question": "Đau đầu thường xuyên có sao không?",
            "answer": "Nếu đau đầu kéo dài bạn nên đi khám để loại trừ nguyên nhân nặng.",
        }
        return [
            good,
            dict(good),  # dup
            {"question": "", "answer": "thiếu câu hỏi"},  # empty
            {"question": "ngắn", "answer": "cũng ngắn"},  # too short
            long_q,
        ]

    def test_keeps_valid_drops_bad(self):
        kept, stats = clean_rows(self._rows())
        assert stats.input_rows == 5
        assert stats.kept == 2
        assert stats.dropped_duplicate == 1
        assert stats.dropped_empty == 1
        assert stats.dropped_short == 1
        assert len(kept) == 2

    def test_strips_whitespace(self):
        rows = [{"question": "  đau bụng nên ăn gì?  ", "answer": "  nên ăn đồ dễ tiêu.  "}]
        kept, _ = clean_rows(rows, min_question_chars=5, min_answer_chars=5)
        assert kept[0]["question"] == "đau bụng nên ăn gì?"
        assert kept[0]["answer"] == "nên ăn đồ dễ tiêu."

    def test_dedupe_normalizes_case_and_spacing(self):
        rows = [
            {"question": "Đau  đầu nên làm gì?", "answer": "Nên đi khám."},
            {"question": "đau đầu nên làm gì?", "answer": "nên đi khám."},  # same sau normalize
        ]
        kept, stats = clean_rows(rows, min_question_chars=5, min_answer_chars=5)
        assert stats.kept == 1
        assert stats.dropped_duplicate == 1

    def test_stats_dict(self):
        _, stats = clean_rows(self._rows())
        d = stats.as_dict()
        assert d["input_rows"] == 5 and d["kept"] == 2
