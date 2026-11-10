"""Unit tests cho agent.data.chunk — pure functions, không cần ML deps."""

from agent.data.chunk import chunk_row, chunk_text, split_sentences


class TestSplitSentences:
    def test_splits_on_punctuation(self):
        text = "Bạn bị sốt 3 ngày. Nên đi khám nhé! Khó thở thì cấp cứu ngay."
        assert split_sentences(text) == [
            "Bạn bị sốt 3 ngày.",
            "Nên đi khám nhé!",
            "Khó thở thì cấp cứu ngay.",
        ]

    def test_empty(self):
        assert split_sentences("   ") == []


class TestChunkText:
    def test_short_text_single_chunk(self):
        assert chunk_text("Một câu duy nhất.") == ["Một câu duy nhất."]

    def test_respects_max_chars(self):
        text = " ".join(f"Câu số {i} dài vừa đủ." for i in range(20))
        chunks = chunk_text(text, max_chars=80)
        assert len(chunks) > 1
        assert all(len(c) <= 80 for c in chunks)

    def test_overlap_shares_sentences_between_adjacent_chunks(self):
        text = "Câu một cụt. Câu hai cụt. Câu ba cụt. Câu bốn cụt."
        chunks = chunk_text(text, max_chars=26, overlap_sentences=1)
        assert len(chunks) == 3
        assert "Câu hai cụt." in chunks[0] and "Câu hai cụt." in chunks[1]
        assert "Câu ba cụt." in chunks[1] and "Câu ba cụt." in chunks[2]

    def test_overlap_dropped_when_it_exceeds_max(self):
        # 2 câu không vừa max_chars -> không thể overlap, mỗi chunk 1 câu
        text = "Câu một cụt. Câu hai cụt. Câu ba cụt."
        chunks = chunk_text(text, max_chars=20, overlap_sentences=1)
        assert chunks == ["Câu một cụt.", "Câu hai cụt.", "Câu ba cụt."]

    def test_hard_splits_long_sentence(self):
        chunks = chunk_text("x" * 250, max_chars=100, overlap_sentences=0)
        assert len(chunks) == 3
        assert all(len(c) <= 100 for c in chunks)


class TestChunkRow:
    def test_includes_question_prefix_and_ids(self):
        row = {
            "id": "row-1",
            "question": "Bé bị ho?",
            "answer": "Bé bị ho có thể do cảm lạnh. Nên theo dõi thêm.",
            "language": "vi",
        }
        chunks = chunk_row(row, max_chars=40)
        assert chunks
        assert chunks[0].text.startswith("Hỏi: Bé bị ho?\nĐáp: ")
        assert all(c.source_row_id == "row-1" for c in chunks)
        assert all(c.question == "Bé bị ho?" for c in chunks)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_no_question_prefix_when_disabled(self):
        row = {"id": "r", "question": "X?", "answer": "Đáp án ở đây."}
        chunks = chunk_row(row, include_question=False)
        assert chunks[0].text == "Đáp án ở đây."

    def test_empty_answer_no_chunks(self):
        assert chunk_row({"id": "r", "question": "X?", "answer": ""}) == []
