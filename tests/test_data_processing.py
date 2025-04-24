import pytest
from src.data_utils.funsd_processor import normalize_bbox, align_labels_with_tokens


def test_normalize_bbox():
    bbox = [50, 100, 150, 200]
    norm = normalize_bbox(bbox, width=200, height=400)
    assert norm == [250, 250, 750, 500]


def test_align_labels_with_tokens():
    labels = [0, 1, 2]
    word_ids = [0, 0, 1, 2, None]
    aligned = align_labels_with_tokens(labels, word_ids)
    assert aligned == [0, -100, 1, 2, -100]