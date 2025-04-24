import numpy as np
import torch
from transformers import LayoutLMTokenizerFast


def load_raw_funsd(data_dir):
    from datasets import load_from_disk
    return load_from_disk(data_dir)


def normalize_bbox(bbox, width, height):
    x0, y0, x1, y1 = bbox
    return [
        int(1000 * x0 / width),
        int(1000 * y0 / height),
        int(1000 * x1 / width),
        int(1000 * y1 / height),
    ]


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id is None:
            new_labels.append(-100)
        elif word_id != current_word:
            new_labels.append(labels[word_id])
            current_word = word_id
        else:
            new_labels.append(-100)
    return new_labels


def preprocess_funsd_for_layoutlm(examples, tokenizer: LayoutLMTokenizerFast, max_seq_length=512):
    # Ensure all lists are batch-aligned
    words_list = examples['words']
    bboxes_list = examples['bboxes']
    labels_list = examples['ner_tags']
    if not isinstance(words_list[0], list):
        words_list = [words_list]
        bboxes_list = [bboxes_list]
        labels_list = [labels_list]

    processed = {'input_ids': [], 'bbox': [], 'attention_mask': [], 'token_type_ids': [], 'labels': []}
    for words, bboxes, labels in zip(words_list, bboxes_list, labels_list):
        # Normalize bounding boxes
        normalized_boxes = [normalize_bbox(b, width=1000, height=1000) for b in bboxes]
        tokenized = tokenizer(
            words,
            padding='max_length',
            truncation=True,
            max_length=max_seq_length,
            return_offsets_mapping=True,
            return_length=True,
            return_tensors='pt'
        )
        max_len = tokenized['input_ids'].shape[1]
        # Pad or truncate boxes
        if len(normalized_boxes) < max_len:
            padded_boxes = normalized_boxes + [[0, 0, 0, 0]] * (max_len - len(normalized_boxes))
        else:
            padded_boxes = normalized_boxes[:max_len]
        boxes_tensor = torch.tensor([padded_boxes])
        tokenized['bbox'] = boxes_tensor
        # Align labels
        word_ids = tokenized.word_ids(batch_index=0)
        aligned_labels = [-100] * len(word_ids)
        previous_word_idx = None
        for i, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            elif word_idx != previous_word_idx:
                if word_idx < len(labels):
                    aligned_labels[i] = labels[word_idx]
                else:
                    aligned_labels[i] = -100
            previous_word_idx = word_idx
        processed['input_ids'].append(tokenized['input_ids'][0])
        processed['bbox'].append(tokenized['bbox'][0])
        processed['attention_mask'].append(tokenized['attention_mask'][0])
        processed['token_type_ids'].append(tokenized['token_type_ids'][0])
        processed['labels'].append(aligned_labels)
    return processed

