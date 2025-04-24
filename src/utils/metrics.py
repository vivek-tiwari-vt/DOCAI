def compute_token_classification_metrics(preds, labels):
    """
    preds: flat list of predicted labels
    labels: flat list of true labels
    Returns a dict with 'f1' score and 'report' classification report text.
    """
    import numpy as np
    from sklearn.metrics import f1_score as sklearn_f1_score
    from sklearn.metrics import classification_report as sklearn_report
    
    # Convert to numpy arrays if they aren't already
    preds_array = np.array(preds) if not isinstance(preds, np.ndarray) else preds
    labels_array = np.array(labels) if not isinstance(labels, np.ndarray) else labels
    
    # Filter out ignored tokens (typically -100 in HuggingFace datasets)
    # This is crucial to avoid the "mix of multiclass-multioutput and multilabel-indicator targets" error
    valid_indices = labels_array != -100
    filtered_preds = preds_array[valid_indices]
    filtered_labels = labels_array[valid_indices]
    
    # Use sklearn metrics with the filtered data
    f1 = sklearn_f1_score(filtered_labels, filtered_preds, average='micro')
    report = sklearn_report(filtered_labels, filtered_preds)
    
    return {
        'f1': f1,
        'report': report
    }
