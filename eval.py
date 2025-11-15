
def recall(retrieved_indices: list, relevant_indices: list) -> float:
    """
    Recall calculation.
    Args:
        retrieved_indices: List of indices retrieved by the system.
        relevant_indices: List of ground truth relevant indices.
    Returns:
        Recall value.
    """
    retrieved = set(retrieved_indices)
    relevant = set(relevant_indices)
    return len(retrieved & relevant) / len(relevant)


def precision(retrieved_indices: list, relevant_indices: list) -> float:
    """
    Precision calculation.
    Args:
        retrieved_indices: List of indices retrieved by the system.
        relevant_indices: List of ground truth relevant indices.
    Returns:
        Precision value.
    """
    retrieved = set(retrieved_indices)
    relevant = set(relevant_indices)
    return len(retrieved & relevant) / len(retrieved)


def average_precision(retrieved_indices: list, relevant_indices: list) -> float:
    """
    Average Precision (AP) calculation.
    Args:
        retrieved_indices: List of indices retrieved by the system.
        relevant_indices: List of ground truth relevant indices.
    Returns:
        AP value.
    """
    ap = 0.0
    relevant_num = 0.
    for rank, retrieved_idx in enumerate(retrieved_indices, start=1):
        if retrieved_idx in relevant_indices:
            relevant_num += 1
            ap += relevant_num / rank
    
    if relevant_num == 0.:
        return 0.0

    ap /= len(relevant_indices)
    return ap


def reciprocal_rank(retrieved_indices: list, relevant_indices: list) -> float:
    """
    Reciprocal Rank (RR) calculation.
    Args:
        retrieved_indices: List of indices retrieved by the system.
        relevant_indices: List of ground truth relevant indices.
    Returns:
        RR value.
    """
    for rank, idx in enumerate(retrieved_indices, start=1):
        if idx in relevant_indices:
            return 1.0 / rank
    return 0.0


if __name__ == "__main__":
    pass
    # TODO: Generate ground truth and evaluate retrieval performance
