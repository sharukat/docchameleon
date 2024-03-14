def split(list_a, chunk_size):
    """
    The split function takes a list and splits it into chunks of the specified size.

    Args:
        list_a: Specify the list that will be split into chunks
        chunk_size: Specify the size of each chunk

    Returns:
        A generator object
    """
    for i in range(0, len(list_a), chunk_size):
        yield list_a[i:i + chunk_size]