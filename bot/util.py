def split_text_into_chunks(text, chunk_size):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
