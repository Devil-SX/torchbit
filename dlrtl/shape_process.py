def get_padlen(dim_size, align):
    return (align - dim_size % align) % align

def pad(dim_size, align):
    return dim_size + get_padlen(dim_size, align)
