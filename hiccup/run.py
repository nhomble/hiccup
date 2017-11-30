import hiccup.iohelper as iohelper

"""
Entry functions for belch
"""


def compress(path, output, style):
    if style is None:
        print("Style cannot be none")
        return
    img = iohelper.open_raw_img(path)
    #transform = make_transform(style)
    #compressed = transform.compress(img)
    ## TODO
    # make huffman coding (and tree?)
    # create HIC image
    # write to binary


def decompress(path, output):
    # read from binary
    # construct HIC image
    # decode huffman
    # reverse transform
    # write image
    pass
