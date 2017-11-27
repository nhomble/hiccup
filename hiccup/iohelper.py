import rawpy


def open_raw_img(path):
    """
    open image from path
    """

    # opencv cannot handle raw images
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess()
    return rgb
