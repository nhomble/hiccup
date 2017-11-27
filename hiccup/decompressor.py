import hiccup.hicimage as hic


class Decompressor:
    @staticmethod
    def load(img_path: str):
        # read binary
        return Decompressor(None)

    def __init__(self, hic: hic.HicImage):
        self._hic = hic

    def explode(self, output_path):
        pass
