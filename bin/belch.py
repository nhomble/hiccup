import argparse
from tkinter import Tk

import hiccup.run as run
from hiccup.ui.app import BelchUI

"""
Provide a simple GUI and command line tool to perform compressing and decompressing of images
"""


def is_gui(args):
    return args.compress is None and args.decompress is None


def run_gui():
    root = Tk()
    _ = BelchUI(root)
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="hombln's image compression")
    parser.add_argument('--compress', '-c', metavar='IMG_PATH', nargs=1,
                        help='compress the image via cmd line with a certain style')
    parser.add_argument('--decompress', '-d', metavar='HIC', nargs=1, help='decompress the hic image via cmd line')
    parser.add_argument('--style', '-s', metavar='STYLE', nargs=1, choices=[
        'FOURIER', 'WAVELET'
    ], help='dictate which compression algorithm we use')
    parser.add_argument('--output', '-o', metavar='OUT', nargs=1, help='output path', default='.')
    args = parser.parse_args()
    if is_gui(args):
        run_gui()
    elif args.compress is not None:
        run.compress(args.compress, args.output, args.style)
    elif args.decompress is not None:
        run.decompress(args.decompress, args.output)
    else:
        raise RuntimeError("Illegal state")


if __name__ == "__main__":
    main()
