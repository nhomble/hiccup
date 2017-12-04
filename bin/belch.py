import argparse
from tkinter import Tk

import hiccup.run as run
import hiccup.model as model
import hiccup.settings as settings

from hiccup.app import BelchUI

"""
Provide a simple GUI and command line tool to perform compressing and decompressing of images
"""


def is_gui(args):
    return args.compress is None and args.decompress is None


def run_gui():
    settings.DEBUG = True # who cares, you're staring at that ugly tkinter window anyway
    root = Tk()
    _ = BelchUI(root)
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="hombln's image compression")
    parser.add_argument('--compress', '-c', metavar='IMG_PATH',
                        help='compress the image via cmd line with a certain style')
    parser.add_argument('--decompress', '-d', metavar='HIC', help='decompress the hic image via cmd line')
    parser.add_argument('--compression', '-s', metavar='STYLE',
                        choices=[model.Compression.HIC.value, model.Compression.JPEG.value],
                        help='dictate which compression algorithm we use', default=model.Compression.HIC.value)
    parser.add_argument('--output', '-o', metavar='OUT', help='output path', default='.')
    parser.add_argument('--verbose', '-v', help='control the debug flag', default=False, action='store_true')
    args = parser.parse_args()

    if not args.verbose:
        print("=== Suppressing debug messages ==")
        settings.DEBUG = args.verbose

    if is_gui(args):
        run_gui()
    elif args.compress is not None:
        run.compress(args.compress, args.output, model.Compression(args.compression))
    elif args.decompress is not None:
        run.decompress(args.decompress)
    else:
        raise RuntimeError("Illegal state")


if __name__ == "__main__":
    main()
