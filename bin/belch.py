import argparse

"""
Provide a simple GUI and command line tool to perform compressing and decompressing of images
"""


def isGuiMode(args):
    return args.compress is None and args.decompress is None


def launchGui():
    pass


def main():
    parser = argparse.ArgumentParser(description="hombln's image compression")
    parser.add_argument('--compress', '-c', metavar='IMG', nargs=1, help='compress the image via cmd line')
    parser.add_argument('--decompress', '-d', metavar='HIC', nargs=1, help='decompress the image via cmd line')
    args = parser.parse_args()
    if isGuiMode(args):
        launchGui()
    elif args.compress is not None:
        pass
    elif args.decompress is not None:
        pass
    else:
        raise RuntimeError("Illegal state")


if __name__ == "__main__":
    main()
