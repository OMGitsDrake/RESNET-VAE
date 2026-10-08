import argparse
from math import floor

def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description='Calcolo output size dopo convoluzione')
    parser.add_argument('in_s')
    parser.add_argument('p', default=0)
    parser.add_argument('k', default=3)
    parser.add_argument('s', default=1)
    args = parser.parse_args(argv)

    out = floor(
        (int(args.in_s) + 2*int(args.p) - int(args.k)) / int(args.s)
    ) + 1

    return out

if __name__ == '__main__':
    print(main())