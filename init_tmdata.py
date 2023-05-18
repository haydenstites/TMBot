import argparse
from tmbot.core import init_tmdata

if __name__ == "__main__":
    description = "Verifies installation of TMData"
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("-o", "--OpenPlanet", nargs="?", type=str, default=None, help="OpenPlanet install location")

    args = parser.parse_args()

    init_tmdata(op_path=args.OpenPlanet)