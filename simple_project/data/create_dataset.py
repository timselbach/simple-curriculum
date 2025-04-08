import sys
sys.path.append(".")
import argparse
from simple_project.data.all_datasets import SimpleWikiDataset, OSEDataset

# parser = argparse.ArgumentParser()
# parser.add_argument('dataset', type=str, help="use either SimpleWiki or OSE to create the respective dataset")
# parser.add_argument('debugging', type=bool, help="create mini dataset for debugging")


def main(name, debugging=False):
    if name == "OSE":
        dataset = OSEDataset()
    elif name == "SimpleWiki":
        dataset = SimpleWikiDataset(debugging=debugging)


if __name__ == "__main__":
    # args = parser.parse_args(["SimpleWiki", ""])
    # main(args.dataset)

    main("SimpleWiki", True)

