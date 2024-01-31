import argparse
from Engine import Engine
from importlib import import_module

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config file')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    conf = import_module(f"configs.{args.config}").config
    conf['config'] = args.config
    engine = Engine(conf)
    engine.train()