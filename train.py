import argparse
import json

from attrdict import AttrDict
from models import DDPM, UNet
from dataset import AnimeFacesDataset

def train(c: AttrDict):
    if c.model == 'unet':
        model = UNet(c)
        print('Training UNet...')
    else:
        print('Model not implemented...')

    diffusion = DDPM(model, c)
    trainset = AnimeFacesDataset(c.batch_size, c.num_workers, c.data_dir)
    diffusion.fit(trainset=trainset)

    print('Training Complete..')


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to configuration file')
    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    config = AttrDict(json.loads(data))
    train(config)

    print('Training Complete..')


if __name__ == '__main__':
    main()
