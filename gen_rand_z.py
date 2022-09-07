import click
import numpy as np

@click.command()
@click.option('-d', '--dim', type=int, default=512)
@click.option('-n', '--num', type=int, default=100)
@click.option('--save_path', type=str, default='data/random_z.npy')

def _main(dim, num, save_path):
    seed = np.random.randn(num, dim)
    print(seed.shape)
    np.save(save_path, seed)

if __name__ == "__main__":
    _main()