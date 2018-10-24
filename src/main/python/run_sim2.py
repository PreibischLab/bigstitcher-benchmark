import argparse
import gputools
from sim_util import sim_from_definition


def main():

    parser = argparse.ArgumentParser(description='Run biobeam simulation')
    parser.add_argument('params_file', type=str,
                        help='JSON file containing the parameters of the simulation')
    parser.add_argument('--gpu_idx', dest='gpu_idx', default=0,
                        help='Index of the GPU to use for simulation')

    args = parser.parse_args()

    # set and check which gpu is used
    gputools.init_device(id_platform=0, id_device=int(args.gpu_idx))
    gputools.get_device()

    # run simulation
    sim_from_definition(args.params_file)


if __name__ == '__main__':
    main()
