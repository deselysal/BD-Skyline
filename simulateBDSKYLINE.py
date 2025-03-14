import logging
import numpy as np
from treesimulator import save_forest, save_log, save_ltt
from treesimulator.generator import generate, observed_ltt
from treesimulator.mtbd_models import CTModel
from BDSkylineIMproved import BirthDeathSkylineModel


def main():
    """
    Entry point for tree/forest generation with the BD-Skyline model with command-line arguments.
    :return: void
    """
    import argparse

    parser = argparse.ArgumentParser(description="Simulates a tree (or a forest of trees) with the BD-Skyline model.")
    
    parser.add_argument('--min_tips', default=5, type=int, help="Minimum number of simulated leaves.")
    parser.add_argument('--max_tips', default=20, type=int, help="Maximum number of simulated leaves.")
    parser.add_argument('--T', required=False, default=np.inf, type=float, help="Total simulation time.")
    parser.add_argument('--la', default=[0.4,0.5,0.6], nargs='+', type=float, help="List of transmission rates for each interval.")
    parser.add_argument('--psi', default=[0.1,0.2,0.3], nargs='+', type=float, help="List of removal rates for each interval.")
    parser.add_argument('--p', default=[0.5,0.6,0.7], nargs='+', type=float, help="List of sampling probabilities for each interval.")
    parser.add_argument('--t', default=[2.0,5.0,10.0], nargs='+', type=float, help="List of time points corresponding to parameters change.")
    parser.add_argument('--upsilon', required=False, default=0, type=float, help='Notification probability')
    parser.add_argument('--max_notified_contacts', required=False, default=1, type=int, help='Maximum notified contacts')
    parser.add_argument('--log', default='output.log', type=str, help="Output log file")
    parser.add_argument('--nwk', default='output.nwk', type=str, help="Output tree or forest file")
    parser.add_argument('--ltt', required=False, default=None, type=str, help="Output LTT file")
    parser.add_argument('-v', '--verbose', default=True, action='store_true', help="Verbose output")
    
    params = parser.parse_args()
    logging.getLogger().handlers = []
    logging.basicConfig(level=logging.DEBUG if params.verbose else logging.INFO, 
                        format='%(asctime)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    # Convert command-line parameters into a matrix
    parameters_matrix = np.array([params.la, params.psi, params.p, params.t])
    
    if parameters_matrix.shape[1] != len(params.t):
        raise ValueError("Mismatch between number of time points and parameter sets!")

    # Log the configuration
    logging.info('BD-Skyline model parameters are configured.')

    # Create a BirthDeathSkylineModel
    model = BirthDeathSkylineModel(parameters_matrix)

    if params.upsilon and params.upsilon > 0:
        logging.info('CT parameters are:\n\tupsilon={}'.format(params.upsilon))
        model = CTModel(model=model, upsilon=params.upsilon)

    if params.T < np.inf:
        logging.info('Total time T={}'.format(params.T))

    # Generate forest using the skyline model
    try:
        forest, (total_tips, u, T), ltt = generate(model, params.min_tips, params.max_tips, T=params.T,
                                                   max_notified_contacts=params.max_notified_contacts)

        # Save outputs
        save_forest(forest, params.nwk)
        save_log(model, total_tips, T, u, params.log)
        if params.ltt:
            save_ltt(ltt, observed_ltt(forest, T), params.ltt)
    except RuntimeWarning as e:
        logging.error("Encountered a runtime warning during simulation: {}".format(e))
    except ValueError as e:
        logging.error("Simulation error: {}".format(e))

if '__main__' == __name__:
    main()