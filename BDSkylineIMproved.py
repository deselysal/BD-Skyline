import numpy as np
from treesimulator.mtbd_models import Model

INFECTED = 'Infected'


class BirthDeathSkylineModel(Model):
    def __init__(self, params, *args, **kwargs):
        if params.shape[0] != 4:
            raise ValueError("The parameter matrix must have exactly 4 rows (la, psi, p, t).")
        
        self.ModelsList = []
        self.current_model_name = None  # Stores the name of the active model
        
        for i in range(params.shape[1]):
            model_name = f'BD{i+1}'
            model_params = {
                'Model': model_name,
                'la': params[0, i],
                'psi': params[1, i],
                'p': params[2, i],
                't': params[3, i]
            }
            self.ModelsList.append(model_params)
        
        # Initialize states correctly using super().__init__()
        super().__init__(states=[INFECTED])

        self.la = None
        self.psi = None

    def select_model(self, time):
        """Select the appropriate model based on the given time."""
        for i, model_params in enumerate(self.ModelsList):
            # Define the start and end of the interval
            start_t = 0 if i == 0 else self.ModelsList[i - 1]["t"]
            end_t = model_params["t"]

            # Check if time lies within the current interval
            if start_t <= time < end_t:
                # Configure and return the corresponding model
                if self.current_model_name != model_params['Model']:
                    print(f"Switching to model {model_params['Model']} for time {time}")
                return self._configure_model_with_parameters(model_params)

        # If time is beyond the last interval, return the final model
        last_model_params = self.ModelsList[-1]
        if self.current_model_name != last_model_params['Model']:
            print(f"Switching to the final model {last_model_params['Model']} for time {time}")
        return self._configure_model_with_parameters(last_model_params)

    def _configure_model_with_parameters(self, model_params):
        """Configure the BD-Skyline model with the selected time-dependent parameters."""
        la = model_params['la']
        psi = model_params['psi']
        p = model_params['p']

        # Setup transmission and removal rates
        las = np.array([[la]], dtype=np.float64)

        # Call parent Model constructor
        super().__init__(states=[INFECTED], transmission_rates=las, removal_rates=[psi], ps=[p])

        # Store parameters
        self.la = la
        self.psi = psi
        self.current_model_name = model_params['Model']

        return self

    def get_name(self):
        """Return the name of the current BD-Skyline model."""
        return self.current_model_name if self.current_model_name else 'BD'

    def get_epidemiological_parameters(self):
        """Convert rate parameters into epidemiological measures."""
        if self.n_recipients is None:
            self.n_recipients = np.ones(len(self.states))
        
        r0 = self.transmission_rates[0, 0] / self.removal_rates[0] * self.n_recipients[0]
        result = {
            'R0': r0,
            'infectious time': 1 / self.removal_rates[0],
            'sampling probability': self.ps[0],
            'transmission rate': self.transmission_rates[0, 0],
            'removal rate': self.removal_rates[0]
        }
        if self.n_recipients[0] > 1:
            result['avg recipient number per transmission'] = self.n_recipients[0]
        return result


# Example usage
#   [0.1, 0.2, 0.3],  # psis
   # [0.5, 0.6, 0.7],  # ps
    #[2.0, 5.0, 10.0]  # ts
#])

#skyline_model = BirthDeathSkylineModel(params)

#time = 11.0
#actual_model = skyline_model.select_model(time)

#if actual_model:
    #print(actual_model.get_name())  # Prints the specific model name, like BD1, BD2, etc.
    #print(actual_model.get_epidemiological_parameters())  # Outputs epidemiological parameters
#else:
    #print("No model available for this time.")
