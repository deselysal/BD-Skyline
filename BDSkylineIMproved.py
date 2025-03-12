import numpy as np

INFECTED = 'Infected'

class Model:
    def __init__(self, states=None, transmission_rates=None, removal_rates=None, ps=None, *args, **kwargs):
        self.states = np.array(states)
        self.transmission_rates = np.array(transmission_rates) if transmission_rates is not None else np.zeros((1, 1))
        self.removal_rates = np.array(removal_rates) if removal_rates is not None else np.zeros(1)
        self.ps = np.array(ps) if ps is not None else np.zeros(1)
        self.n_recipients = None

class BirthDeathSkylineModel(Model):
    def __init__(self, params, *args, **kwargs):
        if params.shape[0] != 4:
            raise ValueError("La matriz de parámetros debe tener exactamente 4 filas (la, psi, p, t).")
        
        self.ModelsList = []
        self.current_model_name = None  # Variable para almacenar el nombre del modelo actual
        
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
    
    def select_model(self, time):
        for model_params in self.ModelsList:
            if time <= model_params['t']:
                return self._configure_model_with_parameters(model_params)
        return None

    def _configure_model_with_parameters(self, model_params):
        la = model_params['la']
        psi = model_params['psi']
        p = model_params['p']
        
        # Configuración de los parámetros específicos para el modelo actual
        las = la * np.ones(shape=(1, 1), dtype=np.float64)
        
        # Inicialización de atributos para el modelo seleccionado
        super().__init__(states=[INFECTED], transmission_rates=las, removal_rates=[psi], ps=[p])
        self.la = la
        self.psi = psi
        
        # Almacenar el nombre del modelo actual
        self.current_model_name = model_params['Model']
        
        return self

    def get_name(self):
        # Devuelve el nombre específico del modelo actual
        return self.current_model_name if self.current_model_name else 'BD'

    def get_epidemiological_parameters(self):
        """Converts rate parameters to the epidemiological ones"""
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

# Ejemplo de uso
params = np.array([
    [2.0, 1.5, 3.0], # las
    [0.1, 0.2, 0.3], # psis
    [0.5, 0.6, 0.7], # ps
    [2.0, 5.0, 10.0] # ts
])

skyline_model = BirthDeathSkylineModel(params)

time = 6.0
actual_model = skyline_model.select_model(time)

if actual_model:
    print(actual_model.get_name())  # Imprime el nombre específico del modelo, como BD0, BD1, etc.
    print(actual_model.get_epidemiological_parameters())  # Usa el método adaptado para parámetros de este modelo
else:
    print("No hay modelo disponible para este tiempo.")