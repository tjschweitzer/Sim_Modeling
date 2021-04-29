import pandas
from scipy.optimize import minimize,rosen
import pandas as pd
import matplotlib.pyplot as plt
from solve_ode import *
from scipy.integrate import solve_ivp
import scipy
import math
import numpy as np
import prettytable
from numpy.polynomial.polynomial import Polynomial
from scipy import integrate
from scipy import optimize


# from numpy import polynomial as Polynomial
class SEIR_class:

    def __init__(self, filename):
        covid_filename = 'Provisional_COVID-19_Death_Counts_by_Week_Ending_Date_and_State.csv'
        pop_filename = 'Population.csv'
        self.death_data = {}
        self.population_dict = {}
        self.p= dict()
        self.computed_data = pd.read_pickle(filename)
        self.computed_dict= self.computed_data.set_index('NAME').T.to_dict('dict')
        raw_data = pd.read_csv(covid_filename, header=0, parse_dates=['End Date'])

        for i in range(len(raw_data)):
            state = raw_data.iloc[i]['State']
            end_date = raw_data.iloc[i]['End Date']
            weekly_death_count = raw_data.iloc[i]['COVID-19 Deaths']

            if math.isnan(weekly_death_count):
                weekly_death_count = 1
            if raw_data.iloc[i]['Group'] != "By Week":
                continue

            if state not in self.death_data:
                self.death_data[state] = {'dates': [],
                                          'deaths': []}

            self.death_data[state]['deaths'].append(weekly_death_count)
            self.death_data[state]['dates'].append(end_date)

        pop_data = pd.read_csv(pop_filename, sep=',', header=0)
        for i in range(len(pop_data)):
            state = pop_data.iloc[i]['Area']
            population = pop_data.iloc[i]['2019']
            if state[0] == '.':
                state = state[1:]
            self.population_dict[state] = population

    def set_parameter(self, q=.5, delta=6, gamma=16,
                      death_rate=.01, Eo_frac=1e-6, degree=6,
                      coefs=None, beta_o=.08):
        """
        This simple routine simply sets the parameters for the model.
        Note they are not all unity, I want you to figure out the
        appropriate parameter values.
        location - location to be modeled
        q - the attenuation of the infectivity, gamma in the population that is E
        delta - the length of time of asymptomatic exposure
        gamma - the length of time of infection
        death_rate - the fraction of infected dying
        Eo_frac    - a small number that is multiplied by the population to give the initially exposed
        degree     - degree of polynomial used for beta(t)
        coeffs     - the set of initial coefficients for the polynomial, None in many cases
        beta_o     - a constant initial value of beta, will be changed in the optimization
        """

        if coefs is None:
            self.beta_function = Polynomial.fit(self.t_eval,[beta_o]*self.t_eval,deg=degree)
        else:
            self.beta_function = Polynomial.basis(deg=degree, domain=[0,self.t_eval[-1]])
            self.beta_function.coef =coefs
            self.p['coefs']=coefs
        self.p['Eo_frac']=Eo_frac

        self.p['beta']=self.beta_function
        self.p['q'] = q
        self.p['delta'] = delta
        self.p['gamma'] = gamma
        self.p['death_rate'] = death_rate

        # Fill out the state vector y
        S = self.N
        E = S * Eo_frac
        I = 0
        R = 0
        D = 0
        self.y0 = np.array([S, E, I, R, D])

    def SEIRD(self, t, y):

        """
        Accepts the state of the system as:
        y[0] = Suseptable population - never impacted by the disease
        y[1] = Exposed population - has the disease, but not yet manifesting symptoms
        y[2] = Infected population - has the disease, and symptoms
        y[3] = Recovered population - has recovered from the disease, and is no longer suseptable

        This function returns the time deriviative of the state, dydt, and uses a dictionary
        of parameters p.
        """
        beta = self.p['beta']  # .1-.8
        q = self.p['q']
        delta = self.p['delta']
        gamma = self.p['gamma']
        N = self.p['n']
        death = self.p['death_rate']

        S, E, I= y[:3]

        beta_eval = beta(t)
        S_change = (beta_eval * S * (I + q * E)) / N
        E_change = E / delta
        I_change = I / gamma

        dS = -S_change
        dE = S_change - E_change
        dI = E_change - I_change
        dR = (1 - death) * I_change
        dD = death * I_change

        return np.array([dS, dE, dI, dR, dD])



    def set_location(self, location):
        """
        Given a location string, this function will read appropriate
        data files to set the population parameter N and
        data fields within the SEIR object appropriate time series of
        deaths from COVID-19.
        """
        if location not in self.death_data:
            print('Invalid Location')
            return
        self.location = location
        self.start_date = self.death_data[location]['dates'][0]
        self.end_date = self.death_data[location]['dates'][-1]
        self.total_days = (self.end_date - self.start_date).days
        self.N = float(self.population_dict[location].replace(",", ""))
        self.deaths = self.death_data[location]['deaths']
        self.deaths_cumsum = np.cumsum(self.deaths)
        self.p['n']=self.N
        self.t_eval = np.array([(date - self.start_date).days for date in self.death_data[self.location]['dates']],
                               dtype=int)


    def get_SSE(self, opt_params):
        """
        The hardest working routine - will
        1. accept a set of parameters for the polynomial coefficients and the death rate
        2. run the SEIR model using the ODE solver, solve_ivp
        3. return an Sum Square Error by comparing model result to data.
        """


        self.set_parameter(death_rate=opt_params[-1],coefs=opt_params[:-2],Eo_frac=opt_params[-2])

        beta_results = [i<0 for i in self.beta_function(self.t_eval)]

        if any(beta_results):
            # print("Beta F Failed")
            return 1e26

        if self.p['death_rate'] < 0:
            # print("Death rate Failed")
            return 1e26

        try:
            self.solution = solve_ivp(self.SEIRD, (0., self.t_eval[-1]), self.y0, "BDF",
                                      self.t_eval, dense_output=True)
        except:
            # print("Try failed")
            return 1e26

        sol_deaths = np.array(self.solution.y.T[:, 4])
        # sol_deaths = self.cumsum_to_weekly(sol_deaths)
        if np.array(self.death_data[self.location]['deaths']).shape == sol_deaths.shape:
            square_error = np.square(sol_deaths - self.deaths_cumsum)
            sse_iter = np.sum(square_error)
        else:
            sse_iter = 1e26

            # print(f"Size error {np.array(self.death_data[self.location]['deaths']).shape}  { sol_deaths.shape}")
        # print(f"{self.location}  Sum Square Error {sse_iter}")
        return sse_iter



    def get_minimize(self, opt_params, options, method='nelder-mead'):

        death_rate = opt_params[-1]


        Eo_frac = opt_params[-2]

        coefs = opt_params[:-2]

        self.set_parameter(q=.5, delta=7, gamma=15, Eo_frac=Eo_frac, coefs=coefs, death_rate=death_rate, degree=6)

        res = minimize(self.get_SSE, opt_params, method=method,
                       options=options)

        return res

    def cumsum_to_weekly(self,deaths):
            weekly_deaths = np.zeros(deaths.shape)
            for i in range(1, deaths.shape[0]):
                weekly_deaths[i] = deaths[i] - deaths[i - 1]
            return weekly_deaths

    def plot_results(self):


        """
        create a 4 panel plot with the following views:
        * Deaths modeled and deaths observed as a function of time.
        * 𝑅𝑜  as a function of time.
        * The susceptible, infected and recovered populations as a function of time.
        * The fraction of the population that has recovered as a function time.
        Observe that by passing a list of dates you can get a nicely formatted time axis.
        """
        weekly_deaths= self.cumsum_to_weekly( self.solution.y[-1,:])

        dates = self.death_data[self.location]['dates']
        Ro_plot = self.beta_function(self.t_eval)*self.p["gamma"]
        labels = ['Suseptable population', 'Exposed population', 'Infected population', 'Recovered population',
                  'Dead population']
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(dates, self.death_data[self.location]['deaths'], label='Observed', )
        axs[0, 0].plot(dates, weekly_deaths, label='Model')
        axs[0, 0].legend()
        axs[0, 0].set_title('Recordered Deaths')
        for i in range(len(self.solution['y'])):
            axs[1, 0].plot(self.death_data[self.location]['dates'], self.solution['y'][i], label=labels[i])
        axs[1, 0].legend()
        axs[0, 1].set_title('Ro')
        axs[0,1].plot( Ro_plot)
        axs[1, 1].plot(self.solution['y'][-2])
        axs[1, 1].set_title('')
        fig.set_figheight(15)
        fig.set_figwidth(15)
        axs[1,1].hlines(1 -  (1 / np.max(Ro_plot[:10])),xmin=dates[0],xmax=dates[-1],label="heard imunity", colors='k')
        plt.show()

        """ 
        create a 4 panel plot with the following views:

        * Deaths modeled and deaths observed as a function of time.
        * 𝑅𝑜  as a function of time.
        * The susceptable, infected and recovered populations as a function of time.
        * The fraction of the population that has recovered as a function time.
        Observe that by passing a list of dates you can get a nicely formated time axis.
        """
        plt.show()

    def get_all_locations(self):
        return self.death_data.keys()

    def get_all_ro(self):
        Ro= {}
        for location in self.computed_dict.keys():
            self.set_location(location)
            C = self.computed_dict[location]['C']
            ro = Polynomial.basis(deg=6, domain=[0,self.t_eval[-1]])
            ro.coef =C
            ro_eval = ro(self.t_eval)*15
            # print(ro_eval)
            Ro[location] = 1-(1 / np.max(ro_eval[:10]))
        return Ro

    def plot_location(self, location):
        self.set_location(location)
        current_data = self.computed_dict[location]
        # print(current_data)
        self.solution = current_data['solutions']

        weekly_deaths= self.cumsum_to_weekly( self.solution.y[-1,:])

        dates = self.death_data[self.location]['dates']
        print(f" Eo {current_data['Eo']}   Death Rate{current_data['Death_rate']}    C  {current_data['C']}")
        self.set_parameter( q=.5, delta=6, gamma=16,
                      death_rate=current_data['Death_rate'], Eo_frac=current_data['Eo'], degree=6,
                      coefs=current_data['C'], beta_o=.08)

        Ro_plot = self.beta_function(self.t_eval)*self.p["gamma"]
        labels = ['Suseptable population', 'Exposed population', 'Infected population', 'Recovered population']
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(dates, self.death_data[self.location]['deaths'], label='Observed', )
        axs[0, 0].plot(dates, weekly_deaths, label='Model')
        axs[0, 0].legend()
        axs[0, 0].set_title('Recordered Deaths')
        for i in range(len(labels)):
            axs[1, 0].plot(self.death_data[self.location]['dates'], self.solution['y'][i], label=labels[i])
        axs[1, 0].legend()
        axs[0, 1].set_title('Axis [1, 0]')
        axs[0,1].plot( Ro_plot)
        axs[1, 1].plot(self.solution['y'][-2]/ current_data["POPULATION"], 'tab:red')
        axs[1, 1].set_title('Axis [1, 1]')
        fig.set_figheight(15)
        fig.set_figwidth(15)
        axs[1,1].plot((1 - np.ones(len(self.t_eval)) * (1 / np.max(Ro_plot[:10]))), c='k')
        plt.show()

    def get_coefs_death_Eo(self,location):
        current_data = self.computed_dict[location]
        return current_data["C"], [current_data['Death_rate']],[current_data['Eo']]

    def plot_mcmc(self, location,x):
        self.set_location(location)

        Eo = x[-1]
        death_rate = x[-2]
        C = x[:-2]
        self.set_parameter(Eo_frac=Eo,death_rate=death_rate,coefs=C)
        self.solution = solve_ivp(self.SEIRD, (0., self.t_eval[-1]), self.y0, "BDF",
                                  self.t_eval, dense_output=True)
        self.plot_results()
if __name__== "__main__":
    app = SEIR_class("data_file_temp.pkl")
    # app.plot_location('Utah')
    # app = SEIR_class("data_file.pkl")
    # app.plot_location('Utah')

    all_locs = app.get_all_locations()
    deaths_results = []
    Ro_Results= []
    data_results = []
    print(all_locs)
    for locs in all_locs:
        print(locs)
        try:
            start_perams = [ 0.09347525,  0.05559237, -0.0764554,   0.12855742,  0.47815674, -0.13081831, -0.13717843,1E-6 ,  0.00148614]

            powell_options = {'disp': True, "xtol": 1e-6}
            app.set_location(locs)
            result =app.get_minimize(start_perams,powell_options,"Powell")
            nm_options = {'xatol': 1e-7, 'adaptive': True}

            result = app.get_minimize(result.x, nm_options, "nelder-mead")
            result = app.get_minimize(result.x, nm_options, "nelder-mead")
            # app.plot_results()

            solution = app.solution

        except:
            continue
        data_results.append({"NAME":locs, "Ro_max":max(app.beta_function(app.t_eval)[:10]), "POPULATION":app.p['n'],
                             'C':app.p['coefs'],'Eo':app.p['Eo_frac'],'Death_rate':app.p['death_rate'], "DEATHS": app.deaths_cumsum[-1],'solutions' : solution})

    data_temp = pandas.DataFrame.from_records(data_results)
    pandas.to_pickle(data_temp, "data_file.pkl")

