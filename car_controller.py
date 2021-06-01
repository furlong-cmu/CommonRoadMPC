import sys
sys.path.insert(0, './commonroad-vehicle-models/PYTHON/')

from vehiclemodels.init_ks import init_ks
from vehiclemodels.init_st import init_st
from vehiclemodels.init_std import init_std
from vehiclemodels.init_mb import init_mb
from vehiclemodels.parameters_vehicle1 import parameters_vehicle1
from vehiclemodels.parameters_vehicle2 import parameters_vehicle2
from vehiclemodels.vehicle_dynamics_ks import vehicle_dynamics_ks
from vehiclemodels.vehicle_dynamics_st import vehicle_dynamics_st
from vehiclemodels.vehicle_dynamics_std import vehicle_dynamics_std
from vehiclemodels.vehicle_dynamics_mb import vehicle_dynamics_mb

from constants import *

from scipy import signal
from multiprocessing import Pool
import time
import csv


from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as geom
import math



from scipy.integrate import odeint

#Covariance Matrix for the input distribution
INITIAL_COVARIANCE = [[0.0, 0], [0, 0.0]] 

#Covariance matrix for every next-step in the trajectory prediction
STEP_COVARIANCE = [[0.2, 0],[0,0.1]]
NUMBER_OF_STEPS_PER_TRAJECTORY = 20

# Must be between 0 and 1 and defines how close to a random walk the trajectories are
# 0: The trajectories are calculated by a completely random control with mean = 0 and variance = step_covariance0 
# 1: The trajectories are a random walk with mean = last control and variance = step_covariance
RANDOM_WALK = 0.3

#Number of trajectories that are calculated for every step
NUMBER_OF_INITIAL_TRAJECTORIES = 1000

#Covariance for calculating simmilar controls to the last one
STRATEGY_COVARIANCE=[[0.01, 0],[0, 0.01]]   
NUMBER_OF_TRAJECTORIES = 30

NUMBER_OF_NEXT_WAYPOINTS = 30

#With of the race track (ToDo) 
DIST_TOLLERANCE = 4 
INVERSE_TEMP = 10
DRAW_CHOSEN_SEQUENCE = True
DRAW_TRAJECTORIES = True

MAX_COST = 1000

def column(matrix, i):
        return [row[i] for row in matrix]
        

def solveEuler(func, x0, t, args):
    history = np.empty([len(t), len(x0)])
    history[0] = x0
    x = x0
    #Calculate dt vector
    for i in range(1, len(t)):
        x = x + np.multiply(t[i] - t[i-1] ,func(x, t, args[0], args[1]))
        history[i] = x
    return history



'''
Can calculate the next control step due to the cost function
'''

class CarController:
    def __init__(self, car):
        self.parameters = parameters_vehicle2()
        self.time = 0 #TODO:
        self.tControlSequence = 0.2  # [s] How long is a control input applied
        self.tEulerStep = 0.01 # [s] One step of solving the ODEINT
        self.tHorizon = 4 # [s] TODO How long should routes be calculated
        self.history = [] #Hostory of real car states
        self.simulated_history = [] #Hostory of simulated car states
        self.simulated_costs = [] #Hostory of simulated car states
        self.car = car
        self.track = self.car.track #Track waypoints for drawing
        self.last_control_input = [0,0]
        self.state = self.car.state
        self.collect_distance_costs = []
        self.collect_speed_costs = []
        self.collect_progress_costs = []
        self.best_control_sequenct = initial_sequence
        self.best_control_sequenct = [] #initial_sequence
        
        self.update_trackline()
       



    def set_state(self, state):
        self.state = state
    
    '''
    Dynamics of the simulated car from common road
    We can define different models here
    '''

    def update_trackline(self):
         # only Next NUMBER_OF_NEXT_WAYPOINTS points of the track
        waypoint_modulus = self.track.waypoints.copy()
        waypoint_modulus.extend(waypoint_modulus[:NUMBER_OF_NEXT_WAYPOINTS])

        closest_to_car_position = self.track.get_closest_index(self.state[:2])
        waypoint_modulus = waypoint_modulus[closest_to_car_position: closest_to_car_position+ NUMBER_OF_NEXT_WAYPOINTS]

        self.trackline = geom.LineString(waypoint_modulus)

    def func_KS(self,x, t, u, p):
        # f = vehicle_dynamics_ks(x, u, p)
        f = vehicle_dynamics_st(x, u, p)
        # f = vehicle_dynamics_std(x, u, p)
        # f = vehicle_dynamics_mb(x, u, p)
        return f

   
    '''
    Calculates the next system state due to a given state and control input
    For one time step
    @state: the system's state
    @control input 
    returns the simulated car state after tControlSequence [s]
    '''
    def simulate_step(self, state, control_input):
        t = np.arange(0, self.tControlSequence, self.tEulerStep) 
        x_next = solveEuler(self.func_KS, state, t, args=(control_input, self.parameters))
        # x_next = odeint(self.func_KS, state, t, args=(control_input, self.parameters))
        return x_next[-1]

    '''
    Simulates a hypothetical trajectory of the car due to a list of control inputs
    @control_inputs: list of control inputs, which last 0.01s each
    '''
    def simulate_trajectory(self, control_inputs):
        original_car_position = geom.Point(self.state[:2])
        simulated_time = self.time
        simulated_state = self.state
        simulated_trajectory = []
        cost = 0
        index = 0 
        number_of_states = len(control_inputs) 

        

        for control_input in control_inputs:
            if cost > MAX_COST:
                cost = MAX_COST
                # continue
            simulated_state = self.simulate_step(simulated_state, control_input)
            simulated_trajectory.append(simulated_state)

            discount = (number_of_states - 0.5 * index) / number_of_states
            cost +=  discount * self.cost_per_step(simulated_state)
            
            index += 1

        simulated_position = geom.Point(simulated_state[:2])
        progress = original_car_position.distance(simulated_position)
        # print("Progress", progress)
        progress_cost = 200/progress

        # print("progress_cost", progress_cost)
        # print("Cost", cost)

        cost = cost + progress_cost
        # print("Cost for whole trajectory", cost )
        return simulated_trajectory, cost 


    '''
    Returns a trajectory for every control input in control_inputs_distribution
    '''
    def simulate_trajectory_distribution(self, control_inputs_distrubution):
        self.simulated_history = []
        # start = time.time()
        results = []
        costs = []
        for control_input in control_inputs_distrubution:
            simulated_trajectory, cost = self.simulate_trajectory(control_input)
            results.append(simulated_trajectory)
            costs.append(cost)

        # with Pool(8) as p:
        #     results = p.map(self.simulate_trajectory, 
        #     control_inputs_distrubution) 

        self.simulated_history = results
        self.simulated_costs = costs
        # end = time.time()
        # print("TIME FOR 50 Trajectories")
        # print(end - start)
        return results, costs

    '''
    Returns a gaussian distribution around the last control input
    '''
    def sample_control_inputs(self, last_control_input):
        initial_steering, initial_acceleration = np.random.multivariate_normal(last_control_input, INITIAL_COVARIANCE, NUMBER_OF_INITIAL_TRAJECTORIES).T

        control_input_sequences = [0]*len(initial_steering)
        for i in range(len(initial_steering)):
            last_control_input = [initial_steering[i], initial_acceleration[i]]
            control_input_sequences[i] = []
            next_control_input = [round(initial_steering[i],3), round(initial_acceleration[i], 3)] 
            # next_control_input = [0, 0]
            for j in range(NUMBER_OF_STEPS_PER_TRAJECTORY):

                zero_control_input = np.array([0,0])
                last_control_input = np.array(last_control_input)
                mean_next_control_input = RANDOM_WALK * zero_control_input + (1 - RANDOM_WALK) * last_control_input

                step_steering, step_acceleration = np.random.multivariate_normal(mean_next_control_input, STEP_COVARIANCE).T
                
                control_input_sequences[i].append([step_steering, step_acceleration])
                next_control_input[0] = step_steering
                next_control_input[1] = step_acceleration
        control_input_sequences = np.array(control_input_sequences)

        return control_input_sequences


    def sample_control_inputs_similar_to_last(self, last_control_sequence):

        # start = time.time()
       
        #Not initialized
        if(len(last_control_sequence) == 0):
            return self.sample_control_inputs([0,0])

        #Delete the first step of the last control sequence because it is already done
        #To keep the length of the control sequence add one to the end
        last_control_sequence = last_control_sequence[1:]
        last_control_sequence = np.append(last_control_sequence, [0,0]).reshape(NUMBER_OF_STEPS_PER_TRAJECTORY,2)

        last_steerings = last_control_sequence[:,0]
        last_accelerations = last_control_sequence[:,1]

        control_input_sequences = np.zeros([NUMBER_OF_TRAJECTORIES, NUMBER_OF_STEPS_PER_TRAJECTORY, 2])

        for i in range(NUMBER_OF_TRAJECTORIES):
            steering_noise, acceleration_noise = np.random.multivariate_normal([0,0], STRATEGY_COVARIANCE, NUMBER_OF_STEPS_PER_TRAJECTORY).T

            next_steerings = last_steerings + steering_noise
            next_accelerations = last_accelerations  + acceleration_noise
            next_control_inputs = np.vstack((next_steerings, next_accelerations)).T
            control_input_sequences[i] = next_control_inputs
     
        control_input_sequences = np.array(control_input_sequences)

        # end = time.time()
        # print("TIME FOR ROLLOUT")
        # print(end - start)

        return control_input_sequences
        
    
    '''
    calculates the cost of a trajectory
    '''
    def cost_function(self, trajectory):
        max_distance = 5
        
        track = self.trackline
        distance_cost = 0
        speed_cost = 0
        progress_cost = 0

        distance_cost_weight = 1
        speed_cost_weight = 50
        progress_cost_weight =100

        number_of_critical_states = 10

        index = 0
        number_of_states = len(trajectory) 

        for state in trajectory:
            discount = (number_of_states - 0.5 * index) / number_of_states

            simulated_position = geom.Point(state[0],state[1])
            distance_to_track = simulated_position.distance(track)
            speed = state[3]
            speed_cost += discount * speed
            distance_cost += discount * distance_to_track
            if(distance_to_track > max_distance):
                if(index < number_of_critical_states):
                    distance_cost += 100
                    # print("Potential crash during critical distance")
            index += 1

        original_car_position = geom.Point(self.state[0],self.state[1])
        progress = original_car_position.distance(simulated_position)


        speed_cost = 1 /speed_cost
        progress_cost = 1/progress

        self.collect_distance_costs.append(distance_cost)
        self.collect_speed_costs.append(speed_cost)
        self.collect_progress_costs.append(progress_cost)

        cost = distance_cost_weight * distance_cost + speed_cost_weight * speed_cost + progress_cost_weight * progress_cost
        # print("Cost", cost)

    

        return cost

    def cost_per_step(self, state):
        # start = time.time()


        max_distance = 5
        
        track =self.trackline
        distance_cost = 0
        speed_cost = 0
        progress_cost = 0

        distance_cost_weight = 1
        speed_cost_weight = 100
        # progress_cost_weight =100

        simulated_position = geom.Point(state[0],state[1])
        distance_to_track = simulated_position.distance(track)
        speed = state[3]
        speed_cost +=  speed
        distance_cost +=  distance_to_track
        if(distance_to_track > max_distance):
            distance_cost += 100
            # print("Potential crash during critical distance")

        speed_cost = 1 /speed_cost

        self.collect_distance_costs.append(distance_cost)
        self.collect_speed_costs.append(speed_cost)
        self.collect_progress_costs.append(progress_cost)

        cost = distance_cost_weight * distance_cost + speed_cost_weight * speed_cost # + progress_cost_weight * progress_cost
        # print("CostPerStep", cost)
        cost = max(0,min(cost, 100))
        # print("Cost_per_step", cost)

        # end = time.time()
        # print("TIME FOR COST PER STEP")
        # print(end - start)
        return cost


    
    '''
    Does one step of control and returns the best control input for the next time step
    '''
    def control_step(self):
        self.update_trackline()
        self.last_control_input[0] = min(self.last_control_input[0], 1)
        self.last_control_input[1] = min(self.last_control_input[1], 0.5)

        # control_sequences = self.sample_control_inputs(self.last_control_input)
        control_sequences = self.sample_control_inputs_similar_to_last(self.best_control_sequenct)



        # input_samples = u_dist #those are the handcrafted inputs
        simulated_history, costs = self.simulate_trajectory_distribution( control_sequences )

        # print("Costs", costs)
        trajectory_index = 0    
        lowest_cost = 100000
        best_index = 0

        weights = np.zeros(len(simulated_history))

        for trajectory in simulated_history:
            cost = costs[trajectory_index]
            #find best
            if cost < lowest_cost:
                best_index = trajectory_index
                lowest_cost = cost
         
            weight =  math.exp((-1/INVERSE_TEMP) * cost)
            # print("Weight", weight)
            weights[trajectory_index] = weight
            trajectory_index += 1

        best_conrol_sequence = control_sequences[best_index]

        #Finding weighted avg input
        next_control_sequence = np.average(control_sequences,axis=0, weights=weights )
        next_control_sequence = signal.medfilt(next_control_sequence, [1, 1])

        # print("next_control_sequence",next_control_sequence)

        self.best_control_sequenct = next_control_sequence
        


        return next_control_sequence
        # return best_conrol_sequence

    """
    draws the simulated history (position and speed) of the car into a plot for a trajectory distribution resp. the history of all trajectory distributions
    """  
    def draw_simulated_history(self, waypoint_index = 0, chosen_trajectory = []):

        plt.clf()

        fig, position_ax = plt.subplots()

        plt.title("Model Predictive Path Integral")
        plt.xlabel("Position x [m]")
        plt.ylabel("Position y [m]")

        s_x = []
        s_y = []
        costs = []
        i = 0
        ind = 0
        indices = []
        for trajectory in self.simulated_history:
            cost = self.simulated_costs[i]
            for state in trajectory:
                if(state[0] > 1):
                    s_x.append(state[0])
                    s_y.append(state[1])
                    costs.append(cost)
                    indices.append(cost)
                    ind += 1
            i += 1
        if(DRAW_TRAJECTORIES):
            trajectory_costs = position_ax.scatter(s_x,s_y, c=indices)
            colorbar = fig.colorbar(trajectory_costs)
            colorbar.set_label('Trajectory costs')


        #Draw car position
        p_x = self.state[0]
        p_y = self.state[1]
        # print("car state: ", self.state)
        position_ax.scatter(p_x,p_y, c ="#FF0000", label="Current car position")

        #Draw waypoints
        waypoint_index = self.track.get_closest_index(self.state[:2])
        w_x = self.track.waypoints_x[waypoint_index:waypoint_index+NUMBER_OF_NEXT_WAYPOINTS]
        w_y = self.track.waypoints_y[waypoint_index:waypoint_index+NUMBER_OF_NEXT_WAYPOINTS]
        position_ax.scatter(w_x,w_y, c ="#000000", label="Next waypoints")

        #Plot Chosen Trajectory
        t_x = []
        t_y =[]
        for state in chosen_trajectory:
            t_x.append(state[0])
            t_y.append(state[1])
        if(DRAW_CHOSEN_SEQUENCE):
            plt.scatter(t_x, t_y, c='#D94496', label="Chosen control")
            plt.legend(  fancybox=True, shadow=True, loc="best")

        
        plt.savefig('sim_history.png')
        return plt


    '''
    This is only needed to deploy on l2race...
    '''
    # def read(self):
    #     """
    #     Computes the next steering angle tying to follow the waypoint list

    #     :return: car_command that will be applied to the car
    #     """

    #     self.set_state(self.car.car_state)
    #     next_control_input = self.control_step(self.last_control_input)
    #     # self.draw_simulated_history()
    #     # print("NEXT CONTROL",  next_control_input)
    #     self.last_control_input = next_control_input
        
    #     self.car_command.steering = next_control_input[0]
    #     self.car_command.throttle = next_control_input[1]

    #     return self.car_command

    