import math
import random
import numpy as np
import matplotlib.pyplot as plt

class SimulatedAnnealing:

    def __init__(self, num_targets=17, area_surv_x=500, area_surv_y=500,
                 types_sensors=[1, 2, 3], range_sensors=[100, 70, 30],
                 cost_sensors=[300, 170, 65], sensor_coverage=1, initial_placement="random", seed=13):

        """
        Simulated Annealing class initialization function.

        Parameters
        ----------
        num_targets : int -> 17
            Number of target sensors in the surveillance area.

        area_surv_x : int -> 500
            Surveillance area width.

        area_surv_y : int -> 500
            Surveillance area height.

        types_sensors : list[int] -> [1,2,3]
            List containing types of sensors. Sensors should be incrementally named starting from 1.

        range_sensors : list[float] -> [100,70,30]
            Range of corresponding type of sensor.

        cost_sensors : list[float] -> [300,170,65]
            Cost of corresponding type of sensor.

        sensor_coverage : int -> 1
            Required number of sensor covering each target in the final solution.

        initial_placement : Str -> "random"
            Value can only be "random" or "deterministic" or "min". Based on value, uses appropriate sensor initialization.

        seed : int -> 13
            Seed for radom generator functions. Helps keep similarity between runs.

        Attributes
        ----------
        num_types_sensors : int
            Calculated as len(types_sensors)

        coord_targets : list[coord_x, coord_y]
            Cartesian co-ordinates of the (num_targets) target positions. Generated randomly.

        coord_sensors : list[coord_x, coord_y, type_sensor]
            Cartesian co-ordinates of the randomly deployed sensors.

        cost : list
            List for holding cost of each step during the optimization process.

        """

        # setting all instance variables
        self.area_surv = np.zeros((area_surv_x, area_surv_y), int)
        self.num_targets = num_targets
        self.area_surv_x = area_surv_x
        self.area_surv_y = area_surv_y
        self.types_sensors = types_sensors
        self.range_sensors = range_sensors
        self.cost_sensors = cost_sensors
        self.sensor_coverage = sensor_coverage
        self.num_types_sensors = len(types_sensors)
        self.cost = []
        self.initial_placement = initial_placement
        self.seed = seed

        self.coord_targets = self.get_target_coords(seed=13)

        if self.initial_placement == "random":
            self.coord_sensors = self.get_sensor_coords_random()
        elif self.initial_placement == "deterministic":
            self.coord_sensors = self.get_sensor_coords_determinstic()
        elif self.initial_placement == "min":
            self.coord_sensors = self.get_sensor_coord_minimum_cost()

        self.update_area_surv()

    # generate random target co-ordinates
    def get_target_coords(self, seed):
        """
        Generates random (num_targets) cartesian co-ordinates for targets from a uniform distribution.

        Returns
        -------
        coord_targets : list
            co-ordiantes of the targets.

        Parameters
        ----------
        seed : int
            Seed for random number generation

        coord_sensors_x : list
            Random X coordinates for sensors.

        coord_sensors_y : list
            Random Y coordinates for sensors.

        Attributes
        ----------
        coord_targets : list

        """
        # targets are randomly and uniformly scattered in surv. area

        # set seed for reproducability
        np.random.seed(seed)

        # get random co-ordinates from random uniform distribution
        coord_sensors_x = np.random.uniform(low=0, high=self.area_surv_x - 1, size=(int(self.num_targets), 1))
        coord_sensors_y = np.random.uniform(low=0, high=self.area_surv_y - 1, size=(int(self.num_targets), 1))

        # set random type of sensors in the random co-ordinates
        coord_sensors = [[int(coord_sensors_x[i]), int(coord_sensors_y[i])] for i in range(self.num_targets)]

        return coord_targets

    # generate random target co-ordinates
    def get_target_coords(self, seed, num_targets):
        """
        Generates random (num_targets) cartesian co-ordinates for targets from a uniform distribution.

        Returns
        -------
        coord_targets : list
            co-ordiantes of the targets.

        Parameters
        ----------
        seed : int
            Seed for random number generation

        num_targets : int
            Number of targets.

        Attributes
        ----------
        coord_targets : list

        """
        # targets are randomly and uniformly scattered in surv. area

        # set seed for reproducability
        np.random.seed(seed)

        # get random co-ordinates from random uniform distribution
        coord_targets = np.random.uniform(low=0, high=499, size=(num_targets, 2))

        # typecase float co-ordinates to int
        for i in range(coord_targets.shape[0]):
            for j in range(coord_targets.shape[1]):
                coord_targets[i][j] = int(coord_targets[i][j])

        # print(coord_targets)

        return coord_targets

    # plot the target co-ordinates in the surveillance area
    def plot_target_coords(self, coord_target):
        """
        Plot the target co-ordinates in the surveillance area

        Parameters
        ----------
        coord_targets : list[coord_x, coord_y]
            Co-ordinates of the targets in the surveillance area

        """
        plt.scatter(x=coord_target[:, [0]], y=coord_target[:, [1]], marker=(5, 2), label="Targets")
        plt.xlim(0, self.area_surv_x)
        plt.ylim(0, self.area_surv_y)
        plt.title("Random Target Placement in Surveillance Area")
        plt.legend(loc="lower center")
        plt.grid()
        plt.show()

    # set sensors in a semi-deterministic way
    def get_sensor_coords_determinstic(self):
        """
        Function for generating sensors in a semi-deterministic way.

        Returns
        -------
        coord_sensors : list[coord_x, coord_y, type_sensor]
            Co-ordinates of the placed sensors.

        Attributes
        ----------
        min_range : float
            The minimum range among the given sensors.

        max_num_of_sensors : int
            Caclulate the max num of sensors of the minimum range to cover the surveillance area.
            This is calculated by (surveillance area) / 2 * pi * min_range

        root_max_num_of_sensors : int
            Calculated as root(max_num_of_sensors). This is to determine how many sensors can be used
            to in a square fashion to cover the surveillance area.

        distance_between_sensors_x : int
            This is the distance between sensors if we put them in a square way over the surveillance area.
            This is along the x axis.

        distance_between_sensors_y : int
            This is along the y axis.

        """
        coord_sensors = []

        # get the sensor with minimum range
        min_range = min(self.range_sensors)

        # if we try to cover the surveillance area, how many sensors with minimum coverage will we need?
        max_num_of_sensors = round((self.area_surv_x * self.area_surv_y) / (2 * 3.1416 * min_range))

        # how do we put these sensors in the surveillance area in a square block fashion?
        root_max_num_of_sensors = round(max_num_of_sensors ** 0.5)

        # now we calculate the distance between the sensors
        distance_between_sensors_x = round(self.area_surv_x / root_max_num_of_sensors)
        distance_between_sensors_y = round(self.area_surv_y / root_max_num_of_sensors)

        # put random sensors of random type distance_between_sensors_ meters away from one another
        # in the entire surveillance area
        for i in range(0, self.area_surv_x, distance_between_sensors_x):
            for j in range(0, self.area_surv_y, distance_between_sensors_y):
                # choose a sensor of random type to put in that co-ordinate
                coord_sensors.append([i, j, np.random.choice(self.types_sensors)])

        return coord_sensors

    # set sensors in a complete random fashion - random co-ordinates and random sensor types
    def get_sensor_coords_random(self):
        """
        Function for generating sensor co-ordinates in a random way.

        Returns
        -------
        coord_sensors_with_types : list[coord_x, coord_y, type_sensor]
            Co-ordinates of the placed sensors.

        Attributes
        ----------
        range_s : float
            Range of specific type of sensor.

        num_s : float
            Maximum number of sensors with range_s required to cover the entire surveillance area.

        num_sensors : list
            Contains num_s for each corresponding type of sensor.

        tot_num_sensors: int
            Calculated as sum(tot_num_sensors).

        coord_sensors_x : list
            Random X coordinates for sensors.

        coord_sensors_y : list
            Random Y coordinates for sensors.

        """
        num_sensors = []

        # for each type of sensor, calculate the number of sensors that should cover the
        # entire surveillance area

        # maximum number of sensors = surveillance_area / 2 * pi * sensor_range
        # we calculate the maximum number of sensors required to cover the area for each type of sensor
        for sensor in self.types_sensors:
            range_s = self.range_sensors[sensor - 1]
            num_s = (self.area_surv_x * self.area_surv_y) / (2 * (3.1416) * range_s)
            num_sensors.append(int(round(num_s)))

        # get total number of sensors required
        tot_num_sensors = sum(num_sensors)

        # get total number of sensors equivalent co-ordinates
        np.random.seed(self.seed)

        # get random co-ordinates
        coord_sensors_x = np.random.uniform(low=0, high=self.area_surv_x - 1, size=(int(tot_num_sensors), 1))
        coord_sensors_y = np.random.uniform(low=0, high=self.area_surv_y - 1, size=(int(tot_num_sensors), 1))

        # set random type of sensors in the random co-ordinates
        coord_sensors_with_types = [[int(coord_sensors_x[i]), int(coord_sensors_y[i]),
                                     np.random.choice(self.types_sensors)] for i in range(tot_num_sensors)]

        return coord_sensors_with_types

    def get_sensor_coord_minimum_cost(self):
        """
        Function for placing the sensor with the minimum cost in a semi-deterministic way.

        Returns
        -------
        coord_sensors : list[coord_x, coord_y, type_sensor]
            Co-ordinates of the placed sensors.

        Attributes
        ----------
        index_mix_cost : int
            index of the sensor with the minimum cost

        range_min_cost : int
            Range of the sensor with the minimum cost

        type_min_cost : int
            Type of the sensor with the minimum cost

        max_num_of_sensors : int
            Caclulate the max num of sensors of the minimum range to cover the surveillance area.
            This is calculated by (surveillance area) / 2 * pi * min_range

        root_max_num_of_sensors : int
            Calculated as root(max_num_of_sensors). This is to determine how many sensors can be used
            to in a square fashion to cover the surveillance area.

        distance_between_sensors_x : int
            This is the distance between sensors if we put them in a square way over the surveillance area.
            This is along the x axis.

        distance_between_sensors_y : int
            This is along the y axis.

        """
        coord_sensors = []

        # get the sensor with minimum cost
        index_min_cost = self.cost_sensors.index(min(self.cost_sensors))
        range_min_cost = self.range_sensors[index_min_cost]
        type_min_cost = self.types_sensors[index_min_cost]

        # if we try to cover the surveillance area, how many sensors with minimum coverage will we need?
        max_num_of_sensors = round((self.area_surv_x * self.area_surv_y) / (2 * 3.1416 * range_min_cost))

        # how do we put these sensors in the surveillance area in a square block fashion?
        root_max_num_of_sensors = round(max_num_of_sensors ** 0.5)

        # now we calculate the distance between the sensors
        distance_between_sensors_x = round(self.area_surv_x / root_max_num_of_sensors)
        distance_between_sensors_y = round(self.area_surv_y / root_max_num_of_sensors)

        # put random sensors of random type distance_between_sensors_ meters away from one another
        # in the entire surveillance area
        for i in range(0, self.area_surv_x, distance_between_sensors_x):
            for j in range(0, self.area_surv_y, distance_between_sensors_y):
                # put the type of sensor with minimum cost in that co-ordinate
                coord_sensors.append([i, j, type_min_cost])

        return coord_sensors

    # function to update the surveillance area after any change in optimal sensor placement
    def update_area_surv(self):
        """
        Function to update the surveillance area after any change in optimal sensor placement

        Attributes
        ----------
        area_surv : list
            2D matrix representing the surveillance area - only types of sensors are represented.
            0 is used to represent free space. Each type of sensor is put in the respective co-ordinate.

        """
        area_surv = np.zeros((self.area_surv_x, self.area_surv_y), int)

        for coord in self.coord_sensors:
            # set the type of sensor in the surveillance area
            self.area_surv[coord[0]][coord[1]] = coord[2]

    # function to get a surveillance area given co-ordinate of sensors
    # used for getting next solution area during the optimization process
    def get_intermediate_area_surv(self, intermediate_coord_sensors):
        """
        Function to generate an intermediate surveillance are given sensor co-ordinates.

        Parameters
        ----------
        intermediate_coord_sensors : list[coord_x, coord_y, type_sensor]
            Sensor co-ordinats generated in an intermediate solution

        Attributes
        ----------
        area_surv : list
            2D matrix representing the surveillance area - only types of sensors are represented.
            0 is used to represent free space. Each type of sensor is put in the respective co-ordinate.

        """
        area_surv = np.zeros((self.area_surv_x, self.area_surv_y), int)

        for coord in intermediate_coord_sensors:
            area_surv[coord[0]][coord[1]] = coord[2]

        return area_surv

    # calculate total cost given a surveillance area
    def get_cost(self, area_surv):
        """
        Function for calculating the total cost given a surveillance area.

        Returns
        -------
        total_cost : float
            Total cost of all sensors deployed in the surveillance area.

        Parameters
        ----------
        area_surv : list
            2D matrix representing the surveillance area - only types of sensors are represented.
            0 is used to represent free space. Each type of sensor is put in the respective co-ordinate.

        Attributes
        ----------
        count_sensors : list
            List containing the number of sensors of each type.

        """
        # list holds number of sensors of each type
        count_sensors = [0 for i in range(self.num_types_sensors)]

        # count the number of sensors of each type
        for i in range(area_surv.shape[0]):
            for j in range(area_surv.shape[1]):

                if area_surv[i][j] == 0:
                    pass
                else:
                    count_sensors[area_surv[i][j] - 1] += 1

        # calculate total cost by multiplying number of sensors with respective cost
        total_cost = sum([count_sensors[i] * self.cost_sensors[i] for i in range(self.num_types_sensors)])

        # print(total_cost)

        return total_cost

    # function to check whether a solution meets the coverage constraint
    def check_coverage(self, area_surv):
        """
        Function for checking whether a given solution meets the coverage constraint.

        Returns
        -------
        Boolean
            True if coverage constraint is met, False otherwise.

        Parameters
        ----------
        area_surv : list
            2D matrix representing the surveillance area - only types of sensors are represented.
            0 is used to represent free space. Each type of sensor is put in the respective co-ordinate.

        Attributes
        ----------
        target_covered_by_number_sensors : list
            Contains the number of sensors covering each target

        type_sensor_index : int
            List index of the type of sensor used.

        """

        # must ensure 100% coverage with any k,
        # meaning each target covered by at least k number of sensors

        # list holds the number of sensors coverint each target co-ordinate
        target_covered_by_number_sensors = np.zeros((self.num_targets), int)

        print("Total number of sensors : {}".format(len((self.coord_sensors))))

        # for each target, calculate the number of sensors that cover the target
        # a target is covered if euclidean distance between sensor coordinate and target coordinate
        # is < range of sensor
        for cid, coord in enumerate(self.coord_targets):
            for i in range(area_surv.shape[0]):
                for j in range(area_surv.shape[1]):
                    if area_surv[i][j] == 0:
                        pass
                    else:
                        # get the index of the type of sensor
                        # as type_sensors is calculated from 1 (eg. [1, 2, 3])
                        # index for any type of sensor is type_of_sensor - 1
                        type_sensor_index = area_surv[i][j] - 1

                        # if euclidean distance between coordinates < range of sensor
                        # increse counter for target
                        if np.linalg.norm(coord - [i, j]) < self.range_sensors[type_sensor_index]:
                            target_covered_by_number_sensors[cid] += 1

        # if any target is not covered by set coverage criteria
        # return false, true otherwise
        for i in target_covered_by_number_sensors:
            if i < self.sensor_coverage:
                return False

        print("Number of sensors covering targets : {}".format(target_covered_by_number_sensors))

        return True

    # function for removing a random sensor from solution
    def remove_a_sensor(self):
        """
        Function for generating neighboring solution by removing a random sensor while optimization.

        Returns
        -------
        coord_sensors_copy : list
            Copy of the class variable coord_sensors

        """
        coord_sensors_copy = self.coord_sensors.copy()

        # shuffle the solution
        random.shuffle(coord_sensors_copy)

        # remove a sensor co-ordinate
        coord_sensors_copy.pop()

        return coord_sensors_copy

    # get optimized solution using the simulated annealing method
    def optimize(self):
        """
        Function for optimizing another function using the simulated annealing algorithm.

        Returns
        -------
        solution : list
            Same as class variable area_surv, but the final optimized solution.

        Attributes
        ----------
        initial_temp : float
            Starting temperature of Simulated Annealing optimization

        final_temp : float
            Stopping temperature of Simulated Annealing optimization

        alpha : float
            The amount by which the temperature decreases.

        step : int
            The count of how many turns the optimization has run

        runs_without_changes : int
            The count of how many steps have passed without any changes.

        current_temp : int
            The temperature of the running step.

        current_state : arr_surv
            The variable holds the current surveillance area matrix.

        previous_solution : arr_surv
            The variable holds the surveillance area matrix from the previous step.

        intermediate_coord_sensors : list
            Sensor co-ordinates of the neighboring solution.

        next_solution : arr_surv
            The neighboring solution chosen by removing a random sensor from previous solution

        current_cost : float
            Cost of the current solution

        next_cost : float
            Cost of the next proposed solution

        cost_diff : float
            Cost differenece between current_cost and next_cost

        """
        # setting up initial temparature parameters
        initial_temp = 90
        final_temp = .1
        alpha = 0.01
        step = 0

        # number of steps where there are no changes in the solution
        runs_without_changes = 0

        # set up initial temperature
        current_temp = initial_temp

        # Start by initializing the current state with the initial state
        current_state = self.area_surv

        # intital cost
        self.cost.append(self.get_cost(current_state))

        solution = current_state
        previous_solution = current_state

        while current_temp > final_temp and runs_without_changes <= 100:

            print("Step : {} Temp : {} Without changes : {}".format(step, current_temp, runs_without_changes))

            # get a next random solution
            intermediate_coord_sensors = self.remove_a_sensor()
            next_solution = self.get_intermediate_area_surv(intermediate_coord_sensors)

            # at first check if coverage constrained is maintained in the next solution
            if self.check_coverage(next_solution):

                # calculate cost difference between solutions
                current_cost = self.get_cost(current_state)
                next_cost = self.get_cost(next_solution)

                cost_diff = current_cost - next_cost

                # accept new solution if it is better
                if cost_diff > 0:

                    # update solution
                    solution = next_solution

                    # update sensor coordinates and update surveillance area
                    self.coord_sensors = intermediate_coord_sensors
                    self.update_area_surv()

                    # update current solution
                    current_state = solution

                    print("Previous Cost : {}".format(current_cost))
                    print("Next Cost : {}".format(next_cost))
                    self.cost.append(next_cost)

                    print("Accepted new solution")

                # even if solution is bad, accept it with probability of e^(-cost/temp)
                else:
                    if random.uniform(0, 1) < math.exp(-cost_diff / current_temp):
                        # update solution
                        solution = next_solution

                        # update state
                        self.coord_sensors = intermediate_coord_sensors
                        self.update_area_surv()

                        current_state = solution

                        print("Previous Cost : {}".format(current_cost))
                        print("Next Cost : {}".format(next_cost))
                        self.cost.append(next_cost)

                        print("Accepted bad solution")

            # decrease temperature
            current_temp -= alpha

            # check if current solution is the same with the previous solution
            if np.mean(solution) == np.mean(previous_solution):
                runs_without_changes += 1
            else:
                runs_without_changes = 0

            # set previous solution
            previous_solution = solution

            step += 1

        return solution