import numpy as np
import time
import sys
import gc
import concurrent.futures

import matplotlib.pyplot as plt
from copy import deepcopy


class GeneticAlgorithm:

    def __init__(self, num_targets=5, area_surv_x=50, area_surv_y=50,
                 types_sensors=[1, 2, 3], range_sensors=[10, 7, 3],
                 cost_sensors=[300, 170, 65], sensor_coverage=1, initial_placement="random", population_size=250,
                 num_generations=2000, crossover_rate=0.8, point_crossover=2, mutation_rate=0.1,
                 point_mutation=50, seed=13):

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

        # TODO : Make more dynamic

        # setting all instance variables
        # self.area_surv = np.zeros((area_surv_x, area_surv_y), int)
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
        self.population_size = population_size
        self.num_generations = num_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.point_crossover = point_crossover
        self.point_mutation = point_mutation
        self.seed = seed

        assert (self.point_crossover % 2 == 0 or self.point_crossover == 1), "Crossover points must be even-numbered"

        self.population_fitness_history = []
        self.best_solution_history = []

        self.coord_targets = self.get_target_coords(seed=13)

        # by default population is in gene seq
        self.population = self.generate_initial_population()

        # initializing best solution chromosome and it's fitness
        self.best_fitness = self.evaluate_fitness_single(self.population[0])
        self.best_chromosome = self.population[0]

        print("Population shape : ", self.population.shape)
        print("Total size of population in memory  (MB) : ", sys.getsizeof(self.population) / 1048576)

        # self.optimize()
        # self.update_area_surv()

        self.init_config = {"num_targets": self.num_targets,
                            "area_surv_x": self.area_surv_x,
                            "area_surv_y": self.area_surv_y,
                            "types_sensors": self.types_sensors,
                            "range_sensors": self.range_sensors,
                            "cost_sensors": self.cost_sensors,
                            "sensor_coverage": self.sensor_coverage,
                            "num_types_sensors": self.num_types_sensors,
                            "cost": self.cost,
                            "initial_placement": self.initial_placement,
                            "population_size": self.population_size,
                            "num_generations": self.num_generations,
                            "crossover_rate": self.crossover_rate,
                            "mutation_rate": self.mutation_rate,
                            "point_crossover": self.point_crossover,
                            "point_mutation": self.point_mutation,
                            "seed": self.seed,
                            "population_fitness_history": self.population_fitness_history,
                            "best_solution_history": self.best_solution_history,
                            "coord_targets": self.coord_targets,
                            "population": self.population,
                            "best_fitness": self.best_fitness,
                            "best_chromosome": self.best_chromosome}

    def save_experiment_results(self, filename=None):

        timestamp = time.time()

        if filename == None:
            filename = "./cc_experiments/{}_pm{}_pc{}_ps{}_ng{}_x{}_y{}.csv".format(
                timestamp, self.point_mutation, self.point_crossover, self.population_size, self.num_generations,
                self.area_surv_x, self.area_surv_y)

        # file format - best_solution + population fitness history
        """
        save_data = np.array([])
        for bsol, phist in zip(self.best_solution_history, self.population_fitness_history):
            row = [i for i in bsol]
            for i in phist:
                row.append(i)

            save_data.append(row)

        """

        print("Saving to file: {}".format(filename))
        # np.savetxt(filename, save_data, delimeter=",")
        save_file = np.hstack((self.best_solution_history, self.population_fitness_history))
        np.savetxt(filename, save_file, "%s", ",")

        gc.collect()

    def re_init(self):

        self.num_targets = self.init_config["num_targets"]
        self.area_surv_x = self.init_config["area_surv_x"]
        self.area_surv_y = self.init_config["area_surv_y"]
        self.types_sensors = self.init_config["types_sensors"]
        self.range_sensors = self.init_config["range_sensors"]
        self.cost_sensors = self.init_config["cost_sensors"]
        self.sensor_coverage = self.init_config["sensor_coverage"]
        self.num_types_sensors = self.init_config["num_types_sensors"]
        self.cost = self.init_config["cost"]
        self.initial_placement = self.init_config["initial_placement"]
        self.population_size = self.init_config["population_size"]
        self.num_generations = self.init_config["num_generations"]
        self.crossover_rate = self.init_config["crossover_rate"]
        self.mutation_rate = self.init_config["mutation_rate"]
        self.point_crossover = self.init_config["point_crossover"]
        self.point_mutation = self.init_config["point_mutation"]
        self.seed = self.init_config["seed"]

        self.population_fitness_history = []
        self.best_solution_history = []

        self.coord_targets = self.init_config["coord_targets"]

        # by default population is in gene seq
        self.population = self.init_config["population"]

        # initializing best solution chromosome and it's fitness
        self.best_fitness = self.init_config["best_fitness"]
        self.best_chromosome = self.init_config["best_chromosome"]

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

        Attributes
        ----------
        coord_targets : list

        """
        # targets are randomly and uniformly scattered in surv. area

        # set seed for reproducability
        # np.random.seed(seed)

        # get random co-ordinates from random uniform distribution
        coord_targets = np.random.uniform(low=0, high=self.area_surv_x, size=(self.num_targets, 2))

        # typecase float co-ordinates to int
        for i in range(coord_targets.shape[0]):
            for j in range(coord_targets.shape[1]):
                coord_targets[i][j] = int(coord_targets[i][j])

        # print(coord_targets)

        return coord_targets

    def get_sensor_type_in_int(self, sensor):
        if sensor == "00":
            return 0
        elif sensor == "01":
            return 1
        elif sensor == "10":
            return 2
        elif sensor == "11":
            return 3

    def get_sensor_type_in_str(self, sensor):
        if sensor == 0:
            return "00"
        elif sensor == 1:
            return "01"
        elif sensor == 2:
            return "10"
        elif sensor == 3:
            return "11"

    def convert_polar2d_to_gene_seq(self, population):
        # population = [[16, 12, 1], [12, 14, 3], ....]
        # TODO : change population to chromosome, only variable name change is sufficient as generate_population is
        # handling the other stuff.
        total_area = self.area_surv_x * self.area_surv_y

        # initialiity we assume all positions do not have
        population_gene_format = np.reshape(np.array(["00" for _ in range(total_area)]),
                                            (self.area_surv_x, self.area_surv_x))

        for sensor in population:
            sx = sensor[0]
            sy = sensor[1]
            st = sensor[2]

            # changing the type of sensor in that position to the sensor type
            population_gene_format[sx][sy] = self.get_sensor_type_in_str(st)

        return population_gene_format.flatten()

    def convert_gene_to_polar2d_coords(self, population):
        # population = ["00", "01", ..]
        # gene will have 18 bits, 8 bits x, 8 bits y, 2 bits type

        population_polar2d_format = np.array([self.get_sensor_type_in_int(i) for i in population])

        return np.resize(population_polar2d_format, (self.area_surv_x, self.area_surv_y))

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
        tot_num_sensors = int(np.average(num_sensors))

        # get total number of sensors equivalent co-ordinates
        # np.random.seed(self.seed)

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

    # function to check whether a solution meets the coverage constraint
    def check_coverage_population(self):
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
        is_covered = []

        for chromosome in self.population:
            is_covered.append(self.check_coverage_single(chromosome))

        if sum(is_covered) == len(is_covered):
            return True
        else:
            return False

    # function to check whether a solution meets the coverage constraint
    def check_coverage_single(self, chromosome):
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

        # chromosome = ["00", "01", "10",...]

        area_surv = self.convert_gene_to_polar2d_coords(chromosome)

        # must ensure 100% coverage with any k,
        # meaning each target covered by at least k number of sensors

        # list holds the number of sensors coverint each target co-ordinate
        target_covered_by_number_sensors = np.zeros((self.num_targets), int)

        # print("Total number of sensors : {}".format(len((self.coord_sensors))))

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
                return 0

        # print("Number of sensors covering targets : {}".format(target_covered_by_number_sensors))

        return 1

    def generate_initial_population(self):

        # each sensor co-ordinate is a chromosome
        # a population is a collection of sensor-coordinates

        # population has not format yet, here population would be the array of multiple sensor_coordinates
        population = []

        if self.initial_placement == "random":
            for _ in range(self.population_size):
                coord_sensors = self.get_sensor_coords_random()
                coord_sensors_gene_format = self.convert_polar2d_to_gene_seq(coord_sensors)
                population.append(coord_sensors_gene_format)
        elif self.initial_placement == "deterministic":
            for _ in range(self.population_size):
                coord_sensors = self.get_sensor_coords_determinstic()
                coord_sensors_gene_format = self.convert_polar2d_to_gene_seq(coord_sensors)
                population.append(coord_sensors_gene_format)
        elif self.initial_placement == "min":
            for _ in range(self.population_size):
                coord_sensors = self.get_sensor_coord_minimum_cost()
                coord_sensors_gene_format = self.convert_polar2d_to_gene_seq(coord_sensors)
                population.append(coord_sensors_gene_format)

        # print(population)

        return np.array(population)

    def evaluate_fitness_population(self):

        fitness_lookup = []

        for chromosome in self.population:
            fitness = self.evaluate_fitness_single(chromosome)
            fitness_lookup.append(fitness)

        return fitness_lookup

    def evaluate_fitness_single(self, chromosome):
        # chromosome = ["00","01","11", etc]

        fitness = 0

        for sensor in chromosome:
            st = self.get_sensor_type_in_int(sensor)
            if st != 0:
                scost = self.cost_sensors[st - 1]
                fitness += scost

        return fitness

    def selection_tournament(self, fitness_population, n=3):
        # select random chromosome first
        selected_id = np.random.randint(0, self.population_size)

        for random_id in np.random.randint(0, self.population_size, n - 1):
            if fitness_population[random_id] < fitness_population[selected_id]:
                selected_id = random_id

        return self.population[selected_id]

    def crossover(self, parent1, parent2):

        child1 = deepcopy(parent1)
        child2 = deepcopy(parent2)

        for _ in range(5):
            if self.crossover_rate > np.random.rand():
                # perform crossover
                # TODO: reminder : len(parent1)/4
                # num_crossover_points = np.random.randint(0, len(parent1)/4)
                num_crossover_points = self.point_crossover
                # print("num_crossover_points: ", num_crossover_points)

                if num_crossover_points == 1:
                    crossover_point = np.random.randint(0, len(parent1))

                    child1_part = child1[crossover_point:]
                    child2_part = child2[crossover_point:]

                    child1[crossover_point:] = child2_part
                    child2[crossover_point:] = child1_part

                else:
                    crossover_points = sorted(np.random.randint(0, len(parent1), [num_crossover_points]))
                    # print(crossover_points)

                    for i in range(0, len(crossover_points), 2):
                        pnt1 = crossover_points[i]
                        pnt2 = crossover_points[i + 1]

                        child1_part = child1[pnt1:pnt2]
                        child2_part = child2[pnt1:pnt2]

                        child1[pnt1:pnt2] = child2_part
                        child2[pnt1:pnt2] = child1_part

                # check k coverage constraint
                if self.check_coverage_single(child1) and self.check_coverage_single(child2):
                    break
                elif self.check_coverage_single(child1) and not self.check_coverage_single(child2):
                    child2 = deepcopy(parent2)
                elif not self.check_coverage_single(child1) and self.check_coverage_single(child2):
                    child1 = deepcopy(parent1)

        return [child1, child2]

    def get_random_mutation(self):
        # TODO: make more dynamic
        # possible_states = [i for i in self.types_sensors]
        # possible_states = [0] + possible_states
        possible_states = [0, 1, 2, 3]
        random_id = np.random.randint(0, len(possible_states))

        return self.get_sensor_type_in_str(possible_states[random_id])
        # return "00"

    def mutation(self, chromosome):

        mutated_chrom = deepcopy(chromosome)

        for _ in range(5):

            if self.mutation_rate > np.random.rand():
                # num_mutations = np.random.randint(0, len(chromosome))
                num_mutations = self.point_mutation

                id_mutations = np.random.randint(0, len(chromosome), [num_mutations])

                """
                futures = []
                mutations = []

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(self.get_random_mutation) for _ in range(num_mutations)]

                mutations = [f.result() for f in futures]
                """

                # print("Mutation IDs selected: ", id_mutations)

                for index, idm in enumerate(id_mutations):
                    mutated_chrom[idm] = self.get_random_mutation()
                    # mutated_chrom[idm] = mutations[index]

                if self.check_coverage_single(chromosome):
                    break
                else:
                    mutated_chrom = deepcopy(chromosome)

        return mutated_chrom

    def optimize(self, filename=None):
        # run for set num_generations
        # for each generation
        # calculate fitness of entire population
        # select 2 individuals of each generations
        # perform crossover of two individuals
        # perform mutation of random solutions in the population
        # ensure k-coverage is maintained in each solution, if not maintained loop in this generaiton again

        no_average_change = 0
        previous_generation_average = 0

        generation = 1
        while (generation != self.num_generations and no_average_change != 50):

            current_generation_average = 0

            startg = time.time()

            new_population = deepcopy(self.population)

            # evaluate fitness of entire population
            start = time.time()

            fitness_population = self.evaluate_fitness_population()
            print(fitness_population)

            self.population_fitness_history.append(fitness_population)

            end = time.time()
            print("Time spend in fitness eval: {}".format(end - start))

            for id, fitness in enumerate(fitness_population):
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_chromosome = self.population[id]
            print("Generation {} : Best Fitness: {} ALLTB : {}".format(generation, min(fitness_population),
                                                                       self.best_fitness))

            min_fitness_in_this_gen = min(fitness_population)
            index_min_fitness_in_this_gen = fitness_population.index(min_fitness_in_this_gen)
            self.best_solution_history.append(self.population[index_min_fitness_in_this_gen])

            start = time.time()
            # select individuals from population
            selected_chromosomes = [self.selection_tournament(fitness_population) for _ in range(self.population_size)]
            end = time.time()
            print("Time spent in selection: {}".format(end - start))

            time_crossover = []
            time_mutation = []

            children = []
            for i in range(0, len(selected_chromosomes), 2):
                parent1 = selected_chromosomes[i]
                parent2 = selected_chromosomes[i + 1]

                start = time.time()
                # perform crossover
                childs = self.crossover(parent1, parent2)
                end = time.time()
                time_crossover.append(end - start)

                start = time.time()
                # perform mutation of the children

                futures = []
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(self.mutation, childs[i]) for i in range(2)]

                child2 = futures[1].result()
                child1 = futures[0].result()

                # child1 = self.mutation(childs[0])
                # child2 = self.mutation(childs[1])

                end = time.time()
                time_mutation.append(end - start)

                # add mutated offspring to the new population
                children.append(child1)
                children.append(child2)

            # update new population
            self.population = children
            print("Avg. time in crossover: {}".format(np.average(time_crossover)))
            print("Avg. time in mutaion: {}".format(np.average(time_mutation)))

            endg = time.time()
            print("Time spent in Generation : {}".format(endg - startg))

            current_generation_average = np.average(fitness_population)

            generation += 1
            if current_generation_average == previous_generation_average:
                no_average_change += 1
            else:
                no_average_change == 0

            print("PGA: {} CGA: {} NAC: {}".format(previous_generation_average, current_generation_average,
                                                   no_average_change))

            previous_generation_average = current_generation_average

        if filename == None:
            self.save_experiment_results()
        else:
            self.save_experiment_results(filename)

    def test_grid_search(self, mode=1, mutation_grid=None, crossover_grid=None):
        """
        Mode 1: test mutation points
        Mode 2: test crossover points
        Mode 3: test combinations of mutation and crossover points
        """

        if mode == 1:
            assert mutation_grid != None, "Mutation grid cannot be empty for mode 1"

            for mutation_point in mutation_grid:
                self.re_init()
                self.point_mutation = mutation_point
                self.optimize()

        elif mode == 2:
            assert crossover_grid != None, "Crossover grid cannot be emptry for mode 2"

            for crossover_point in crossover_grid:
                self.re_init()
                self.point_crossover = crossover_point
                self.optimize()

        elif mode == 3:
            # this is grid search
            assert ((crossover_grid != None) and (mutation_grid != None)), "Crossover and mutation grid cannot be empty"

            """for i in range(len(mutation_grid)):
                for j in range(len(crossover_grid)):

                    self.re_init()
                    self.point_crossover = crossover_grid[j]
                    self.point_mutation = mutation_grid[i]

                    timestamp = time.time()

                    filename = "./cc_experiments/grid_{}_pm{}_pc{}_ps{}_ng{}_x{}_y{}.csv".format(
                timestamp, self.point_mutation, self.point_crossover, self.population_size, self.num_generations,
            self.area_surv_x, self.area_surv_y)

                    print("Experimenting Mutation Points : {} Crossover Points: {}".format(self.point_mutation, self.point_crossover))

                    self.optimize(filename)"""

            for i in range(len(mutation_grid)):
                for j in range(i):
                    if i == j:
                        break
                    else:
                        self.re_init()
                        self.point_crossover = crossover_grid[j]
                        self.point_mutation = mutation_grid[i]

                        timestamp = time.time()

                        filename = "./cc_experiments/grid_{}_pm{}_pc{}_ps{}_ng{}_x{}_y{}.csv".format(
                            timestamp, self.point_mutation, self.point_crossover, self.population_size,
                            self.num_generations,
                            self.area_surv_x, self.area_surv_y)

                        print("Experimenting Mutation Points : {} Crossover Points: {}".format(self.point_mutation,
                                                                                               self.point_crossover))

                        self.optimize(filename)

        elif mode == 4:
            # this is increasing mutation rate and decreasing crossover rate

            # run for set num_generations
            # for each generation
            # calculate fitness of entire population
            # select 2 individuals of each generations
            # perform crossover of two individuals
            # perform mutation of random solutions in the population
            # ensure k-coverage is maintained in each solution, if not maintained loop in this generaiton again

            self.point_crossover = 512
            self.point_mutation = 2

            no_average_change = 0
            previous_generation_average = 0

            generation = 1
            while (generation != self.num_generations and no_average_change != 50):

                current_generation_average = 0

                mut_rate, cross_rate = self.get_mut_cross_rate_for_gen_ilmdhc(generation)
                num_chrom_to_mutate = round(self.population_size * mut_rate)
                num_chrom_to_cross = round(self.population_size * cross_rate)

                print("For Generation {} : Mutation rate {} Crossover rate {}".format(generation, mut_rate, cross_rate))

                if num_chrom_to_cross % 2 != 0:
                    num_chrom_to_cross = num_chrom_to_cross + 1

                # num_remaining_chrom = self.population_size - num_chrom_to_cross

                # selected_chromosomes = [self.selection_tournament(fitness_population) for _ in range(num_chrom_to_cross)]

                startg = time.time()

                new_population = deepcopy(self.population)

                # evaluate fitness of entire population
                start = time.time()

                fitness_population = self.evaluate_fitness_population()
                print(fitness_population)

                self.population_fitness_history.append(fitness_population)

                end = time.time()
                print("Time spend in fitness eval: {}".format(end - start))

                for id, fitness in enumerate(fitness_population):
                    if fitness < self.best_fitness:
                        self.best_fitness = fitness
                        self.best_chromosome = self.population[id]
                print("Generation {} : Best Fitness: {} ALLTB : {}".format(generation, min(fitness_population),
                                                                           self.best_fitness))

                min_fitness_in_this_gen = min(fitness_population)
                index_min_fitness_in_this_gen = fitness_population.index(min_fitness_in_this_gen)
                self.best_solution_history.append(self.population[index_min_fitness_in_this_gen])

                start = time.time()
                # select individuals from population
                selected_chromosomes = [self.selection_tournament(fitness_population) for _ in
                                        range(self.population_size)]
                end = time.time()
                print("Time spent in selection: {}".format(end - start))

                time_crossover = []
                time_mutation = []

                children = deepcopy(self.population)
                for i in range(0, num_chrom_to_cross, 2):
                    parent1 = selected_chromosomes[i]
                    parent2 = selected_chromosomes[i + 1]

                    start = time.time()
                    # perform crossover
                    childs = self.crossover(parent1, parent2)
                    end = time.time()
                    time_crossover.append(end - start)

                    children[i] = deepcopy(childs[0])
                    children[i + 1] = deepcopy(childs[1])

                id_mutations = np.random.randint(0, self.population_size, [num_chrom_to_mutate])

                start = time.time()
                # perform mutation of the children

                futures = []
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(self.mutation, children[i]) for i in id_mutations]

                for future, i in zip(futures, id_mutations):
                    result = future.result()
                    children[i] = deepcopy(result)

                # child2 = futures[1].result()
                # child1 = futures[0].result()

                # child1 = self.mutation(childs[0])
                # child2 = self.mutation(childs[1])

                end = time.time()
                time_mutation.append(end - start)

                # update new population
                self.population = deepcopy(children)
                print("Avg. time in crossover: {}".format(np.average(time_crossover)))
                print("Avg. time in mutaion: {}".format(np.average(time_mutation)))

                endg = time.time()
                print("Time spent in Generation : {}".format(endg - startg))

                current_generation_average = np.average(fitness_population)

                generation += 1
                if current_generation_average == previous_generation_average:
                    no_average_change += 1
                else:
                    no_average_change == 0

                previous_generation_average = current_generation_average

            timestamp = time.time()
            filename = "./cc_experiments/ilmdhc_{}_pm{}_pc{}_ps{}_ng{}_x{}_y{}.csv".format(
                timestamp, self.point_mutation, self.point_crossover, self.population_size, self.num_generations,
                self.area_surv_x, self.area_surv_y)
            self.save_experiment_results(filename)


        elif mode == 5:
            # this is increasing corssover rate and decreasing mutation rate

            # run for set num_generations
            # for each generation
            # calculate fitness of entire population
            # select 2 individuals of each generations
            # perform crossover of two individuals
            # perform mutation of random solutions in the population
            # ensure k-coverage is maintained in each solution, if not maintained loop in this generaiton again

            self.point_crossover = 16
            self.point_mutation = 2

            no_average_change = 0
            previous_generation_average = 0

            generation = 1
            while (generation != self.num_generations and no_average_change != 50):

                current_generation_average = 0

                mut_rate, cross_rate = self.get_mut_cross_rate_for_gen_dhmilc(generation)
                num_chrom_to_mutate = round(self.population_size * mut_rate)
                num_chrom_to_cross = round(self.population_size * cross_rate)

                print("For Generation {} : Mutation rate {} Crossover rate {}".format(generation, mut_rate, cross_rate))

                if num_chrom_to_cross % 2 != 0:
                    num_chrom_to_cross = num_chrom_to_cross + 1

                # num_remaining_chrom = self.population_size - num_chrom_to_cross

                # selected_chromosomes = [self.selection_tournament(fitness_population) for _ in range(num_chrom_to_cross)]

                startg = time.time()

                new_population = deepcopy(self.population)

                # evaluate fitness of entire population
                start = time.time()

                fitness_population = self.evaluate_fitness_population()
                print(fitness_population)

                self.population_fitness_history.append(fitness_population)

                end = time.time()
                print("Time spend in fitness eval: {}".format(end - start))

                for id, fitness in enumerate(fitness_population):
                    if fitness < self.best_fitness:
                        self.best_fitness = fitness
                        self.best_chromosome = self.population[id]
                print("Generation {} : Best Fitness: {} ALLTB : {}".format(generation, min(fitness_population),
                                                                           self.best_fitness))

                min_fitness_in_this_gen = min(fitness_population)
                index_min_fitness_in_this_gen = fitness_population.index(min_fitness_in_this_gen)
                self.best_solution_history.append(self.population[index_min_fitness_in_this_gen])

                start = time.time()
                # select individuals from population
                selected_chromosomes = [self.selection_tournament(fitness_population) for _ in
                                        range(self.population_size)]
                end = time.time()
                print("Time spent in selection: {}".format(end - start))

                time_crossover = []
                time_mutation = []

                children = deepcopy(self.population)
                for i in range(0, num_chrom_to_cross, 2):
                    parent1 = selected_chromosomes[i]
                    parent2 = selected_chromosomes[i + 1]

                    start = time.time()
                    # perform crossover
                    childs = self.crossover(parent1, parent2)
                    end = time.time()
                    time_crossover.append(end - start)

                    children[i] = deepcopy(childs[0])
                    children[i + 1] = deepcopy(childs[1])

                id_mutations = np.random.randint(0, self.population_size, [num_chrom_to_mutate])

                start = time.time()
                # perform mutation of the children

                futures = []
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(self.mutation, children[i]) for i in id_mutations]

                for future, i in zip(futures, id_mutations):
                    result = future.result()
                    children[i] = deepcopy(result)

                # child2 = futures[1].result()
                # child1 = futures[0].result()

                # child1 = self.mutation(childs[0])
                # child2 = self.mutation(childs[1])

                end = time.time()
                time_mutation.append(end - start)

                # update new population
                self.population = deepcopy(children)
                print("Avg. time in crossover: {}".format(np.average(time_crossover)))
                print("Avg. time in mutaion: {}".format(np.average(time_mutation)))

                endg = time.time()
                print("Time spent in Generation : {}".format(endg - startg))

                current_generation_average = np.average(fitness_population)

                generation += 1
                if current_generation_average == previous_generation_average:
                    no_average_change += 1
                else:
                    no_average_change == 0

                previous_generation_average = current_generation_average

            timestamp = time.time()
            filename = "./cc_experiments/dhmilc_{}_pm{}_pc{}_ps{}_ng{}_x{}_y{}.csv".format(
                timestamp, self.point_mutation, self.point_crossover, self.population_size, self.num_generations,
                self.area_surv_x, self.area_surv_y)
            self.save_experiment_results(filename)

    def get_mut_cross_rate_for_gen_ilmdhc(self, curr_gen_num):

        mr = curr_gen_num / self.num_generations
        cr = 1 - mr

        return (mr, cr)

    def get_mut_cross_rate_for_gen_dhmilc(self, curr_gen_num):

        cr = curr_gen_num / self.num_generations
        mr = 1 - cr

        return (mr, cr)


if __name__ == '__main__':

    mutation_points = [2 ** i for i in range(11)]
    crossover_points = [2 ** i for i in range(11)]

    experiment = GeneticAlgorithm()

    experiment.test_grid_search(mode=5)

    """
    fitness_initial = testobj.evaluate_fitness_population()
    print(fitness_initial)
    print("MAX: ", max(fitness_initial))
    print("MIN: ", min(fitness_initial))
    """
    for name in dir():
        if not name.startswith('_'):
            del globals()[name]