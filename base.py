########################################################################################################################
# Dual Sourcing Inventory Routing Problem with route-dependent-lead-times in a rolling horizon framework
# Adaptive Variable Neighborhood Search Algorithm
# Python>=3.8
# Author: Weibo Zheng
########################################################################################################################

import numpy as np
import math
import re
import random
import copy


##################################################################################

# ------------------------------------ Class --------------------------------------------


class Problem:
    def __init__(self, name, locations, initial_inventory_level):
        self.instance_name = name
        self.C = 3  # planning cycle
        self.T = 24  # planning period
        self.H = copy.deepcopy(self.T)  # inventory planning period
        self.h = 0.05  # unit holding cost
        self.s = 2  # unit shortage cost
        self.dm = 20  # demand mean
        self.sigma = 1  # demand standard deviation
        self.dh = 10  # demand maximum deviation
        self.kv = 30  # speed of land vehicle
        self.hv = 130  # speed of aerial vehicle
        self.f_v = 50  # vehicle unit using cost
        self.f_h = 250  # helicopter unit using cost
        self.fix_v = 500  # vehicle fixed cost
        self.fix_h = 10000  # helicopter fixed cost
        self.segments_capacity = 50  # segment capability
        self.temperature = 30000
        self.cooling_rate = 0.995
        self.land_capacity = 5000
        self.aerial_capacity = 2000
        self.reaction_factor = 0.1
        self.score_factor_1 = 10
        self.score_factor_2 = 5
        self.score_factor_3 = 2
        self.locations = locations
        self.initial_inventory_level = [i * 20 for i in initial_inventory_level]
        self.N = len(self.locations)
        self.K = self.N // 15 + 1  # land vehicle number
        self.A = self.N // 30 + 1  # aerial vehicle number
        self.best_replenishment_time = [0]
        self.locations = np.array(self.locations)
        self.distance_timecost_matrix_v = np.zeros((self.N, self.N))
        self.distance_timecost_matrix_h = np.zeros((self.N, self.N))
        for from_index in range(self.N):
            for to_index in range(self.N):
                self.distance_timecost_matrix_v[from_index][to_index] = \
                    calculate_distance(self.locations[from_index], self.locations[to_index]) / self.kv
                self.distance_timecost_matrix_h[from_index][to_index] = \
                    calculate_distance(self.locations[from_index], self.locations[to_index]) / self.hv
        self.inventory_cost_matrix_zero = 99999999999 * np.ones(self.N)
        self.shelter_order_size_matrix_zero = np.zeros(self.N)
        self.inventory_cost_matrix_one = 99999999999 * np.ones((self.N, self.T + 1))
        self.shelter_order_size_matrix_one = np.zeros((self.N, self.T + 1))
        self.inventory_cost_matrix_two = 99999999999 * np.ones((self.N, self.T + 1, self.T + 1))
        self.shelter_order_size_matrix_two_q_f = np.zeros((self.N, self.T + 1, self.T + 1))
        self.shelter_order_size_matrix_two_q_s = np.zeros((self.N, self.T + 1, self.T + 1))


class Solution:
    def __init__(self, problem):
        self.route_v = [[] for i in range(problem.K)]
        self.route_h = [[] for i in range(problem.A)]
        self.pool_v = []
        self.pool_h = []
        self.obj = 9999999999999999
        self.order_size_total = np.zeros(problem.N)


class Shaking_operators:
    def __init__(self, shaking_operators_list):
        self.shaking_operator_score = {}
        self.shaking_operator_weight = {}
        self.shaking_operator_times = {}
        for i in shaking_operators_list:
            self.shaking_operator_score[i] = 0
            self.shaking_operator_weight[i] = 1
            self.shaking_operator_times[i] = 0

    def select(self, shaking_operators_list):
        selected_operator = 'Null'
        total_weight = sum(self.shaking_operator_weight.values())
        probabilities = {}
        for i in shaking_operators_list:
            probabilities[i] = self.shaking_operator_weight[i] / total_weight
        total_prob = sum(probabilities.values())
        selector = random.random() * total_prob
        cumulative_probability = 0
        for i in shaking_operators_list:
            cumulative_probability += probabilities[i]
            if cumulative_probability >= selector:
                selected_operator = i
                break
        return selected_operator

    def update_weight(self, shaking_operators_list, problem):
        for i in shaking_operators_list:
            if self.shaking_operator_times[i] >= 1:
                self.shaking_operator_weight[i] = \
                    (1 - problem.reaction_factor) * self.shaking_operator_weight[i] + \
                    problem.reaction_factor * self.shaking_operator_score[i] / self.shaking_operator_times[i]
                if self.shaking_operator_weight[i] < 0.1:
                    self.shaking_operator_weight[i] = 0.1
        # reset score and times
        for i in shaking_operators_list:
            self.shaking_operator_score[i] = 0
            self.shaking_operator_times[i] = 0
        return


class Record:
    def __init__(self, shaking_operators_list):
        self.initial_generator_time_cost = 0
        self.total_time_cost = 0
        self.shaking_operator_weights_record = {}
        self.shaking_operator_times_record = {}
        self.shaking_operator_score_record = {}
        self.shaking_operator_max_best_times ={}
        self.shaking_operator_max_usage_times = {}
        for i in shaking_operators_list:
            self.shaking_operator_weights_record[i] = []
            self.shaking_operator_times_record[i] = []
            self.shaking_operator_score_record[i] = []
            self.shaking_operator_max_usage_times[i] = 0
            self.shaking_operator_max_best_times[i] = 0
        self.best_solution_update_iteration_num = []
        self.best_solution_update_times = 0


# -------------------------- Function --------------------------------------

def calculate_distance(point0, point1):
    dx = point1[0] - point0[0]
    dy = point1[1] - point0[1]
    return math.sqrt(dx * dx + dy * dy)


def generate_problem(instance):
    locations = []
    initial_inventoy_level = []
    initial_inventoy_level_out = []
    with open('test_set/' + instance, 'r') as file_to_read:
        lines = file_to_read.readlines()
        flg_locations = False
        flg_demands = False
        for line in lines:
            if "NAME" in line:
                name = re.findall(r"NAME : A-(.+?)-k", line)
            if "NODE_COORD_SECTION" in line:
                flg_locations = True
            if "DEMAND_SECTION" in line:
                flg_locations = False
                flg_demands = True
            if "DEPOT_SECTION" in line:
                flg_demands = False
            if flg_locations:
                temp = line[:-1].split(' ')
                temp.pop(0)
                temp.pop(0)
                locations.append(temp)
            if flg_demands:
                temp = line[:-1].split(' ')
                temp.pop(0)
                temp.pop(-1)
                initial_inventoy_level.append(temp)
        locations.pop(0)
        initial_inventoy_level.pop(0)
        for i in range(len(locations)):
            for j in range(len(locations[i])):
                locations[i][j] = int(locations[i][j])
        for i in range(len(initial_inventoy_level)):
            for j in range(len(initial_inventoy_level[i])):
                initial_inventoy_level_out.append(int(initial_inventoy_level[i][j]))
    problem = Problem(name, locations, initial_inventoy_level_out)
    return problem


def calculate_objective(solution, problem):
    # -------------------decoding-----------------------
    shelter_arrival_time_v = np.zeros(problem.N)
    shelter_arrival_time_h = np.zeros(problem.N)
    shelters_visit_times = np.zeros(problem.N)
    shelters_order_size_total = np.zeros(problem.N)
    shelters_order_size_q_f = np.zeros(problem.N)
    shelters_order_size_q_s = np.zeros(problem.N)
    from_v = 0
    from_h = 0
    inventory_cost = 0
    route_cost = 0
    for i in range(problem.K):
        if solution.route_v[i]:
            route_cost += problem.fix_v
            for j in range(len(solution.route_v[i])):
                shelter_arrival_time_v[solution.route_v[i][j]] = shelter_arrival_time_v[from_v] + \
                                                                 problem.distance_timecost_matrix_v[
                                                                     from_v, solution.route_v[i][j]]
                shelters_visit_times[solution.route_v[i][j]] += 1
                from_v = solution.route_v[i][j]
                finish_time_v = shelter_arrival_time_v[from_v] + problem.distance_timecost_matrix_v[from_v, 0]
                route_cost += problem.f_v * finish_time_v
    for i in range(problem.A):
        if solution.route_h[i]:
            route_cost += problem.fix_h
            for j in range(len(solution.route_h[i])):
                shelter_arrival_time_h[solution.route_h[i][j]] = shelter_arrival_time_h[from_h] + \
                                                                 problem.distance_timecost_matrix_h[
                                                                     from_h, solution.route_h[i][j]]
                shelters_visit_times[solution.route_h[i][j]] += 1
                from_h = solution.route_h[i][j]
                finish_time_h = shelter_arrival_time_h[from_h] + problem.distance_timecost_matrix_h[from_h, 0]
                route_cost += problem.f_h * finish_time_h
    # --------------------inventory cost----------------------------
    for i in range(1, problem.N):
        if shelters_visit_times[i] == 0:
            inventory_cost += problem.inventory_cost_matrix_zero[i]
        elif shelters_visit_times[i] == 1:
            inventory_cost += problem.inventory_cost_matrix_one[
                i, math.ceil(min(max(shelter_arrival_time_v[i], shelter_arrival_time_h[i]), problem.T))]
            shelters_order_size_total[i] = problem.shelter_order_size_matrix_one[
                i, math.ceil(min(max(shelter_arrival_time_v[i], shelter_arrival_time_h[i]), problem.T))]
        elif shelters_visit_times[i] == 2:
            inventory_cost += problem.inventory_cost_matrix_two[
                i, math.ceil(min(shelter_arrival_time_v[i], shelter_arrival_time_h[i], problem.T)),
                math.ceil(min(max(shelter_arrival_time_v[i], shelter_arrival_time_h[i]), problem.T))]
            shelters_order_size_q_f[i] = problem.shelter_order_size_matrix_two_q_f[
                i, math.ceil(min(shelter_arrival_time_v[i], shelter_arrival_time_h[i], problem.T)),
                math.ceil(min(max(shelter_arrival_time_v[i], shelter_arrival_time_h[i]), problem.T))]
            shelters_order_size_q_s[i] = problem.shelter_order_size_matrix_two_q_s[
                i, math.ceil(min(shelter_arrival_time_v[i], shelter_arrival_time_h[i], problem.T)),
                math.ceil(min(max(shelter_arrival_time_v[i], shelter_arrival_time_h[i]), problem.T))]
            shelters_order_size_total[i] = shelters_order_size_q_f[i] + shelters_order_size_q_s[i]
        else:
            print('Invalid shelter visit times for shelter No. ', i)
    solution.order_size_total = shelters_order_size_total
    # --------------------penalty-------------------------------------
    penalty = 0
    for i in range(problem.K):
        order_size = 0
        for j in range(len(solution.route_v[i])):
            if shelters_visit_times[solution.route_v[i][j]] == 0:
                order_size += 0
            elif shelters_visit_times[solution.route_v[i][j]] == 1:
                order_size += shelters_order_size_total[solution.route_v[i][j]]
            elif shelters_visit_times[solution.route_v[i][j]] == 2:
                if shelter_arrival_time_v[solution.route_v[i][j]] < shelter_arrival_time_h[solution.route_v[i][j]]:
                    order_size += shelters_order_size_q_f[solution.route_v[i][j]]
                else:
                    order_size += shelters_order_size_q_s[solution.route_v[i][j]]
            else:
                print('Error: wrong visit times')
        if order_size > problem.land_capacity:
            penalty += 100000000
    for i in range(problem.A):
        order_size = 0
        for j in range(len(solution.route_h[i])):
            if shelters_visit_times[solution.route_h[i][j]] == 0:
                order_size += 0
            elif shelters_visit_times[solution.route_h[i][j]] == 1:
                order_size += shelters_order_size_total[solution.route_h[i][j]]
            elif shelters_visit_times[solution.route_h[i][j]] == 2:
                if shelter_arrival_time_v[solution.route_h[i][j]] < shelter_arrival_time_h[solution.route_h[i][j]]:
                    order_size += shelters_order_size_q_s[solution.route_h[i][j]]
                else:
                    order_size += shelters_order_size_q_f[solution.route_h[i][j]]
            else:
                print('Error: wrong visit times')
        if order_size > problem.aerial_capacity:
            penalty += 100000000
    total_cost = route_cost + inventory_cost + penalty
    return total_cost, shelters_order_size_total


def do_shaking(operator_name, solution_in, problem):
    solution = copy.deepcopy(solution_in)
    best_mode = 'Null'
    best_route = []
    best_num = -1
    best_obj = solution.obj
    best_order_size = solution.order_size_total
    best_position = -1
    if operator_name == 'Delete':
        best_route = []
        best_operation_cost = 9999999999999999
        if random.random() < 0.5: # -------------------测试
            for i in range(problem.K):
                temp = solution.route_v[i]
                solution.route_v[i] = []
                new_obj, new_order_size = calculate_objective(solution, problem)
                operation_cost = new_obj - solution.obj
                solution.route_v[i] = temp
                if operation_cost < best_operation_cost:
                    best_operation_cost = operation_cost
                    best_route = temp
                    best_mode = 'vehicle'
                    best_obj = new_obj
                    best_order_size = new_order_size
                    best_num = i
        else:
            for i in range(problem.A):
                temp = solution.route_h[i]
                solution.route_h[i] = []
                new_obj, new_order_size = calculate_objective(solution, problem)
                operation_cost = new_obj - solution.obj
                solution.route_h[i] = temp
                if operation_cost < best_operation_cost:
                    best_operation_cost = operation_cost
                    best_route = temp
                    best_mode = 'helicopter'
                    best_obj = new_obj
                    best_order_size = new_order_size
                    best_num = i
        if best_route:
            if best_mode == 'vehicle':
                solution.pool_v.extend(solution.route_v[best_num])
                solution.route_v[best_num] = []
                solution.obj = best_obj
                solution.order_size_total = best_order_size
            elif best_mode == 'helicopter':
                solution.pool_h.extend(solution.route_h[best_num])
                solution.route_h[best_num] = []
                solution.obj = best_obj
                solution.order_size_total = best_order_size
            else:
                print('Delete operator error!')
            if best_mode == 'vehicle':
                return 'helicopter', range(problem.A), solution
            else:
                return 'vehicle', range(problem.K), solution
        else:
            return 'Null', [], solution
    elif operator_name == 'Removal':
        best_location = -1
        best_operation_cost = 9999999999999999
        if random.random() < 0.5: # -------------------测试
            for i in range(problem.K):
                if solution.route_v[i]:
                    for j in range(len(solution.route_v[i])):
                        temp = solution.route_v[i].pop(j)
                        new_obj, new_order_size = calculate_objective(solution, problem)
                        operation_cost = new_obj - solution.obj
                        solution.route_v[i].insert(j, temp)
                        if operation_cost < best_operation_cost:
                            best_operation_cost = operation_cost
                            best_mode = 'vehicle'
                            best_obj = new_obj
                            best_order_size = new_order_size
                            best_route = i
                            best_location = j
        else:
            for i in range(problem.A):
                if solution.route_h[i]:
                    for j in range(len(solution.route_h[i])):
                        temp = solution.route_h[i].pop(j)
                        new_obj, new_order_size = calculate_objective(solution, problem)
                        operation_cost = new_obj - solution.obj
                        solution.route_h[i].insert(j, temp)
                        if operation_cost < best_operation_cost:
                            best_operation_cost = operation_cost
                            best_mode = 'helicopter'
                            best_obj = new_obj
                            best_order_size = new_order_size
                            best_route = i
                            best_location = j
        if best_location != -1:
            if best_mode == 'vehicle':
                temp = solution.route_v[best_route].pop(best_location)
                solution.pool_v.append(temp)
                solution.obj = best_obj
                solution.order_size_total = best_order_size
            elif best_mode == 'helicopter':
                temp = solution.route_h[best_route].pop(best_location)
                solution.pool_h.append(temp)
                solution.obj = best_obj
                solution.order_size_total = best_order_size
            else:
                print('Removal operator error!')
            #            return best_mode, [best_route], solution
            if best_mode == 'vehicle':
                return 'helicopter', range(problem.A), solution
            else:
                return 'vehicle', range(problem.K), solution
        else:
            #print('Removal operator error!')
            return 'Null', [], solution
    elif operator_name == 'Insertion':
        best_location = -1
        best_operation_cost = 9999999999999999
        if random.random() < 0.5: # -------------------测试
            if solution.pool_v:
                for i in range(len(solution.pool_v)):
                    for j in range(problem.K):
                        for k in range(len(solution.route_v[j]) + 1):
                            solution.route_v[j].insert(k, solution.pool_v[i])
                            new_obj, new_order_size = calculate_objective(solution, problem)
                            operation_cost = new_obj - solution.obj
                            solution.route_v[j].pop(k)
                            if operation_cost < best_operation_cost:
                                best_operation_cost = operation_cost
                                best_mode = 'vehicle'
                                best_obj = new_obj
                                best_order_size = new_order_size
                                best_location = i
                                best_route = j
                                best_position = k
        else:
            if solution.pool_h:
                for i in range(len(solution.pool_h)):
                    for j in range(problem.A):
                        for k in range(len(solution.route_h[j]) + 1):
                            solution.route_h[j].insert(k, solution.pool_h[i])
                            new_obj, new_order_size = calculate_objective(solution, problem)
                            operation_cost = new_obj - solution.obj
                            solution.route_h[j].pop(k)
                            if operation_cost < best_operation_cost:
                                best_operation_cost = operation_cost
                                best_mode = 'helicopter'
                                best_obj = new_obj
                                best_order_size = new_order_size
                                best_location = i
                                best_route = j
                                best_position = k
        if best_location != -1:
            if best_mode == 'vehicle':
                temp = solution.pool_v.pop(best_location)
                solution.route_v[best_route].insert(best_position, temp)
                solution.obj = best_obj
                solution.order_size_total = best_order_size
            elif best_mode == 'helicopter':
                temp = solution.pool_h.pop(best_location)
                solution.route_h[best_route].insert(best_position, temp)
                solution.obj = best_obj
                solution.order_size_total = best_order_size
            else:
                print('Insertion operator mode error!')
            #            return best_mode, [best_route], solution
            if best_mode == 'vehicle':
                return 'helicopter', range(problem.A), solution
            else:
                return 'vehicle', range(problem.K), solution
        else:
            #print('No best insertion target')
            return 'Null', [], solution
    elif operator_name == 'Exchange':
        if random.random() < 0.5:
            routes = random.sample(range(0, problem.K), 2)
            if len(solution.route_v[routes[0]]) == 0:
                segment_1 = []
            else:
                segment_1 = random.sample(range(0, len(solution.route_v[routes[0]]) + 1), 2)
                segment_1.sort()
            if len(solution.route_v[routes[1]]) == 0:
                segment_2 = []
            else:
                segment_2 = random.sample(range(0, len(solution.route_v[routes[1]]) + 1), 2)
                segment_2.sort()
            # -----------do exchange
            if segment_1 and segment_2 == []:
                solution.route_v[routes[1]] = solution.route_v[routes[0]][segment_1[0]:segment_1[1]]
                del solution.route_v[routes[0]][segment_1[0]:segment_1[1]]
            elif segment_2 and segment_1 == []:
                solution.route_v[routes[0]] = solution.route_v[routes[1]][segment_2[0]:segment_2[1]]
                del solution.route_v[routes[1]][segment_2[0]:segment_2[1]]
            elif segment_1 and segment_2:
                route1 = solution.route_v[routes[0]][0:segment_1[0]]
                route1.extend(solution.route_v[routes[1]][segment_2[0]:segment_2[1]])
                route1.extend(solution.route_v[routes[0]][segment_1[1]:len(solution.route_v[routes[0]])])
                route2 = solution.route_v[routes[1]][0:segment_2[0]]
                route2.extend(solution.route_v[routes[0]][segment_1[0]:segment_1[1]])
                route2.extend(solution.route_v[routes[1]][segment_2[1]:len(solution.route_v[routes[1]])])
                solution.route_v[routes[0]] = route1
                solution.route_v[routes[1]] = route2
            solution.obj, solution.order_size_total = calculate_objective(solution, problem)
            return 'vehicle', routes, solution
        else:
            routes = random.sample(range(0, problem.A), 2)
            if len(solution.route_h[routes[0]]) == 0:
                segment_1 = []
            else:
                segment_1 = random.sample(range(0, len(solution.route_h[routes[0]]) + 1), 2)
                segment_1.sort()
            if len(solution.route_h[routes[1]]) == 0:
                segment_2 = []
            else:
                segment_2 = random.sample(range(0, len(solution.route_h[routes[1]]) + 1), 2)
                segment_2.sort()
                # -----------do exchange
            if segment_1 and segment_2 == []:
                solution.route_h[routes[1]] = solution.route_h[routes[0]][segment_1[0]:segment_1[1]]
                del solution.route_h[routes[0]][segment_1[0]:segment_1[1]]
            elif segment_2 and segment_1 == []:
                solution.route_h[routes[0]] = solution.route_h[routes[1]][segment_2[0]:segment_2[1]]
                del solution.route_h[routes[1]][segment_2[0]:segment_2[1]]
            elif segment_1 and segment_2:
                route1 = solution.route_h[routes[0]][0:segment_1[0]]
                route1.extend(solution.route_h[routes[1]][segment_2[0]:segment_2[1]])
                route1.extend(solution.route_h[routes[0]][segment_1[1]:len(solution.route_h[routes[0]])])
                route2 = solution.route_h[routes[1]][0:segment_2[0]]
                route2.extend(solution.route_h[routes[0]][segment_1[0]:segment_1[1]])
                route2.extend(solution.route_h[routes[1]][segment_2[1]:len(solution.route_h[routes[1]])])
                solution.route_h[routes[0]] = route1
                solution.route_h[routes[1]] = route2
            solution.obj, solution.order_size_total = calculate_objective(solution, problem)
            return 'helicopter', routes, solution
    elif operator_name == 'Swap':
        if random.random() < 0.5:
            routes = random.sample(range(0, problem.K), 2)
            if len(solution.route_v[routes[0]]) == 0:
                segment_1 = []
            else:
                segment_1 = random.sample(range(0, len(solution.route_v[routes[0]])), 1)
            if len(solution.route_v[routes[1]]) == 0:
                segment_2 = []
            else:
                segment_2 = random.sample(range(0, len(solution.route_v[routes[1]])), 1)
            # -----------do exchange
            if segment_1 == [] and segment_2:
                solution.route_v[routes[0]] = solution.route_v[routes[1]][segment_2[0]:len(solution.route_v[routes[1]])]
                del solution.route_v[routes[1]][segment_2[0]:len(solution.route_v[routes[1]])]
            elif segment_2 == [] and segment_1:
                solution.route_v[routes[1]] = solution.route_v[routes[0]][segment_1[0]:len(solution.route_v[routes[1]])]
                del solution.route_v[routes[0]][segment_1[0]:len(solution.route_v[routes[1]])]
            elif segment_1 and segment_2:
                route1 = solution.route_v[routes[0]][0:segment_1[0]]
                route1.extend(solution.route_v[routes[1]][segment_2[0]:len(solution.route_v[routes[1]])])
                route2 = solution.route_v[routes[1]][0:segment_2[0]]
                route2.extend(solution.route_v[routes[0]][segment_1[0]:len(solution.route_v[routes[0]])])
                solution.route_v[routes[0]] = route1
                solution.route_v[routes[1]] = route2
            solution.obj, solution.order_size_total = calculate_objective(solution, problem)
            return 'vehicle', routes, solution
        else:
            routes = random.sample(range(0, problem.A), 2)
            if len(solution.route_h[routes[0]]) == 0:
                segment_1 = []
            else:
                segment_1 = random.sample(range(0, len(solution.route_h[routes[0]])), 1)
            if len(solution.route_h[routes[1]]) == 0:
                segment_2 = []
            else:
                segment_2 = random.sample(range(0, len(solution.route_h[routes[1]])), 1)
            # -----------do swap
            if segment_1 == [] and segment_2:
                solution.route_h[routes[0]] = solution.route_h[routes[1]][segment_2[0]:len(solution.route_h[routes[1]])]
                del solution.route_h[routes[1]][segment_2[0]:len(solution.route_h[routes[1]])]
            elif segment_2 == [] and segment_1:
                solution.route_h[routes[1]] = solution.route_h[routes[0]][segment_1[0]:len(solution.route_h[routes[1]])]
                del solution.route_h[routes[0]][segment_1[0]:len(solution.route_h[routes[1]])]
            elif segment_1 and segment_2:
                route1 = solution.route_h[routes[0]][0:segment_1[0]]
                route1.extend(solution.route_h[routes[1]][segment_2[0]:len(solution.route_h[routes[1]])])
                route2 = solution.route_h[routes[1]][0:segment_2[0]]
                route2.extend(solution.route_h[routes[0]][segment_1[0]:len(solution.route_h[routes[0]])])
                solution.route_h[routes[0]] = route1
                solution.route_h[routes[1]] = route2
            solution.obj, solution.order_size_total = calculate_objective(solution, problem)
            return 'helicopter', routes, solution
    elif operator_name == 'Relocate':
        if random.random() < 0.5:
            routes = random.sample(range(0, problem.K), 2)
            if len(solution.route_v[routes[0]]) == 0:
                target = -1
            else:
                target = random.randint(0, len(solution.route_v[routes[0]]) - 1)

            position = random.randint(0, len(solution.route_v[routes[1]]))
            # -----------do relocate
            if target != -1:
                temp = solution.route_v[routes[0]].pop(target)
                solution.route_v[routes[1]].insert(position, temp)
            solution.obj, solution.order_size_total = calculate_objective(solution, problem)
            return 'vehicle', routes, solution
        else:
            routes = random.sample(range(0, problem.A), 2)
            if len(solution.route_h[routes[0]]) == 0:
                target = -1
            else:
                target = random.randint(0, len(solution.route_h[routes[0]]) - 1)

            position = random.randint(0, len(solution.route_h[routes[1]]))
            # -----------do relocate
            if target != -1:
                temp = solution.route_h[routes[0]].pop(target)
                solution.route_h[routes[1]].insert(position, temp)
            solution.obj, solution.order_size_total = calculate_objective(solution, problem)
            return 'helicopter', routes, solution
    else:
        print('error: Invalid Shaking Operator Name')
    return


def do_VND(mode, routes, solution_in, problem):
    solution = copy.deepcopy(solution_in)
    if routes != []:
        for route_num in range(len(routes)):
            if mode == 'vehicle':
                route_len = len(solution.route_v[routes[route_num]])
                if route_len >= 2:
                    best_obj = solution.obj
                    current_obj = solution.obj
                    new_solution = copy.deepcopy(solution)
                    current_route = solution.route_v[routes[route_num]]
                    best_route = solution.route_v[routes[route_num]]
                    current_order_size = solution.order_size_total
                    best_order_size = solution.order_size_total
                    l = 1
                    while l <= 3:
                        if l == 1:  # 2opt
                            route = best_route
                            for j in range(1, route_len):
                                for i in range(j):
                                    temp = route[i:j + 1]
                                    temp = temp[::-1]
                                    new_route = route[0:i]
                                    new_route.extend(temp)
                                    new_route.extend(route[j + 1:route_len])
                                    new_solution.route_v[routes[route_num]] = new_route
                                    new_obj, new_order_size = calculate_objective(new_solution, problem)
                                    if new_obj < current_obj:
                                        current_obj = new_obj
                                        current_order_size = new_order_size
                                        current_route = new_route
                            if current_obj < best_obj:
                                l = 1
                                best_obj = current_obj
                                best_route = current_route
                                best_order_size = current_order_size
                            else:
                                l += 1
                        elif l == 2:  # or_opt
                            if route_len >= 3:
                                route = best_route
                                for k in range(2, route_len):
                                    for j in range(1, k):
                                        for i in range(j):
                                            temp1 = route[i:j]
                                            temp2 = route[j:k + 1]
                                            new_route = route[0:i]
                                            new_route.extend(temp2)
                                            new_route.extend(temp1)
                                            new_route.extend(route[k + 1:route_len])
                                            new_solution.route_v[routes[route_num]] = new_route
                                            new_obj, new_order_size = calculate_objective(new_solution, problem)
                                            if new_obj < current_obj:
                                                current_obj = new_obj
                                                current_route = new_route
                                                current_order_size = new_order_size
                                if current_obj < best_obj:
                                    l = 1
                                    best_obj = current_obj
                                    best_route = current_route
                                    best_order_size = current_order_size
                                else:
                                    l += 1
                            else:
                                l = 999
                        elif l == 3:  # 3_opt
                            if route_len >= 3:
                                route = best_route
                                for k in range(2, route_len):
                                    for j in range(1, k):
                                        for i in range(j):
                                            temp1 = route[i:j]
                                            temp1 = temp1[::-1]
                                            temp2 = route[j:k + 1]
                                            temp2 = temp2[::-1]
                                            new_route = route[0:i]
                                            new_route.extend(temp1)
                                            new_route.extend(temp2)
                                            new_route.extend(route[k + 1:route_len])
                                            new_solution.route_v[routes[route_num]] = new_route
                                            new_obj, new_order_size = calculate_objective(new_solution, problem)
                                            if new_obj < current_obj:
                                                current_obj = new_obj
                                                current_route = new_route
                                                current_order_size = new_order_size
                                for k in range(2, route_len):
                                    for j in range(1, k):
                                        for i in range(j):
                                            temp1 = route[i:j]
                                            temp1 = temp1[::-1]
                                            temp2 = route[j:k + 1]
                                            new_route = route[0:i]
                                            new_route.extend(temp2)
                                            new_route.extend(temp1)
                                            new_route.extend(route[k + 1:route_len])
                                            new_solution.route_v[routes[route_num]] = new_route
                                            new_obj, new_order_size = calculate_objective(new_solution, problem)
                                            if new_obj < current_obj:
                                                current_obj = new_obj
                                                current_route = new_route
                                                current_order_size = new_order_size
                                for k in range(2, route_len):
                                    for j in range(1, k):
                                        for i in range(j):
                                            temp1 = route[i:j]
                                            temp2 = route[j:k + 1]
                                            temp2 = temp2[::-1]
                                            new_route = route[0:i]
                                            new_route.extend(temp2)
                                            new_route.extend(temp1)
                                            new_route.extend(route[k + 1:route_len])
                                            new_solution.route_v[routes[route_num]] = new_route
                                            new_obj, new_order_size = calculate_objective(new_solution, problem)
                                            if new_obj < current_obj:
                                                current_obj = new_obj
                                                current_route = new_route
                                                current_order_size = new_order_size
                                if current_obj < best_obj:
                                    l = 1
                                    best_obj = current_obj
                                    best_route = current_route
                                    best_order_size = current_order_size
                                else:
                                    l += 1
                            else:
                                l = 999
                        else:
                            print('Error: wrong l, l = ', l)
                    solution.route_v[routes[route_num]] = best_route
                    solution.obj = best_obj
                    solution.order_size_total = best_order_size
            elif mode == 'helicopter':
                route_len = len(solution.route_h[routes[route_num]])
                if route_len >= 2:
                    best_obj = solution.obj
                    current_obj = solution.obj
                    new_solution = copy.deepcopy(solution)
                    current_route = solution.route_h[routes[route_num]]
                    best_route = solution.route_h[routes[route_num]]
                    current_order_size = solution.order_size_total
                    best_order_size = solution.order_size_total
                    l = 1
                    while l <= 3:
                        if l == 1:  # 2opt
                            route = best_route
                            for j in range(1, route_len):
                                for i in range(j):
                                    temp = route[i:j + 1]
                                    temp = temp[::-1]
                                    new_route = route[0:i]
                                    new_route.extend(temp)
                                    new_route.extend(route[j + 1:route_len])
                                    new_solution.route_h[routes[route_num]] = new_route
                                    new_obj, new_order_size = calculate_objective(new_solution, problem)
                                    if new_obj < current_obj:
                                        current_obj = new_obj
                                        current_route = new_route
                                        current_order_size = new_order_size
                            if current_obj < best_obj:
                                l = 1
                                best_obj = current_obj
                                best_route = current_route
                                best_order_size = current_order_size
                            else:
                                l += 1
                        elif l == 2:  # or_opt
                            if route_len >= 3:
                                route = best_route
                                for k in range(2, route_len):
                                    for j in range(1, k):
                                        for i in range(j):
                                            temp1 = route[i:j]
                                            temp2 = route[j:k + 1]
                                            new_route = route[0:i]
                                            new_route.extend(temp2)
                                            new_route.extend(temp1)
                                            new_route.extend(route[k + 1:route_len])
                                            new_solution.route_h[routes[route_num]] = new_route
                                            new_obj, new_order_size = calculate_objective(new_solution, problem)
                                            if new_obj < current_obj:
                                                current_obj = new_obj
                                                current_route = new_route
                                                current_order_size = new_order_size
                                if current_obj < best_obj:
                                    l = 1
                                    best_obj = current_obj
                                    best_route = current_route
                                    best_order_size = current_order_size
                                else:
                                    l += 1
                            else:
                                l = 999
                        elif l == 3:  # 3_opt
                            if route_len >= 3:
                                route = best_route
                                for k in range(2, route_len):
                                    for j in range(1, k):
                                        for i in range(j):
                                            temp1 = route[i:j]
                                            temp1 = temp1[::-1]
                                            temp2 = route[j:k + 1]
                                            temp2 = temp2[::-1]
                                            new_route = route[0:i]
                                            new_route.extend(temp1)
                                            new_route.extend(temp2)
                                            new_route.extend(route[k + 1:route_len])
                                            new_solution.route_h[routes[route_num]] = new_route
                                            new_obj, new_order_size = calculate_objective(new_solution, problem)
                                            if new_obj < current_obj:
                                                current_obj = new_obj
                                                current_route = new_route
                                                current_order_size = new_order_size
                                for k in range(2, route_len):
                                    for j in range(1, k):
                                        for i in range(j):
                                            temp1 = route[i:j]
                                            temp1 = temp1[::-1]
                                            temp2 = route[j:k + 1]
                                            new_route = route[0:i]
                                            new_route.extend(temp2)
                                            new_route.extend(temp1)
                                            new_route.extend(route[k + 1:route_len])
                                            new_solution.route_h[routes[route_num]] = new_route
                                            new_obj, new_order_size = calculate_objective(new_solution, problem)
                                            if new_obj < current_obj:
                                                current_obj = new_obj
                                                current_route = new_route
                                                current_order_size = new_order_size
                                for k in range(2, route_len):
                                    for j in range(1, k):
                                        for i in range(j):
                                            temp1 = route[i:j]
                                            temp2 = route[j:k + 1]
                                            temp2 = temp2[::-1]
                                            new_route = route[0:i]
                                            new_route.extend(temp2)
                                            new_route.extend(temp1)
                                            new_route.extend(route[k + 1:route_len])
                                            new_solution.route_h[routes[route_num]] = new_route
                                            new_obj, new_order_size = calculate_objective(new_solution, problem)
                                            if new_obj < current_obj:
                                                current_obj = new_obj
                                                current_route = new_route
                                                current_order_size = new_order_size
                                if current_obj < best_obj:
                                    l = 1
                                    best_obj = current_obj
                                    best_route = current_route
                                    best_order_size = current_order_size
                                else:
                                    l += 1
                            else:
                                l = 999
                        else:
                            print('Error: wrong l, l = ', l)
                    solution.route_h[routes[route_num]] = current_route
                    solution.obj = best_obj
                    solution.order_size_total = best_order_size
    return solution


def generate_initial_solution(problem):
    solution = Solution(problem)
    route_all = random.sample(range(1, problem.N), problem.N - 1)
    solution.pool_v = copy.deepcopy(route_all)
    solution.pool_h = copy.deepcopy(route_all)
    new_obj, new_order_size = calculate_objective(solution, problem)
    solution.obj = new_obj
    solution.order_size_total = new_order_size

    new_solution = copy.deepcopy(solution)
    while 1:
        best_location = -1
        best_operation_cost = 9999999999999999
        if new_solution.pool_v:
            for i in range(len(new_solution.pool_v)):
                for j in range(problem.K):
                    for k in range(len(new_solution.route_v[j]) + 1):
                        new_solution.route_v[j].insert(k, new_solution.pool_v[i])
                        new_obj, new_order_size = calculate_objective(new_solution, problem)
                        operation_cost = new_obj - new_solution.obj
                        new_solution.route_v[j].pop(k)
                        if operation_cost < best_operation_cost:
                            best_operation_cost = operation_cost
                            best_mode = 'vehicle'
                            best_obj = new_obj
                            best_order_size = new_order_size
                            best_location = i
                            best_route = j
                            best_position = k
        if new_solution.pool_h:
            for i in range(len(new_solution.pool_h)):
                for j in range(problem.A):
                    for k in range(len(new_solution.route_h[j]) + 1):
                        new_solution.route_h[j].insert(k, new_solution.pool_h[i])
                        new_obj, new_order_size = calculate_objective(new_solution, problem)
                        operation_cost = new_obj - new_solution.obj
                        new_solution.route_h[j].pop(k)
                        if operation_cost < best_operation_cost:
                            best_operation_cost = operation_cost
                            best_mode = 'helicopter'
                            best_obj = new_obj
                            best_order_size = new_order_size
                            best_location = i
                            best_route = j
                            best_position = k
        if best_location != -1:
            if best_mode == 'vehicle':
                temp = new_solution.pool_v.pop(best_location)
                new_solution.route_v[best_route].insert(best_position, temp)
                new_solution.obj = best_obj
                new_solution.order_size_total = best_order_size
            elif best_mode == 'helicopter':
                temp = new_solution.pool_h.pop(best_location)
                new_solution.route_h[best_route].insert(best_position, temp)
                new_solution.obj = best_obj
                new_solution.order_size_total = best_order_size
        if new_solution.obj >= solution.obj:
            break
        else:
            solution = copy.deepcopy(new_solution)
    return solution



def calculate_inventory_cost_matrix(problem):
    for i in range(1, problem.N):
        inventory_cost = 0
        for t in range(1, problem.H + 1):
            inventory_cost += max(problem.h * (problem.initial_inventory_level[i] - t * (problem.dm - problem.dh)),
                                  -1 * problem.s * (problem.initial_inventory_level[i] - t * (problem.dm + problem.dh)))
        if inventory_cost == 0:
            problem.inventory_cost_matrix_zero[i] = 99999999999
        else:
            problem.inventory_cost_matrix_zero[i] = inventory_cost
    for i in range(1, problem.N):
        for l in range(1, problem.T + 1):
            inventory_cost = 0
            q = 0
            for t in range(1, l):
                inventory_cost += max(problem.h * (problem.initial_inventory_level[i] - t * (problem.dm - problem.dh)),
                                      -1 * problem.s * (
                                              problem.initial_inventory_level[i] - t * (problem.dm + problem.dh)))
            A = (problem.s + problem.h) * problem.dm + (problem.s - problem.h) * problem.dh
            if problem.initial_inventory_level[i] < A / (problem.s + problem.h) * problem.H:
                q = A * (problem.s * problem.H + problem.h * l) / ((problem.s + problem.h) ** 2) - \
                    problem.initial_inventory_level[i]
            else:
                q = 0
            for t in range(l, problem.H + 1):
                inventory_cost += max(
                    problem.h * (problem.initial_inventory_level[i] + q - t * (problem.dm - problem.dh)),
                    -1 * problem.s * (problem.initial_inventory_level[i] + q - t * (problem.dm + problem.dh)))
            if inventory_cost == 0:
                problem.inventory_cost_matrix_one[i, l] = 99999999999
            else:
                problem.inventory_cost_matrix_one[i, l] = inventory_cost
                problem.shelter_order_size_matrix_one[i, l] = q
    for i in range(1, problem.N):
        for l_fast in range(1, problem.T + 1):
            for l_slow in range(l_fast, problem.T + 1):
                inventory_cost = 0
                q_f = 0
                q_s = 0
                for t in range(1, l_fast):
                    inventory_cost += max(
                        problem.h * (problem.initial_inventory_level[i] - t * (problem.dm - problem.dh)),
                        -1 * problem.s * (problem.initial_inventory_level[i] - t * (problem.dm + problem.dh)))
                A = (problem.s + problem.h) * problem.dm + (problem.s - problem.h) * problem.dh
                if problem.initial_inventory_level[i] < A / (problem.s + problem.h) * l_slow:
                    q_f = A * (problem.s * l_slow + problem.h * l_fast) / ((problem.s + problem.h) ** 2) - \
                          problem.initial_inventory_level[i]
                else:
                    q_f = 0
                if problem.initial_inventory_level[i] < A / (problem.s + problem.h) * problem.H - q_f:
                    q_s = A * (problem.s * problem.H + problem.h * l_slow) / ((problem.s + problem.h) ** 2) - \
                          problem.initial_inventory_level[i] - q_f
                else:
                    q_s = 0
                for t in range(l_fast, l_slow):
                    inventory_cost += max(
                        problem.h * (problem.initial_inventory_level[i] + q_f - t * (problem.dm - problem.dh)),
                        -1 * problem.s * (problem.initial_inventory_level[i] + q_f - t * (problem.dm + problem.dh)))
                for t in range(l_slow, problem.H + 1):
                    inventory_cost += max(
                        problem.h * (problem.initial_inventory_level[i] + q_f + q_s - t * (problem.dm - problem.dh)),
                        -1 * problem.s * (
                                problem.initial_inventory_level[i] + q_f + q_s - t * (problem.dm + problem.dh)))
                if inventory_cost == 0:
                    problem.inventory_cost_matrix_two[i, l_fast, l_slow] = 99999999999
                else:
                    problem.inventory_cost_matrix_two[i, l_fast, l_slow] = inventory_cost
                    problem.shelter_order_size_matrix_two_q_f[i, l_fast, l_slow] = q_f
                    problem.shelter_order_size_matrix_two_q_s[i, l_fast, l_slow] = q_s
    return
