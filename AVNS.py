########################################################################################################################
# Dual-sourcing Inventory Routing Problem with route-dependent-lead-times in a rolling horizon framework
# Adaptive Variable Neighborhood Search Algorithm
# Python>=3.8
# Author: Weibo Zheng
########################################################################################################################

import base
import numpy as np
import math
import re
import csv
import random
import copy
import time
import os

# --------------------------- Global Configuration ---------------------------------------

shaking_operators_list = ['Removal', 'Insertion', 'Exchange', 'Swap', 'Relocate']

##################################################################################

for root, dirs, files in os.walk('./Test_set'):
    for set_name in files:
        print(set_name)
        time_start = time.time()
        instance = set_name
        problem = base.generate_problem(instance)
        base.calculate_inventory_cost_matrix(problem)
        records = base.Record(shaking_operators_list)
        solution = base.generate_initial_solution(problem)
        shaking_operators = base.Shaking_operators(shaking_operators_list)
        current_solution = copy.deepcopy(solution)
        best_solution = copy.deepcopy(solution)
        time_end = time.time()
        time_cost = time_end - time_start
        records.initial_generator_time_cost = time_cost
        time_start = time.time()
        iteration_num = 0
        iteration_counter = 0
        segment_counter = 0
        while problem.temperature >= 0.01:
            if iteration_counter >= problem.segments_capacity:
                segment_counter += 1
                iteration_counter = 0
                for i in shaking_operators_list:
                    records.shaking_operator_score_record[i].append(shaking_operators.shaking_operator_score[i])
                    records.shaking_operator_weights_record[i].append(shaking_operators.shaking_operator_weight[i])
                    records.shaking_operator_times_record[i].append(shaking_operators.shaking_operator_times[i])
                shaking_operators.update_weight(shaking_operators_list, problem)
                for i in shaking_operators_list:
                    shaking_operators.shaking_operator_score[i] = 0
                    shaking_operators.shaking_operator_times[i] = 0
            solution = copy.deepcopy(current_solution)
            selected_operator = shaking_operators.select(shaking_operators_list)
            shaking_operators.shaking_operator_times[selected_operator] += 1
            mode, routes, solution = base.do_shaking(selected_operator, solution, problem)
            solution = base.do_VND(mode, routes, solution, problem)
            if solution.obj < current_solution.obj:
                if solution.obj < best_solution.obj:
                    shaking_operators.shaking_operator_score[selected_operator] += problem.score_factor_1
                    current_solution = copy.deepcopy(solution)
                    best_solution = copy.deepcopy(solution)
                    print(selected_operator, ' get best solution= ', best_solution.obj)
                    records.best_solution_update_iteration_num.append(iteration_num)
                else:
                    shaking_operators.shaking_operator_score[selected_operator] += problem.score_factor_2
                    current_solution = copy.deepcopy(solution)
                    print(selected_operator, ' AC!!!!!!!!!')
            elif random.random() < math.exp(-(solution.obj - current_solution.obj) / problem.temperature):
                if solution.obj > best_solution.obj:
                    shaking_operators.shaking_operator_score[selected_operator] += problem.score_factor_3
                    current_solution = copy.deepcopy(solution)
                    print(selected_operator, ' SAC!!!!!!!!!!!')
            else:
                print(selected_operator, 'Reject!!!!!!!')
            iteration_num += 1
            iteration_counter += 1
            time_end = time.time()
            time_cost = time_end - time_start
            problem.temperature = problem.temperature * problem.cooling_rate
            print('iteration ' + str(iteration_num) + ' finished!')
        print(problem.instance_name, ' has finished!')
        with open('Data.csv', 'a', newline='', encoding="ANSI") as f_out:
            csv.writer(f_out, delimiter=',').writerow([set_name, best_solution.obj, time_cost])
        print(problem.instance_name, ' has finished!')
