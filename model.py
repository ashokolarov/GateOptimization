import os
from typing import Final

import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator

from data_processing import process_data
from utils import DOMESTIC_CAROUSELS, OPEN_PARK_GATES, total_seconds_to_hms


class GateAssignment:
    M: Final[int] = 1e6 # big constant used to define constraints

    def __init__(self, config_name: str):
        self.config, self.data = process_data(config_name)

        self.identifier = self.config.get("identifier", "default")
        self.results_dir = self.config.get("output_path", "results") + "/" + self.identifier
        if not os.path.exists(self.results_dir):
            os.mkdir(self.results_dir)
        print(f"Initializing model with identifier:\n{self.identifier} \n")
        
        ### Ideal and Nadir Points ###
        self.d_min = self.config.get("d_min", 0)
        self.d_max = self.config.get("d_max", 0)
        self.f_min = self.config.get("f_min", 0)
        self.f_max = self.config.get("f_max", 0)
        

    def _define_model(self, model_config: dict):
        self.model = gp.Model(model_config["identifier"])

        flights = self.data["Flights"]
        gates = self.data["Gates"]
        carousels = self.data["Carousels"]

        number_of_flights = len(flights)
        number_of_gates = len(gates)
        number_of_carousels = len(carousels)

        ### Decision Variables ###
        q = {} # the gate entrance time of aircraft i for gate j
        x = {} # binary variable that takes a value of 1 if aircraft i is assigned to gate j and baggage carousel k, otherwise, it is zero
        d = {} # distance from gate to baggage carousel for aircraft i
        f = {} # the fuel consumption value of aircraft i during taxi manoeuvre

        for i in range(number_of_flights):
            d[i] = self.model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name=f"d_{i}")
            f[i] = self.model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name=f"f_{i}")
            for j in range(number_of_gates):
                q[i,j] = self.model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name=f"q_{i}_{j}")
                for k in range(number_of_carousels):
                    x[i,j,k] = self.model.addVar(lb=0, ub=1, vtype=gp.GRB.BINARY, name=f"x_{i}_{j}_{k}")

        self.model.update()

        ### Constraints ###

        # Constraint 1: Each aircraft is assigned to one gate and one baggage carousel
        for i in range(number_of_flights):
            lhs_ct_1 = gp.LinExpr()
            for j in range(number_of_gates):
                for k in range(number_of_carousels):
                    lhs_ct_1 += x[i,j,k]

            self.model.addLConstr(lhs=lhs_ct_1, sense=gp.GRB.EQUAL, rhs=1,
                                  name=f"flight_{i}_unique_gate_and_carousel")
        
        # Constraints 2,3 and 4
        for i in range(number_of_flights):
            flight = flights[i]
            for j in range(number_of_gates):
                gate = gates[j]
                for k in range(number_of_carousels):
                    carousel = carousels[k]

                    if flight.category not in gate.allowed_categories:
                        flight_gate_category = 0
                    else:
                        flight_gate_category = 1
                    # Constraint 2: Each aircraft is assigned to a gate that is compatible with its performance category
                    self.model.addLConstr(lhs=x[i,j,k], sense=gp.GRB.LESS_EQUAL, rhs=flight_gate_category,
                                          name=f"flight_{i}_gate_{j}_category_compatibility_{k}")
                    
                    if flight.type not in gate.allowed_types:
                        flight_gate_type = 0
                    else:
                        flight_gate_type = 1
                    # Constraint 3: Each aircraft is assigned to a gate that is compatible with its operational type
                    self.model.addLConstr(lhs=x[i,j,k], sense=gp.GRB.LESS_EQUAL, rhs=flight_gate_type,
                                          name=f"flight_{i}_gate_{j}_type_compatibility_{k}")
                    
                    if gate.number not in carousel.allowed_gates:
                        gate_carousel_compatibility = 0
                    else:
                        gate_carousel_compatibility = 1
                    # Constraint 4: Each aircraft can reach the baggage carousel using the assigned gate
                    self.model.addLConstr(lhs=x[i,j,k], sense=gp.GRB.LESS_EQUAL, rhs=gate_carousel_compatibility,
                                          name=f"gate_{j}_carousel_{k}_type_compatibility_{i}")
                    
        # Constraint 5: the gate entrance time of aircraft i for gate j using taxi distance and arrival time
        for i in range(number_of_flights):
            flight = flights[i]
            for j in range(number_of_gates):
                gate = gates[j]

                rhs_ct_5 = gp.LinExpr()
                rhs_ct_5 += flight.arrival_time - self.config['buffer_time'] // 2
                for k in range(number_of_carousels):
                    rhs_ct_5 += int(gate.distance_to_runway / self.config['taxi_speed']) * x[i,j,k]
                
                self.model.addLConstr(lhs=q[i,j], sense=gp.GRB.EQUAL, rhs=rhs_ct_5,
                                      name=f"flight_{i}_gate_{j}_gate_entrance_time")
                
        # Constraint 6: maintains the separation time between two consecutive aircraft at the same gate
        for j in range(number_of_gates):
            for i1 in range(number_of_flights-1):
                for i2 in range(1, number_of_flights):
                    if i1 != i2:
                        for k1 in range(number_of_carousels):
                            for k2 in range(number_of_carousels):
                                lhs_ct_6 = q[i2, j] - q[i1, j]
                                rhs_ct_6 = flights[i1].occupancy_time + self.config['buffer_time'] - self.M * (2 - x[i1, j, k1] - x[i2, j, k2])
                                self.model.addLConstr(lhs=lhs_ct_6, sense=gp.GRB.GREATER_EQUAL, rhs=rhs_ct_6,
                                                      name=f"gate_{j}_separation_time_flights_{i1}_{i2}_{k1}_{k2}")

        # Constraint 7 & 8
        for i in range(number_of_flights):
            flight = flights[i]

            lhs_ct_7 = gp.LinExpr()
            lhs_ct_8 = gp.LinExpr()
            for j in range(number_of_gates):
                gate = gates[j]
                for k in range(number_of_carousels):
                    carousel = carousels[k]

                    # Constraint 7: calculate the fuel consumption value of aircraft i during taxi manoeuvre
                    fuel_consumption = self.config['fuel_consumption'][flight.type]
                    lhs_ct_7 += int(gate.distance_to_runway / self.config['taxi_speed']) * fuel_consumption * x[i,j,k]

                    # Constraint 8: calculate the walking distance from gate to baggage carousel for each aircraft
                    # If gate is not in allowed gates, assign a big value to the distance
                    if gate.number in carousel.allowed_gates:
                        lhs_ct_8 += carousel.distances_to_gates[gate.number] * x[i,j,k]
                    else:
                        lhs_ct_8 += self.M * x[i,j,k]

            if 7 not in model_config["constraints"]:
                self.model.addLConstr(lhs=f[i], sense=gp.GRB.EQUAL, rhs=lhs_ct_7,
                                    name=f"flight_{i}_fuel_consumption")

            if 8 not in model_config["constraints"]:
                self.model.addLConstr(lhs=d[i], sense=gp.GRB.EQUAL, rhs=lhs_ct_8,
                                    name=f"flight_{i}_distance_gate_to_carousel")
            
        # Constraint 9 & 10
        lhs_ct_9 = gp.LinExpr()
        lhs_ct_10 = gp.LinExpr()
        for i in range(number_of_flights):
            lhs_ct_9 += f[i]
            lhs_ct_10 += d[i]

        # Constraint 9: Calculate nadir point for fuel consumption
        if 9 not in model_config["constraints"]:
            self.model.addLConstr(lhs=lhs_ct_9, sense=gp.GRB.LESS_EQUAL, rhs=self.f_min,
                                name=f"fuel_consumption_nadir")
        
        # Constraint 10: Calculate nadir point for walking distance from gate to baggage carousel
        if 10 not in model_config["constraints"]:
            self.model.addLConstr(lhs_ct_10, sense=gp.GRB.LESS_EQUAL, rhs=self.d_min,
                                name=f"distance_nadir")

        self.model.update()

        ### Objective Function ###
        dist_obj = gp.LinExpr()
        fuel_obj = gp.LinExpr()
        total_objective = gp.LinExpr()

        for i in range(number_of_flights):
            fuel_obj += f[i]
            dist_obj += d[i]

        total_objective = model_config["objective"]['dist_weight'] * dist_obj \
            + model_config["objective"]['fuel_weight'] * fuel_obj

        self.model.setObjective(total_objective, gp.GRB.MINIMIZE)
        self.model.update()
        
        self.model.write(model_config['save_dir'] + "model.lp")

        
    def _optimize(self):
        self.model.optimize()

        status = self.model.status
    
        if status == gp.GRB.Status.UNBOUNDED:
            raise Exception('Model was proven to be unbounded. \
                  Important note: an unbounded status indicates the presence of an unbounded ray that allows the objective to improve without limit.')
        elif status == gp.GRB.Status.INFEASIBLE:
            raise Exception('Model was proven to be infeasible.')
        elif status == gp.GRB.Status.OPTIMAL or True:
            solution = {"x" : {}, "q" : {}, "d" : {}, "f" : {}}
        
            for v in self.model.getVars():
                if v.varName[0] == "x" and v.x != 0:
                    i, j, k = map(int, v.varName.split("_")[1:])
                    solution["x"][i, j, k] = v.x
                if v.varName[0] == "q" and v.x != 0:
                    i, j = map(int, v.varName.split("_")[1:])
                    solution["q"][i, j] = v.x
                if v.varName[0] == "d" and v.x != 0:
                    i = int(v.varName.split("_")[1])
                    solution["d"][i] = v.x
                if v.varName[0] == "f" and v.x != 0:
                    i = int(v.varName.split("_")[1])
                    solution["f"][i] = v.x

            return self.model.objVal, solution

    def _plot_results(self, solution, model_identifier, save_dir=None):
        flights = self.data["Flights"]
        gates = self.data["Gates"]

        number_of_gates = len(gates)

        fig, ax = plt.subplots(figsize=(16, 8))
        
        print(model_identifier)
        for (i, j, k), _ in solution["x"].items():
            flight = flights[i]
            gate = gates[j]

            gate.num_allocation += 1

            arrival_time = solution["q"][i, j]
            occupancy_time = flight.occupancy_time
            departure_time = arrival_time + occupancy_time

            print(f"Flight {i} assigned to Gate {gate.id} and Carousel {k} at {total_seconds_to_hms(arrival_time, to_str=True)}, leaving at {total_seconds_to_hms(departure_time, to_str=True)}")

            ax.barh(j, left=arrival_time, width=occupancy_time, edgecolor='black', linewidth=1.5, zorder=3)
            ax.barh(j, left=arrival_time - self.config['buffer_time'] // 2, width=self.config['buffer_time'] // 2, color='lightgrey', edgecolor='black', linewidth=1.5, zorder=3)
            ax.barh(j, left=departure_time , width=self.config['buffer_time'] // 2, color='lightgrey', edgecolor='black', linewidth=1.5, zorder=3)
            ax.text(arrival_time + occupancy_time // 2, j, f"Flight {i}, Category: {flight.category}, Type: {flight.type}", ha='center', va='center', color='white', fontsize=10)
            
        def format_func(value, tick_number):
            # Convert the decimal time to hours, minutes, and seconds
            hms = total_seconds_to_hms(value, to_str=True)
            hm = hms[:-3]
            return hm

        ### X-Axis ###
        # Create a FuncFormatter object
        formatter = FuncFormatter(format_func)

        # Set the formatter for the x-axis
        ax.xaxis.set_major_formatter(formatter)
        ax.set_xlabel('Time [HH:MM]', fontsize=15)
        border_time = 2.5*60 + self.config['buffer_time']
        xmin = flights[0].arrival_time - border_time
        xmax = flights[-1].arrival_time + flights[-1].occupancy_time + border_time
        delta_time = 30*60
        x_ticks = np.arange(xmin, xmax, delta_time)
        x_ticks = x_ticks - (x_ticks % delta_time)
        plt.xticks(x_ticks, rotation=45)
        ax.set_xlim([xmin if x_ticks[0] < xmin else x_ticks[0], xmax if x_ticks[-1] < xmax else x_ticks[-1]])

        ### Y-Axis ###
        ax.set_ylabel('Gate', fontsize=15)
        y_ticks = np.arange(number_of_gates)
        ax.set_yticks(y_ticks)
        ax.set_ylim([y_ticks[0]-1, y_ticks[-1]+1])

        # Plot a dashed horizontal line at every y-tick
        for y in y_ticks:
            ax.hlines(y, xmin=xmin, xmax=xmax, colors='grey', linestyles='dashed', linewidth=1)
        
        labels = [item.get_text() for item in ax.get_yticklabels()]
        for i, label in enumerate(labels):
            labels[i] = gates[i].id
        ax.set_yticklabels(labels)

        if self.config['plot_flight_gate_verification']:
            ax2 = ax.twinx()
            ax2.set_yticks(y_ticks)
            ax2.set_ylim(ax.get_ylim())

            # Set the labels of the second y-axis to the extra information
            ax2_labels = []
            for gate in gates:
                ctgs = str(gate.allowed_categories)[1:-1]
                tps = str(gate.allowed_types)[1:-1]
                if len(gate.allowed_categories) == 1:
                    ax2_label = f"C: {ctgs}      T: {tps}"
                else:
                    ax2_label = f"C: {ctgs}  T: {tps}"
                ax2_labels.append(ax2_label)
            ax2.set_yticklabels(ax2_labels, fontsize=10)

        plt.tight_layout()
        if save_dir is not None:
            plt.savefig(save_dir+ f"{model_identifier}_flight_gate_assignment.pdf")
        plt.show()

        fig, ax = plt.subplots(figsize=(10, 6))

        # Create dummy bar plots for the labels
        ax.bar(0, 0, color='lightgreen', edgecolor='black', linewidth=1.5, label="Open Park Gates")
        ax.bar(0, 0, color='lightcoral', edgecolor='black', linewidth=1.5, label="Terminal Gates")

        ### X-Axis ###
        ax.set_xlabel('Gate', fontsize=15)
        x_ticks = np.arange(number_of_gates)
        plt.xticks(x_ticks, rotation=45)

        for gate in gates:
            if gate.id in OPEN_PARK_GATES:
                ax.bar(gate.number, gate.num_allocation, color='lightgreen', edgecolor='black', linewidth=1.5)
            else:
                ax.bar(gate.number, gate.num_allocation, color='lightcoral', edgecolor='black', linewidth=1.5)
            
            if gate.num_allocation != 0:
                ax.hlines(gate.num_allocation, xmin=x_ticks[0]-1, xmax=x_ticks[-1]+1, colors='grey', linestyles='dashed', linewidth=1)

        labels = [item.get_text() for item in ax.get_xticklabels()]
        for i, label in enumerate(labels):
            labels[i] = gates[i].id
        ax.set_xticklabels(labels)
        ax.set_xlim([x_ticks[0]-1, x_ticks[-1]+1])

        ### Y-Axis ###
        ax.set_ylabel('Number of Allocations', fontsize=15)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1, ncol=2, loc='upper center', fontsize=12, bbox_to_anchor=(0.5, 1.1))
        if save_dir is not None:
            plt.savefig(save_dir + f"{model_identifier}_gate_allocation.pdf")
        plt.show()

        carousels = self.data["Carousels"]

        number_of_carousels = len(carousels)
        for (i, j, k), _ in solution["x"].items():
            carousel = carousels[k]
            carousel.num_allocation += 1

        fig, ax = plt.subplots(figsize=(10, 6))

        # Create dummy bar plots for the labels
        ax.bar(0, 0, color='tan', edgecolor='black', linewidth=1.5, label="Domestic Carousel")
        ax.bar(0, 0, color='royalblue', edgecolor='black', linewidth=1.5, label="International Carousel")

        ### X-Axis ###
        ax.set_xlabel('Carousel', fontsize=15)
        x_ticks = np.arange(number_of_carousels)
        plt.xticks(x_ticks, rotation=45)

        for carousel in carousels:
            if carousel.number in DOMESTIC_CAROUSELS:
                ax.bar(carousel.number, carousel.num_allocation, color='tan', edgecolor='black', linewidth=1.5)
            else:
                ax.bar(carousel.number, carousel.num_allocation, color='royalblue', edgecolor='black', linewidth=1.5)
            
            if carousel.num_allocation != 0:
                ax.hlines(carousel.num_allocation, xmin=x_ticks[0]-1, xmax=x_ticks[-1]+1, colors='grey', linestyles='dashed', linewidth=1)

        ### Y-Axis ###
        ax.set_ylabel('Number of Allocations', fontsize=15)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1, ncol=2, loc='upper center', fontsize=12, bbox_to_anchor=(0.5, 1.1))
        if save_dir is not None:
            plt.savefig(save_dir + f"{model_identifier}_carousel_allocation.pdf")
        plt.show()
        

    def _reset_allocation(self):
        gates = self.data["Gates"]
        carousels = self.data["Carousels"]

        for gate in gates:
            gate.num_allocation = 0
        for carousel in carousels:
            carousel.num_allocation = 0

    def _compute_nadir_values(self):
        ### 1. Optimize to find miniminum distance nadir point
        if self.d_min == 0:
            model_identifier = "d_min_model"
            objective_weights = {"dist_weight" : 1, "fuel_weight" : 0}
            constraint_relaxation = [7, 9, 10]
            model_dir = self.results_dir + "/" + model_identifier + "/"
            if not os.path.exists(model_dir):
                os.mkdir(model_dir)
            d_min_config = {"identifier" : model_identifier, "objective" : objective_weights, "constraints" : constraint_relaxation, "save_dir" : model_dir}
            self._define_model(d_min_config)
            self.d_min, solution = self._optimize()
            self._plot_results(solution, model_identifier, model_dir)
            self._reset_allocation()
        ### 2. Optimize to find minimum fuel nadir point
        if self.f_min == 0:
            model_identifier = "f_min_model"
            objective_weights = {"dist_weight" : 0, "fuel_weight" : 1}
            constraint_relaxation = [8, 9, 10]
            model_dir = self.results_dir + "/" + model_identifier + "/"
            if not os.path.exists(model_dir):
                os.mkdir(model_dir)
            f_min_config = {"identifier" : model_identifier, "objective" : objective_weights, "constraints" : constraint_relaxation, "save_dir" : model_dir}
            self._define_model(f_min_config)
            self.f_min, solution = self._optimize()
            self._plot_results(solution, model_identifier, model_dir)
            self._reset_allocation()
        ### 3. Optimize to find maximum distance nadir point
        if self.d_max == 0:
            model_identifier = "d_max_model"
            objective_weights = {"dist_weight" : 1, "fuel_weight" : 0}
            constraint_relaxation = [10]
            model_dir = self.results_dir + "/" + model_identifier + "/"
            if not os.path.exists(model_dir):
                os.mkdir(model_dir)
            f_min_config = {"identifier" : model_identifier, "objective" : objective_weights, "constraints" : constraint_relaxation, "save_dir" : model_dir}
            self._define_model(f_min_config)
            self.d_max, solution = self._optimize()
            self._plot_results(solution, model_identifier, model_dir)
            self._reset_allocation()
        ### 4. Optimize to find maximum fuel nadir point
        if self.f_max == 0:
            model_identifier = "f_max_model"
            objective_weights = {"dist_weight" : 0, "fuel_weight" : 1}
            constraint_relaxation = [9]
            model_dir = self.results_dir + "/" + model_identifier + "/"
            if not os.path.exists(model_dir):
                os.mkdir(model_dir)
            f_min_config = {"identifier" : model_identifier, "objective" : objective_weights, "constraints" : constraint_relaxation, "save_dir" : model_dir}
            self._define_model(f_min_config)
            self.f_max, solution = self._optimize()
            self._plot_results(solution, model_identifier, model_dir)
            self._reset_allocation()

        print("Nadir values calculated")
        print(f"d_min: {self.d_min}, d_max: {self.d_max}")
        print(f"f_min: {self.f_min}, f_max: {self.f_max}")

        return self.d_max, self.f_max
    
    def _verification_case_1(self):
        model_identifier = "_verification_case_1"
        objective_weights = {"dist_weight" : 0, "fuel_weight" : 0}
        constraint_relaxation = [7, 8, 9, 10]
        model_dir = self.results_dir + "/" + model_identifier + "/"
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        v1_config = {"identifier" : model_identifier, "objective" : objective_weights, "constraints" : constraint_relaxation, "save_dir" : model_dir}
        self._define_model(v1_config)
        _, solution = self._optimize()
        self._plot_results(solution, model_identifier, model_dir)
        self._reset_allocation()

    def _verification_case_2(self):
        model_identifier = "_verification_case_2"
        objective_weights = {"dist_weight" : 1, "fuel_weight" : 0}
        constraint_relaxation = [7, 9, 10]
        model_dir = self.results_dir + "/" + model_identifier + "/"
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        v1_config = {"identifier" : model_identifier, "objective" : objective_weights, "constraints" : constraint_relaxation, "save_dir" : model_dir}
        self._define_model(v1_config)
        _, solution = self._optimize()
        self._plot_results(solution, model_identifier, model_dir)
        self._reset_allocation()

    def _verification_case_3(self):
        model_identifier = "_verification_case_3"
        objective_weights = {"dist_weight" : 0, "fuel_weight" : 1}
        constraint_relaxation = [8, 9, 10]
        model_dir = self.results_dir + "/" + model_identifier + "/"
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        v1_config = {"identifier" : model_identifier, "objective" : objective_weights, "constraints" : constraint_relaxation, "save_dir" : model_dir}
        self._define_model(v1_config)
        _, solution = self._optimize()
        self._plot_results(solution, model_identifier, model_dir)
        self._reset_allocation()

    def _sensitivity_analysis(self):
        self._compute_nadir_values()
        model_identifier = "_sensitivity_case_1"
        w = 0.5
        objective_weights = {"dist_weight" : (1-w) * (1/self.d_max), "fuel_weight" : w * (1/self.f_max)}
        constraint_relaxation = [9, 10]
        model_dir = self.results_dir + "/" + model_identifier + "/"
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        s1_config = {"identifier" : model_identifier, "objective" : objective_weights, "constraints" : constraint_relaxation, "save_dir" : model_dir}
        self._define_model(s1_config)
        _, solution = self._optimize()
        self._plot_results(solution, model_identifier, model_dir)
        self._reset_allocation()

    def _final_results(self):
        self._compute_nadir_values()
        model_identifier = "final_results"
        objective_weights = {"dist_weight" : 0.5, "fuel_weight" : 0.5}
        constraint_relaxation = [9, 10]
        model_dir = self.results_dir + "/" + model_identifier + "/"
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        final_config = {"identifier" : model_identifier, "objective" : objective_weights, "constraints" : constraint_relaxation, "save_dir" : model_dir}
        self._define_model(final_config)
        _, solution = self._optimize()
        self._plot_results(solution, model_identifier, model_dir)
        self._reset_allocation()


if __name__ == "__main__":
    config_name = "data/config.yaml"

    problem = GateAssignment(config_name)
    problem._final_results()
