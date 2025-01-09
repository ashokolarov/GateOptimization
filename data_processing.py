from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar, Tuple

import numpy as np
import pandas as pd
import yaml

from utils import hms_to_total_seconds, total_seconds_to_hms


@dataclass
class Flight:
    number: int             # Flight number
    category: int           # Performance category
    type: int               # Operational type
    arrival_time: datetime  # Arrival time
    occupancy_time: int     # Time spent at the gate

@dataclass
class Gate:
    """
    number_to_id and id_to_number are helper dictionaries to map the gate number(0, 1,...)
    to the gate id(101, 102, ...) and vice versa.
    """
    number_to_id: ClassVar[dict]
    id_to_number: ClassVar[dict]

    number: int                # Gate number
    distance_to_runway: int    # Distance to the runway
    allowed_categories: list   # List of allowed categories
    allowed_types: list        # List of allowed types
    id: int = None             # Gate id

    num_allocation: int = 0    # Number of flights allocated to the gate in total

    def __post_init__(self):
        if type(self.allowed_categories) is int:
            self.allowed_categories = [int(self.allowed_categories)]
        else:
            self.allowed_categories = [int(x) for x in self.allowed_categories.split(",")]
        if type(self.allowed_types) is int:
            self.allowed_types = [int(self.allowed_types)]
        else:
            self.allowed_types = [int(x) for x in self.allowed_types.split(",")]
        
        self.id = self.number_to_id[self.number]
      
@dataclass
class BaggageCarousel:
    number: int              # Carousel number
    allowed_gates: list      # List of allowed gates
    distances_to_gates: list # List of distances to the gates

    num_allocation: int = 0  # Number of gates allocated to the carousel in total

    def __post_init__(self):
        self.allowed_gates = [Gate.id_to_number[int(x)] for x in self.allowed_gates.split(",")]
        self.distances_to_gates = [int(x) for x in self.distances_to_gates.split(",")]

        mapping = zip(self.allowed_gates, self.distances_to_gates)
        self.distances_to_gates = dict(mapping)

def read_config(filename: str) -> dict:
    """
    Read the configuration file and return the configuration as a dictionary.
    filename: str - The name of the configuration file
    """
    with open(filename, 'r') as file:
        config = yaml.safe_load(file)

    # Convert gate occupancy from minutes to seconds
    config['gate_occupancy'][0] *= 60    
    config['gate_occupancy'][1] *= 60
    # Convert buffer time from minutes to seconds
    config['buffer_time'] *= 60
    
    return config

def process_data(config_name: str) -> Tuple[dict, dict]:
    """
    Read the data from the Excel file and return the configuration and the data as dictionaries.
    config_name: str - The name of the configuration file
    """

    config = read_config(config_name)
   
    data_file = pd.ExcelFile(config['data_file'])
    data = {"Flights" : [], "Gates" : [], "Carousels" : []}

    flights = pd.read_excel(data_file, "Flights")
    number_of_flights = len(flights)

    # Set the date to today to create datetime object
    for i in range(number_of_flights):
        arrival_time = list(map(int, flights["Arrival Time"][i].split(":")))
        arrival_time = hms_to_total_seconds(*arrival_time)
        occupancy_time = config['gate_occupancy'][flights['Performance Category'][i]]

        data["Flights"].append(Flight(number=flights['Number'][i],
                                      category=flights['Performance Category'][i],
                                      type=flights['Operational Type'][i],
                                      arrival_time=arrival_time,
                                      occupancy_time=occupancy_time))
    
    gates = pd.read_excel(data_file, "Gates")
    number_of_gates = len(gates)

    Gate.number_to_id = dict(zip(gates["Number"], gates['ID']))
    Gate.id_to_number = dict(zip(gates['ID'], gates["Number"]))

    for i in range(number_of_gates):
        data["Gates"].append(Gate(number=gates['Number'][i],
                                  distance_to_runway=gates['Distance To Runway'][i],
                                  allowed_categories=gates['Allowed Performance Categories'][i],
                                  allowed_types=gates['Allowed Operational Types'][i]))
        
    baggage = pd.read_excel(data_file, "Baggage")
    number_of_carousels = len(baggage)
    for i in range(number_of_carousels):
        data["Carousels"].append(BaggageCarousel(number=baggage['Number'][i],
                                                allowed_gates=baggage['Allowed Gates'][i],
                                                distances_to_gates=baggage['Distances To Gates'][i]))

    return config, data

def generate_flights(config_name: str) -> None:
    """
    Generate new flights and write them to the Excel file.
    config_name: str - The name of the configuration file
    """
    config = read_config(config_name)

    num_flights = config['num_flights']
    traffic_hours = config['traffic_hours']
    domestic_to_international_ratio = config['domestic_to_total_ratio']
    narrow_to_wide_ratio = config['narrow_to_total_ratio']

    numbers = np.arange(num_flights)
    
    # Create operational types by sampling a binominal trial with biased probability
    flight_types = np.random.binomial(1, size=num_flights, p=1-domestic_to_international_ratio)
    # Create performance categories by sampling a binominal trial with biased probability
    flight_categories = np.random.binomial(1, size=num_flights, p=1-narrow_to_wide_ratio)

    # Create arrival times by sampling a uniform distribution
    arrival_times = np.sort(np.random.randint(traffic_hours[0], traffic_hours[1], num_flights))
    arrival_times = [total_seconds_to_hms(arrival_time, to_str=True) for arrival_time in arrival_times]

    # Create a new DataFrame with the new data
    new_flights = pd.DataFrame({
        "Number": numbers,
        "Performance Category": flight_categories,
        "Operational Type": flight_types,
        "Arrival Time": arrival_times
    })

    # Write the DataFrame to the Excel file
    with pd.ExcelWriter(config['data_file'], mode='a', if_sheet_exists='replace') as writer:
        new_flights.to_excel(writer, sheet_name="Flights", index=False)


if __name__ == "__main__":
    generate_flights("data/config.yaml")
    # config, data = process_data("data/config.yaml")
    
