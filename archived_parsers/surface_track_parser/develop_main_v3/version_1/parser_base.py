from abc import ABC, abstractmethod


##########################################################################
class Parser(ABC):
    @abstractmethod
    def organize_stats(self):
        # a function to organize data
        pass

    @abstractmethod
    def generate_csv(self):
        # a function to combine information to generate final csv
        pass

    @abstractmethod
    def save_csv(self):
        # a function to write csv information to disk
        pass

    @abstractmethod
    def process(self):
        # a function to to run a end to end pipeline on a single entity
        # in most cases its and end to end data extraction, organization and saving
        pass

    @abstractmethod
    def filter_stats(self):
        # a function to filter out data we dont need from raw ims data
        pass

    @abstractmethod
    def extract_and_save(self):
        # this function is the funtion that gets called externally
        # we can have this function as a ray method to help with distributed execution
        pass
