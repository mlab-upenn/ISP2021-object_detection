import sys
sys.path.append(".") # Adds higher directory to python modules path.

from perception.helper import Helper
import os
with open('maps/Example/config_example_map.yaml') as file:
    conf_dict = yaml.load(file, Loader=yaml.FullLoader)
