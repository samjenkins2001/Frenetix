import os
import sys
import glob


mod_path = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))
sys.path.append(mod_path)
stack_path = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))

scenarios_folder = os.path.join(stack_path, "commonroad-scenarios", "scenarios")
files = sorted(glob.glob(scenarios_folder + "/*"))

print("DONE")
