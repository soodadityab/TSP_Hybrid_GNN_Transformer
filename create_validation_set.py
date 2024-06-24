from data import generate_tsp_instances_validation
import pickle

val_set_len = 2000
num_cities = 10
validation_set = generate_tsp_instances_validation(val_set_len, num_cities)

with open("validation_set.pkl", "wb") as f:
  pickle.dump(validation_set, f)
print(f"Generated and saved validation set with {val_set_len} instances")
