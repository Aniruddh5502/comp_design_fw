Previos run simulation scripts
[map_parameters](map_parameter.py)      # maps parameters form names to Ansys P numbers
[script_datagen](scripts_datagen.py)    # used to generate data from user preference
[user_input_sim](suer_input_sim.py)     # has run_sim() to run ansys simulations

# For Data analysis
Here we shall work updating the processing pipeline and model training and further

## Data cleaning
- Remove empty values and np.nan values.
- Remove designpoints with large deflections
- Seperate the columns that will be used to train and test models.
