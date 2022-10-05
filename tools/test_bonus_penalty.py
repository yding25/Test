# compute penalty
result = [37, 21, 22, 136, 108, 112, 116, 120, 124, 128, 132, 131, 135, 107, 53, 57, 119, 123, 32, 17, 148, 147, 55,
          109,
          110, 138, 24, 84, 83, 139, 47]

school_lane = 17
gas1_lane = 107
gas2_lane = 60
grocery1_lane = 72
grocery2_lane = 138

# which gas station is visted?
if gas1_lane in result:
    gas_lane = gas1_lane
else:
    gas_lane = gas2_lane
# which grocery store is visted?
if grocery1_lane in result:
    grocery_lane = grocery1_lane
else:
    grocery_lane = grocery2_lane
gas_index = result.index(gas_lane)
grocery_index = result.index(grocery_lane)
school_index = result.index(school_lane)

