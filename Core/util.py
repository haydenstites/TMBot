# Conversion util functions

def norm_float(value : float, min : float, max : float) -> float:
    value = float(value)
    return ((value - min) / (max - min)) * 2 - 1 # Scale value proportionally between min and max

def binary_strbool(value : str) -> int:
    if value == "true":
        return 1
    elif value == "false":
        return 0
    raise ValueError

# None, tech, plastic, dirt, grass, ice, other
#   0     1      2      3      4     5     6
mat_idx = {
    "Asphalt" : 1,
    "Concrete" : 1,
    "Pavement" : 1,
    "Grass" : 4,
    "Ice" : 5,
    "Metal" : 1,
    "Sand" : 4,
    "Dirt" : 3,
    "DirtRoad" : 3,
    "Rubber" : 2,
    "SlidingRubber" : 2,
    "Water" : 2,
    "WetDirtRoad" : 3,
    "WetAsphalt" : 1,
    "WetPavement" : 1,
    "WetGrass" : 4,
    "Snow" : 4,
    "Tech" : 1,
    "TechHook" : 1,
    "TechGround" : 1,
    "TechWall" : 1,
    "TechArrow" : 1,
    "RoadIce" : 5,
    "Green" : 4,
    "Plastic" : 2,
    "XXX_Null" : 0,
}

def mat_index(value : str) -> int:
    try:
        return mat_idx[value]
    except:
        print(value, "is not an assigned surface")
        return 6 # Other

# None, Playing, Finish
#   0      1       2
race_idx = {
    "Playing" : 1,
    "Finish" : 2,
}

def race_index(value : str) -> int:
    try:
        return race_idx[value]
    except:
        return 0 # Other
