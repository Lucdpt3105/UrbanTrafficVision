"""
Color palette for detection categories.
All colors in BGR format for OpenCV.
"""

# Primary category colors
COLORS = {
    "vehicle":        (0,   200, 255),   # amber-orange
    "person":         (0,   255, 128),   # bright green
    "traffic_light":  (255,  80, 200),   # magenta-pink
    "obstacle":       (100, 100, 255),   # soft red-pink
    "default":        (200, 200, 200),   # grey
}

# Traffic light state colors
TRAFFIC_STATE_COLORS = {
    "RED":    (0,   0,   220),
    "YELLOW": (0,   200, 240),
    "GREEN":  (0,   220,   0),
    "UNKNOWN":(150, 150, 150),
}

# Per-vehicle-type accent (for bbox edge)
VEHICLE_SUBTYPE_COLORS = {
    "Sedan":                (0, 215, 255),
    "Hatchback":            (0, 200, 240),
    "SUV":                  (0, 180, 255),
    "Coupe":                (0, 160, 220),
    "Crossover":            (0, 200, 210),
    "Sports Car":           (0, 140, 255),
    "Convertible":          (0, 120, 200),
    "Wagon":                (0, 190, 250),
    "Minivan":              (0, 160, 200),
    "Electric Car":         (0, 255, 180),
    "Van / MPV":            (0, 180, 200),
    "Pickup Truck":         (40, 180, 255),
    "Box Truck":            (60, 160, 230),
    "Dump Truck":           (80, 140, 210),
    "Flatbed Truck":        (60, 180, 220),
    "Semi-Truck / Container":(20, 120, 255),
    "Tanker Truck":         (30, 140, 230),
    "Garbage Truck":        (70, 150, 200),
    "City Bus":             (255, 160,  0),
    "Double-Decker Bus":    (255, 140, 20),
    "Minibus":              (255, 180, 40),
    "School Bus":           (0,  200, 255),
    "Tourist Bus":          (255, 170, 10),
    "Electric Bus":         (0,  230, 200),
    "Motorcycle":           (130, 255, 80),
    "Scooter":              (100, 240, 80),
    "Sports Bike":          (80,  255, 60),
    "Cruiser":              (110, 230, 70),
    "Electric Scooter":     (60,  255, 100),
    "Bicycle":              (180, 255, 100),
    "Mountain Bike":        (160, 240, 100),
    "Road Bike":            (140, 255, 120),
    "Electric Bicycle":     (130, 240, 150),
    "Aeroplane":            (200, 200, 200),
    "Train":                (200, 180, 160),
    "Boat / Watercraft":    (255, 200, 100),
    "Driver":               (60,  255,  60),
    "Pedestrian":           (80,  220, 80),
}
