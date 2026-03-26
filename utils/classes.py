"""
COCO class names (80 classes) - for YOLOv8
"""
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "aeroplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "sofa", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair dryer", "toothbrush"
]

# Detailed vehicle sub-types (used by classifier fallback heuristic)
VEHICLE_SUBTYPES = {
    "car": [
        "Sedan", "Hatchback", "SUV", "Coupe", "Crossover", "Sports Car",
        "Convertible", "Wagon", "Minivan", "Electric Car"
    ],
    "truck": [
        "Pickup Truck", "Box Truck", "Dump Truck", "Flatbed Truck",
        "Semi-Truck / Container", "Tanker Truck", "Garbage Truck"
    ],
    "bus": [
        "City Bus", "Double-Decker Bus", "Minibus", "School Bus",
        "Tourist Bus", "Electric Bus"
    ],
    "motorcycle": [
        "Motorcycle", "Scooter", "Sports Bike", "Cruiser", "Electric Scooter"
    ],
    "bicycle": [
        "Bicycle", "Mountain Bike", "Road Bike", "Electric Bicycle"
    ],
    "aeroplane": ["Aeroplane"],
    "train":    ["Train"],
    "boat":     ["Boat / Watercraft"],
}

# Special vehicles by aspect ratio heuristic
SPECIAL_LABEL_BY_RATIO = {
    "car":   [(2.0, "Van / MPV"), (1.6, "SUV"), (0.0, "Sedan")],
    "truck": [(3.0, "Semi-Truck / Container"), (2.0, "Box Truck"), (0.0, "Pickup Truck")],
    "bus":   [(2.5, "City Bus"), (0.0, "Minibus")],
}

# Which COCO classes belong to "vehicle" group
VEHICLE_CLASSES = {"bicycle", "car", "motorcycle", "aeroplane", "bus", "train", "truck", "boat"}

# Which COCO classes are "obstacle" (besides vehicles and persons and traffic lights)
OBSTACLE_CLASSES = {
    "fire hydrant", "stop sign", "parking meter", "bench",
    "backpack", "umbrella", "suitcase", "chair", "sofa", "bed",
    "dining table", "potted plant", "vase", "clock", "scissors"
}

# Traffic light class name
TRAFFIC_LIGHT_CLASS = "traffic light"
PERSON_CLASS = "person"
