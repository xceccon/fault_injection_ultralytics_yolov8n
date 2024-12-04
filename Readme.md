# Fault Injection in Ultralytics YOLO
Model: yolov8n
Fault Injection type: a single bitflip in the weights file
We found two bitflips which can break the model: 


## Results
Detected Classes in image 1: 1 motorcycle, 14 buss, 45 trains, 152 traffic lights, 8 suitcases, 16 sports balls, 55 bottles, 6 cups, 3 oranges

Detected Classes in image 2: 3 bicycles, 17 trains, 236 traffic lights, 6 sports balls, 29 bottles, 1 wine glass, 7 cups, 1 bowl


![alt text](assets/consti.jpg)
![alt text](assets/gernika.jpg)

## How to use the repo
First navigate to the find_faulty_bitflips folder 
"""
cd find_faulty_bitflips
"""
then run search_bitflip.py

"""
python search_bitflip.py
"""

The program will create a bitflip_tests folder (which is added to the gititgnore) and save the bitflipped models as well as a json file containing the tests.

After a few hours, you can try finding bitflips in the json by running find_bitflips_in_the_json.py
"""
find_bitflips_in_the_json.py
"""

