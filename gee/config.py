import os 

# File with the desired coordinates separated by tab
input_file = "../barragem_2019_novas_coordenadas.tsv"

# Column in the file where the latitude is located
column_latitude = 3

# Column in the file where the longitude is located
column_longitude = 4

# List of satellites to be used
satellites = ["sentinel", "landsat8"]

# List of dates
dates = [['2016-01-01', '2016-12-31'], ['2017-01-01', '2017-12-31'],
         ['2018-01-01', '2018-12-31'], ['2019-01-01', '2019-12-31']]

# Extract dams and/or non-dams
not_dam_list = [False]
