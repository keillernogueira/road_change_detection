# Import the Earth Engine Python Package
import ee
# import pandas as pd
import numpy as np
import geojson
import time
import os

list_states = ["MG"]

for state in list_states:

    path = os.path.join("/home/users/matheusb/datasets/shapefiles_brazil_cities/data/", state)

    # Para mudar de conta:
    # earthengine authenticate

    # Initialize the Earth Engine object, using the authentication credentials.
    ee.Initialize()

    # Use these bands for prediction.
    bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12']

    # Use Sentinel 2 surface reflectance data.
    sentinel = ee.ImageCollection("COPERNICUS/S2")


    # Cloud masking function.
    def maskL8sr(image):
        cloudShadowBitMask = ee.Number(2).pow(3).int()
        cloudsBitMask = ee.Number(2).pow(5).int()
        qa = image.select('pixel_qa')
        mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0).And(qa.bitwiseAnd(cloudsBitMask).eq(0))
        return image.updateMask(mask).select(bands).divide(10000)


    def maskS2clouds(image):
        cloudShadowBitMask = ee.Number(2).pow(3).int()
        cloudsBitMask = ee.Number(2).pow(5).int()
        qa = image.select('QA60')
        mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0).And(qa.bitwiseAnd(cloudsBitMask).eq(0))
        return image.updateMask(mask).select(bands).divide(10000)


    def utf8(names_cities):
        for i in range(names_cities.shape[0]):
            names_cities[i][0] = names_cities[i][0].replace("ô", "o")
            names_cities[i][0] = names_cities[i][0].replace("õ", "o")
            names_cities[i][0] = names_cities[i][0].replace("í", "i")
            names_cities[i][0] = names_cities[i][0].replace("ó", "o")
            names_cities[i][0] = names_cities[i][0].replace("ç", "c")
            names_cities[i][0] = names_cities[i][0].replace("ú", "u")
            names_cities[i][0] = names_cities[i][0].replace(" ", "_")
            names_cities[i][0] = names_cities[i][0].replace("á", "a")
            names_cities[i][0] = names_cities[i][0].replace("ã", "a")
            names_cities[i][0] = names_cities[i][0].replace("â", "a")
            names_cities[i][0] = names_cities[i][0].replace("Ó", "O")
            names_cities[i][0] = names_cities[i][0].replace("ü", "u")
            names_cities[i][0] = names_cities[i][0].replace("É", "E")
            names_cities[i][0] = names_cities[i][0].replace("é", "e")
            names_cities[i][0] = names_cities[i][0].replace("\'", "")
            names_cities[i][0] = names_cities[i][0].replace("'", "")
            names_cities[i][0] = names_cities[i][0].upper()
            names_cities[i][0] = names_cities[i][0] + ".json"

        return names_cities


    # The image input data is cloud-masked median composite.
    image = sentinel.filterDate('2019-01-01', '2020-01-01').filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 30)).map(
        maskS2clouds).median()  # .filterBounds(ee.Geometry.Polygon(geometry))
    # image = sentinel.filterDate('2019-01-01','2020-01-01').filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 30)).select(bands).median().divide(10000)

    # names_cities_metropolitan = pd.read_csv("/home/datasets/shapefiles_brazil_cities/metropolitanAreaBHNames.tsv")
    # names_cities_metropolitan = utf8(names_cities_metropolitan.values)
    # names_cities_metropolitan = set(names_cities_metropolitan.flatten())
    # # print(names_cities_metropolitan)

    names_cities = np.array(["IBIA.json"])  # np.sort(np.array(os.listdir(path)))

    print(len(names_cities))
    # names_cities = set(names_cities.flatten())

    # diff = names_cities - names_cities_metropolitan

    # diff = np.array(list(diff))

    # diff.sort()

    # diff_0 = diff[:18]
    # diff_1 = diff[19:36]
    # diff_2 = diff[37:54]
    # diff_3 = diff[55:72]
    # diff_4 = diff[73:90]
    # diff_5 = diff[91:108]
    # diff_6 = diff[109:126]
    # diff_7 = diff[127:144]

    exception_list = []
    count = 1

    for name in names_cities:

        if count % 10 == 0:
            print("waiting...")
            time.sleep(10 * 60)
        count += 1

        with open(os.path.join(path, name), encoding='utf-8') as f:
            gj = geojson.load(f)

            pol_num = 0
            for polygon in gj['features']["geometry"]["coordinates"]:
                pol_num += 1

                geometry = ee.Geometry.Polygon(polygon)
                region = polygon

                file_name = name[:-5].replace("'", "") + "_" + str(pol_num)

                task = ee.batch.Export.image.toDrive(image=image.clip(geometry),
                                                     description=file_name,
                                                     folder=state,
                                                     region=region,
                                                     scale=10,
                                                     fileFormat='GeoTIFF',
                                                     skipEmptyTiles=True,
                                                     maxPixels=1e13)

                print(file_name)

                try:
                    # Send the task for the engine.
                    task.start()
                except:
                    exception_list.append(task)
                    print("size exception_list: ", len(exception_list))

    print("writing tasks list in a file...")
    f = open("initial_list_problem.txt", "w+")
    for task in exception_list:
        f.write(task.__repr__() + "\n")
    f.close()

    list_round = []
    flag_wait = 0

    while len(exception_list) != 0:
        try:
            print(exception_list[0])
            exception_list[0].start()
            exception_list.pop(0)

            if flag_wait > 2:
                print("writing tasks list in a file...")
                f = open("final_list_problem.txt", "w+")
                for task in exception_list:
                    f.write(task.__repr__() + "\n")
                f.close()
                exit()

        except:
            print("using exception...")
            first = exception_list[0]
            exception_list.pop(0)
            exception_list.append(first)

            list_round.append(len(exception_list))

            if len(list_round) > 10:
                list_round.pop(0)

            if (len(set(list_round)) == 1):
                print("waiting...")
                print(list_round)
                time.sleep(60 * 60)
                flag_wait += 1

    print("script is done!")

# def degree_conv(var):
#     data = var.split("°",1)
#     if data[0][0] == '-':
#         d = data[0][1:]
#         negative = True
#     else:
#         d = data[0]
#         negative = False
#     data = data[1].split("'",1)
#     minutes = data[0]
#     data = data[1].split('"',2)
#     sec =  data[0]

#     #DMS to decimal degrees converter
#     if negative:
#         dd = (int(d) + (float(minutes)/60) + (float(sec)/3600)) * -1
#     else:
#         dd = int(d) + (float(minutes)/60) + (float(sec)/3600)
#     return round(dd, 6)


# data = pd.read_csv('classificacaoBarragens.csv', sep= ';', usecols=[0, 2, 3])
# # data = pd.read_csv('../datasets/parques_nacionais.csv', sep= ';', usecols=[0, 1, 2])

# data = data.values


# # Monitor the task.
# while task.status()['state'] in ['READY', 'RUNNING']:
#     print(task.status())
#     time.sleep(10)
# else:
#     print(task.status())

# # # with open('kml-brasil-master/lib/2010/municipios/MG/geojson/NOVA_LIMA.geojson', encoding='utf-8') as f:
# # #     gj = geojson.load(f)
# # # geometry = ee.Geometry.Polygon([gj['features'][0]['geometry']['coordinates']])


# for barragem in range(data.shape[0]):
# 	# Logitude, Latitude 

# 	x, y = degree_conv(data[barragem][2]), degree_conv(data[barragem][1])
# 	x, y = degree_conv("51°11\'28.95\""), degree_conv("35°45\'14.21\"")
# 	# y = round(y + 0.021395*2, 6)

# 	llx = x - 0.02785  #0.1114 
# 	lly = y - 0.021395 #0.08558 
# 	urx = x + 0.02785  #0.1114 
# 	ury = y + 0.021395 #0.08558 


# 	geometry = [[llx,lly], [llx,ury], [urx,ury], [urx,lly]]

# 	task_config = {
# 	    'scale':  10 ,
# 	    'region': geometry
# 	    }


# task = ee.batch.Export.image(image, "imagem_gabriel", task_config) #"{0:0=3d}".format(barragem+1)
# task.start()


# print("DONE!")


# for i in range(data.shape[0]):
#     data[i][0] = data[i][0].replace("ô", "o")
#     data[i][0] = data[i][0].replace("õ", "o")
#     data[i][0] = data[i][0].replace("í", "i")
#     data[i][0] = data[i][0].replace("ó", "o")
#     data[i][0] = data[i][0].replace("ç", "c")
#     data[i][0] = data[i][0].replace("ú", "u")
#     data[i][0] = data[i][0].replace("-", " ")
#     data[i][0] = data[i][0].replace("á", "a")
#     data[i][0] = data[i][0].replace("ã", "a")
#     data[i][0] = data[i][0].replace("Ó", "O")
#     data[i][0] = data[i][0].replace("ü", "u")


# x, y = data[barragem][2], data[barragem][1]

# name = data[barragem][0]
# name = name.lower().replace(" ", "_").replace("-", "_").replace("/", "_")


# check_intersection = False
# for check in range(data.shape[0]):
# 	if check != barragem:
# 		x_, y_ = degree_conv(data[check][2]), degree_conv(data[check][1])
# 		if (llx < x_ < urx) and (lly < y_ < ury):
# 			check_intersection = True

# if not check_intersection:
# print("N" + "{0:0=3d}".format(barragem+1))
