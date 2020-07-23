# Import the Earth Engine Python Package
import ee
import pandas as pd
import numpy as np
import time
import configparser
from gee.utils import init_log
from gee.config import *

import os

# Start logging
logging = init_log(".")
_print = logging.info

# Para mudar de conta:
# earthengine authenticate

# Initialize the Earth Engine object, using the authentication credentials.
# ee.Initialize()


def degree_conv(var):
    data = var.split("°", 1)
    if data[0][0] == '-':
        d = data[0][1:]
        negative = True
    else:
        d = data[0]
        negative = False
    data = data[1].split("'", 1)
    minutes = data[0]
    data = data[1].split('"', 2)
    sec = data[0]

    # DMS to decimal degrees converter
    if negative:
        dd = (int(d) + (float(minutes) / 60) + (float(sec) / 3600)) * -1
    else:
        dd = int(d) + (float(minutes) / 60) + (float(sec) / 3600)
    return round(dd, 6)


def send_task(satellite_name, satellite, bands, x, y, day, flag_clouds, mask, dam):
    llx = x - 0.0172
    lly = y - 0.0172
    urx = x + 0.0172
    ury = y + 0.0172

    geometry = [[llx, lly], [llx, ury], [urx, ury], [urx, lly]]

    cloudy_percentage = 30

    # The image input data is cloud-masked median composite.
    dataset = satellite.filterDate(day[0], day[1]).filter(ee.Filter.lte(flag_clouds, cloudy_percentage)).map(
        mask).filterBounds(ee.Geometry.Polygon(geometry))

    while (dataset.size().getInfo() == 0) and (cloudy_percentage < 100):
        _print("Do not exists images in this dataset using cloudy percentage equal to {}%".format(cloudy_percentage))
        cloudy_percentage = cloudy_percentage + 10
        _print("Increasing cloudy_percentage to {}%".format(cloudy_percentage))
        dataset = satellite.filterDate(day[0], day[1]).filter(ee.Filter.lte(flag_clouds, cloudy_percentage)).map(
            mask).filterBounds(ee.Geometry.Polygon(geometry))

    image = dataset.median()

    task = ee.batch.Export.image.toDrive(image=image.toFloat(),
                                         description="{0:0=3d}_{1}".format(dam + 1, day[0][:4]),
                                         folder=day[0][:4] + "_" + satellite_name,
                                         region=geometry,
                                         scale=10,
                                         shardSize=384,
                                         fileDimensions=(384, 384),
                                         fileFormat='GeoTIFF')

    _print("{0:0=3d}_{1}".format(dam + 1, day[0][:4]))
    task.start()


# def send_not_dam_task(x, y, bands, satellite, flag_clouds, mask, data, string, satellite_name):
#     if string == "N":
#         new_y = round(y + 0.0172 * 2, 6)
#     else:
#         new_y = round(y - 0.0172 * 2, 6)
#
#     llx = x - 0.0172
#     lly = new_y - 0.0172
#     urx = x + 0.0172
#     ury = new_y + 0.0172
#
#     geometry = [[llx, lly], [llx, ury], [urx, ury], [urx, lly]]
#
#     cloudy_percentage = 30
#
#     # The image input data is cloud-masked median composite.
#     dataset = satellite.filterDate(day[0], day[1]).filter(ee.Filter.lte(flag_clouds, cloudy_percentage)).map(
#         mask).filterBounds(ee.Geometry.Polygon(geometry))
#
#     while (dataset.size().getInfo() == 0) and (cloudy_percentage < 100):
#         _print("Do not exists images in this dataset using cloudy percentage equal to {}%".format(cloudy_percentage))
#         cloudy_percentage = cloudy_percentage + 10
#         _print("Increasing cloudy_percentage to {}%".format(cloudy_percentage))
#         dataset = satellite.filterDate(day[0], day[1]).filter(ee.Filter.lte(flag_clouds, cloudy_percentage)).map(
#             mask).filterBounds(ee.Geometry.Polygon(geometry))
#
#     image = dataset.median()
#
#     check_intersection = False
#     for check in range(data.shape[0]):
#         if check != dam:
#             x_, y_ = degree_conv(data[check][1]), degree_conv(data[check][0])
#             if (llx < x_ < urx) and (lly < y_ < ury):
#                 check_intersection = True
#
#     if not check_intersection:
#         task = ee.batch.Export.image.toDrive(image=image.toFloat(),
#                                              description="{0}{1:0=3d}_{2}".format(string, dam + 1, day[0][:4]),
#                                              folder=day[0][:4] + "_NOT_DAM_" + satellite_name,
#                                              region=geometry,
#                                              scale=10,
#                                              shardSize=384,
#                                              fileDimensions=(384, 384),
#                                              fileFormat='GeoTIFF')
#
#         _print("{0}{1:0=3d}_{2}".format(string, dam + 1, day[0][:4]))
#         task.start()


def main():
    # Reading data
    data = pd.read_csv(input_file, sep='\t', usecols=[column_latitude, column_longitude])
    data = data.values

    for satellite_name in satellites:
        for day in dates:
            for dam in range(data.shape[0]):
                ee.Initialize()

                if satellite_name == "sentinel":
                    # Use these bands for prediction.
                    bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12']

                    # Use Sentinel 2 surface reflectance data.
                    satellite = ee.ImageCollection("COPERNICUS/S2")

                    flag_clouds = 'CLOUDY_PIXEL_PERCENTAGE'

                    def maskS2clouds(image):
                        cloudShadowBitMask = ee.Number(2).pow(3).int()
                        cloudsBitMask = ee.Number(2).pow(5).int()
                        qa = image.select('QA60')
                        mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0).And(
                            qa.bitwiseAnd(cloudsBitMask).eq(0))
                        return image.updateMask(mask).select(bands).divide(10000)

                    mask = maskS2clouds
                elif satellite_name == "landsat8":
                    # Use these bands for prediction.
                    bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11']

                    # Use Sentinel 2 surface reflectance data.
                    satellite = ee.ImageCollection("LANDSAT/LC08/C01/T1_TOA")

                    flag_clouds = 'CLOUD_COVER'

                    # Cloud masking function.
                    def maskL8sr(image):
                        cloudsBitMask = ee.Number(2).pow(4).int()
                        qa = image.select('BQA')
                        mask = qa.bitwiseAnd(cloudsBitMask).eq(0)
                        return image.updateMask(mask).select(bands).divide(10000)

                    mask = maskL8sr
                else:
                    raise NotImplementedError

                # Logitude, Latitude
                x, y = degree_conv(data[dam][1]), degree_conv(data[dam][0])
                # x, y = float(data[dam][1].replace(",", ".")), float(data[dam][0].replace(",", "."))

                # if not_dam:
                #     # North
                #     try:
                #         send_not_dam_task(x, y, bands, satellite, flag_clouds, mask, data, "N", satellite_name)
                #     except:
                #         _print("Error in {0}{1:0=3d}_{2}".format("N", dam + 1, day[0][:4]))
                #         try:
                #             _print("Trying image {0}{1:0=3d}_{2} again".format("N", dam + 1, day[0][:4]))
                #             send_not_dam_task(x, y, bands, satellite, flag_clouds, mask, data, "N", satellite_name)
                #         except:
                #             _print("Image {0}{1:0=3d}_{2} could not be downloaded".format("N", dam + 1, day[0][:4]))
                #
                #     # South
                #     try:
                #         send_not_dam_task(x, y, bands, satellite, flag_clouds, mask, data, "S", satellite_name)
                #     except:
                #         _print("Error in {0}{1:0=3d}_{2}".format("S", dam + 1, day[0][:4]))
                #         try:
                #             _print("Trying image {0}{1:0=3d}_{2} again".format("S", dam + 1, day[0][:4]))
                #             send_not_dam_task(x, y, bands, satellite, flag_clouds, mask, data, "S", satellite_name)
                #         except:
                #             _print("Image {0}{1:0=3d}_{2} could not be downloaded".format("S", dam + 1, day[0][:4]))
                #
                # else:
                # Dam
                try:
                    send_task(satellite_name, satellite, bands, x, y, day, flag_clouds, mask, dam)
                except:
                    _print("Error in {0:0=3d}_{1}".format(dam + 1, day[0][:4]))
                    try:
                        _print("Trying image {0:0=3d}_{1} again".format(dam + 1, day[0][:4]))
                        send_task(satellite_name, satellite, bands, x, y, day, flag_clouds, mask, dam)
                    except:
                        _print("Image {0:0=3d}_{1} could not be downloaded".format(dam + 1, day[0][:4]))


if __name__ == '__main__':
    main()
