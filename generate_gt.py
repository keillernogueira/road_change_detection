import os
import overpass
import geojson
import json
import imageio
import numpy as np
import logging
import argparse

from skimage.morphology import dilation, square, disk
from shapely.geometry import shape, MultiLineString, MultiPolygon
from osgeo import ogr, gdal, gdalconst

from utils import str2bool


# python generate_gt.py --area_json data\\area2.geojson --area_tif C:\\Users\\keill\\Desktop\\Datasets\\road_detection\\analise\\area2_landsat7_sr_2002.tif --gt_file C:\\Users\\keill\\Desktop\\teste.png


def open_json(path):
    with open(path) as f:
        gj = json.load(f)
    return gj


def save_json(gj, path):
    with open(path, 'w') as f:
        json.dump(gj, f)


def open_geojson(path):
    with open(path) as f:
        gj = geojson.load(f)
    return gj


def save_geojson(gj, path):
    with open(path, 'w') as f:
        geojson.dump(gj, f)


def min_max_coordinates(gj):
    log = []
    lat = []
    for feat in gj['features'][0]['geometry']['coordinates'][0][0]:
        log.append(feat[0])
        lat.append(feat[1])
    return (min(log), min(lat)), (max(log), max(lat))


def query_osm(min_c, max_c):
    api = overpass.API()
    query = '(way["highway"](' + str(min_c[1]) + ', ' + str(min_c[0]) + ', ' +\
            str(max_c[1]) + ', ' + str(max_c[0]) + '););out ids geom;'
    response = api.get(query)  # , responseformat="json")
    # print(response)
    return response


def calculate_intersection(gj, gj1):
    g1 = MultiPolygon([shape(feature["geometry"]).buffer(0) for feature in gj['features']])
    g2 = MultiLineString([shape(feature["geometry"])
                          for feature in gj1['features'] if feature['geometry']['coordinates']])
    return g1.intersection(g2)


def save_shapely(data, output_file):
    # Now convert it to a shapefile with OGR
    driver = ogr.GetDriverByName("GeoJSON")
    # srs = osr.SpatialReference()
    # srs.ImportFromEPSG(31984)

    ds = driver.CreateDataSource(output_file)
    layer = ds.CreateLayer('', None, ogr.wkbMultiLineString)
    # Add one attribute
    layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
    defn = layer.GetLayerDefn()

    # for i, feat in enumerate(data):
    # Create a new feature (attribute and geometry)
    feat = ogr.Feature(defn)
    feat.SetField('id', 1)

    # Make a geometry, from Shapely object
    geom = ogr.CreateGeometryFromWkb(data.wkb)
    feat.SetGeometry(geom)

    layer.CreateFeature(feat)
    feat = geom = None  # destroy these

    # Save and close everything
    ds = layer = feat = geom = None


# gdal_rasterize -l area2_lines -burn 1.0 -ts 2792.0 1072.0 -a_nodata 0.0
# -te -54.866797645978856 -9.470560761817564 -54.61611971034132 -9.374296922444158
def rasterize_vector(vector_path, area_tif, output_file):
    # Open area image
    area_img = gdal.Open(area_tif, gdal.GA_ReadOnly)
    # print(area_img.RasterXSize, area_img.RasterYSize)

    geo_transform = area_img.GetGeoTransform()
    x_min = geo_transform[0]
    y_max = geo_transform[3]
    x_max = x_min + geo_transform[1] * area_img.RasterXSize
    y_min = y_max + geo_transform[5] * area_img.RasterYSize
    pixel_width = geo_transform[1]

    # Open area shapefile
    # area_shapefile = ogr.Open(area_json)
    # area_shapefile_layer = area_shapefile.GetLayer()
    # x_min, x_max, y_min, y_max = area_shapefile_layer.GetExtent()
    # print(x_min, x_max, y_min, y_max)

    # Open shapefile
    shapefile = ogr.Open(vector_path)
    shapefile_layer = shapefile.GetLayer()

    # # v1
    # ds = gdal.Rasterize(output_file, vector_path,  # layers=vector_path,
    #                     width=area_img.RasterXSize, height=area_img.RasterYSize,
    #                     burnValues=1, noData=0, outputBounds=[x_min, y_min, x_max, y_max],
    #                     outputType=gdal.GDT_CFloat32)
    # ds = None

    # v2
    # Rasterise
    output = gdal.GetDriverByName('GTiff').Create(output_file, area_img.RasterXSize, area_img.RasterYSize,
                                                  1, gdal.GDT_Byte)
    # print(area_img.GetProjectionRef())
    # print(area_img.GetGeoTransform())
    # output.SetProjection(area_img.GetProjectionRef())
    # output.SetGeoTransform(area_img.GetGeoTransform())
    output.SetGeoTransform((x_min, pixel_width, 0, y_min, 0, pixel_width))

    # Write data to band 1
    band = output.GetRasterBand(1)
    band.SetNoDataValue(0)
    band.FlushCache()
    gdal.RasterizeLayer(output, [1], shapefile_layer, burn_values=[1])

    # Close datasets
    band = None
    output = None
    image = None
    shapefile = None


# deprecated
def rasterize_vector2(vector_path, area_json, area_tif, output_file):
    # ndsm = '/home/zeito/pyqgis_data/utah_demUTM2.tif'
    # shp = '/home/zeito/pyqgis_data/polygon8.shp'
    # output = '/home/zeito/pyqgis_data/my.tif'

    data = gdal.Open(area_tif, gdalconst.GA_ReadOnly)
    geo_transform = data.GetGeoTransform()
    # source_layer = data.GetLayer()
    x_min = geo_transform[0]
    y_max = geo_transform[3]
    x_max = x_min + geo_transform[1] * data.RasterXSize
    y_min = y_max + geo_transform[5] * data.RasterYSize
    x_res = data.RasterXSize
    y_res = data.RasterYSize

    mb_v = ogr.Open(vector_path)
    mb_l = mb_v.GetLayer()
    pixel_width = geo_transform[1]

    target_ds = gdal.GetDriverByName('GTiff').Create(output_file, x_res, y_res, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform((x_min, pixel_width, 0, y_min, 0, pixel_width))
    band = target_ds.GetRasterBand(1)
    NoData_value = 0
    band.SetNoDataValue(NoData_value)
    band.FlushCache()
    gdal.RasterizeLayer(target_ds, [1], mb_l, options=["ATTRIBUTE=hedgerow"])

    target_ds = None


def dilate_gt(input_path, output_path, save_readable=False):
    img = imageio.imread(input_path)
    print('before dilation', img.shape, np.bincount(img.astype(int).flatten()))

    dil_out = dilation(img, disk(3))
    print('after dilation', img.shape, np.bincount(dil_out.astype(int).flatten()))

    imageio.imwrite(output_path, dil_out)
    if save_readable:
        imageio.imwrite(os.path.splitext(output_path)[0] + '_readable.png', dil_out * 255)


def main():
    parser = argparse.ArgumentParser(description='generate_gt.py')

    parser.add_argument('--area_json', type=str, required=True,
                        help='Json/GeoJson file defining the area in which we want to get the ground truth.')
    parser.add_argument('--area_tif', type=str, required=True,
                        help='Tif file defining the area in which we want to get the ground truth.')
    parser.add_argument('--gt_file', type=str, required=True,
                        help='File path to save the ground truth (png format only).')
    parser.add_argument('--save_osm_response', type=str2bool, required=False, default=False,
                        help='Save OpenStreetMap response')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.info(args)

    # opening the json/geojson area
    gj = open_geojson(args.area_json)
    # gj = open_json(args.area_json)

    # defining the bounding box coordinates
    min_c, max_c = min_max_coordinates(gj)

    # calling the overpass to get all highways inside the bounding box
    response = query_osm(min_c, max_c)

    if args.save_osm_response:
        save_geojson(response, os.path.join(os.path.dirname(args.gt_file),
                                            os.path.basename(args.area_json).split('.')[0] + "_osm_response.geojson"))
        # save_json(response, "C:\\Users\\keill\\Desktop\\test.geojson")

    # calculate the intersection between the areas
    intersect = calculate_intersection(gj, response)
    save_shapely(intersect, args.gt_file.replace('.png', '.geojson'))

    rasterize_vector(args.gt_file.replace('.png', '.geojson'), args.area_tif, args.gt_file.replace('.png', '.tif'))
    dilate_gt(args.gt_file.replace('.png', '.tif'), args.gt_file, save_readable=True)


if __name__ == '__main__':
    main()
