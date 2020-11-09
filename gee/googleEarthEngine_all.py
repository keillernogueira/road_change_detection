# Import the Earth Engine Python Package
import ee
import geojson
import os
import argparse
import logging

import sys
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import str2bool


'''
python googleEarthEngine_all.py --satellite landsat7_toa --shape_path ..\data\\area1.geojson --start_date 2019-01-01 --end_date 2020-01-01

# landsat8_toa 2013
python googleEarthEngine_all.py --satellite landsat8_sr --shape_path ..\data\\area2.geojson --file_name area2_landsat8_sr_2013_sync --start_date 2013-04-01 --end_date 2014-01-01
# landsat7_toa 2002
python googleEarthEngine_all.py --satellite landsat7_toa --shape_path ..\data\\area2.geojson --file_name area2_landsat7_toa_2002_pansharpen --start_date 2002-01-01 --end_date 2003-01-01 --panshapern True
'''


def main():
    parser = argparse.ArgumentParser(description='googleEarthEngine_all')

    parser.add_argument('--satellite', type=str, required=True,
                        help='Flag to define the satellite [Options: landsat[7|8]_toa || landsat[7|8]_sr || sentinel2]')
    parser.add_argument('--shape_path', type=str, required=True,
                        help='Path to the shape that is used as reference for cropping the satellite image')
    parser.add_argument('--file_name', type=str, required=True,
                        help='File name to save')
    parser.add_argument('--start_date', type=str, required=True,
                        help='First date to filter data')
    parser.add_argument('--end_date', type=str, required=True,
                        help='End date to filter data')
    parser.add_argument('--only_panchromatic', type=str2bool, required=False, default=False,
                        help='Download only Panchromatic band. Only works from landsat_toa')
    parser.add_argument('--sync_bands_L7_L8', type=str2bool, required=False, default=False,
                        help='Sync bands for L7 and L8')
    parser.add_argument('--panshapern', type=str2bool, required=False, default=False,
                        help='Panshapen L7 and L8 TOA RGB bands using panchromatic')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logging.info(args)

    if args.panshapern is True:
        assert("landsat" in args.satellite and args.satellite.split('_')[1] == 'toa')

    # Initialize the Earth Engine object, using the authentication credentials.
    ee.Initialize()

    if args.satellite == "sentinel2":
        # selected bands
        # B2, B3, B4, B8 (BGR, NIR) - > 10 meter resolution
        bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12']  # 12

        # Use Sentinel 2 surface reflectance data.
        satellite = ee.ImageCollection("COPERNICUS/S2")

        flag_clouds = 'CLOUDY_PIXEL_PERCENTAGE'

        def maskS2clouds(image):
            cloudShadowBitMask = ee.Number(2).pow(3).int()
            cloudsBitMask = ee.Number(2).pow(5).int()
            qa = image.select('QA60')
            _mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0).And(qa.bitwiseAnd(cloudsBitMask).eq(0))
            return image.updateMask(_mask).select(bands).divide(10000)

        mask = maskS2clouds
    elif "landsat" in args.satellite:
        flag_clouds = 'CLOUD_COVER'

        if args.satellite.split('_')[1] == 'toa':
            # Cloud masking function.
            def mask_func(data):
                qa = data.select('BQA')
                clouds_bitmask = (1 << 4)
                _mask = qa.bitwiseAnd(clouds_bitmask).eq(0)
                return data.updateMask(_mask).select(bands)

            mask = mask_func

            if int(args.satellite[7]) == 7:
                # Use these bands
                if args.only_panchromatic is True:
                    bands = ['B8']
                else:
                    bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6_VCID_1', 'B6_VCID_2', 'B7', 'B8']  # 9

                def pan_sharpen_panchhromatic(data):
                    rgb = data.select('B3', 'B2', 'B1')
                    pan = data.select('B8')
                    # Convert to HSV, swap in the pan band, and convert back to RGB.
                    huesat = rgb.rgbToHsv().select('hue', 'saturation')
                    upres = ee.Image.cat(huesat, pan).hsvToRgb()
                    return upres

                # Use landsat7 Top of Top of Atmosphere data
                satellite = ee.ImageCollection("LANDSAT/LE07/C01/T1_TOA")
            elif int(args.satellite[7]) == 8:
                # Use these bands for prediction.
                bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11']  # 11

                def pan_sharpen_panchhromatic(data):
                    rgb = data.select('B4', 'B3', 'B2')
                    pan = data.select('B8')
                    # Convert to HSV, swap in the pan band, and convert back to RGB.
                    huesat = rgb.rgbToHsv().select('hue', 'saturation')
                    upres = ee.Image.cat(huesat, pan).hsvToRgb()
                    return upres

                # Use landsat8 Top of Top of Atmosphere data
                satellite = ee.ImageCollection("LANDSAT/LC08/C01/T1_TOA")
            else:
                raise NotImplementedError
        elif args.satellite.split('_')[1] == 'sr':  # SURFACE REFLECTANCE
            if int(args.satellite[7]) == 7:
                # Use these bands for prediction.
                if args.sync_bands_L7_L8 is False:
                    bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']  # 7
                else:
                    bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B7']

                # Use landsat7 Surface Reflectance data
                satellite = ee.ImageCollection("LANDSAT/LE07/C01/T1_SR")

                # Cloud masking function.
                def mask_func(data):
                    qa = data.select('pixel_qa')
                    cloud = qa.bitwiseAnd(1 << 5).And(qa.bitwiseAnd(1 << 7)).Or(qa.bitwiseAnd(1 << 3))
                    _mask = data.mask().reduce(ee.Reducer.min())
                    return data.updateMask(cloud.Not()).updateMask(_mask).select(bands).divide(10000)
                mask = mask_func
            elif int(args.satellite[7]) == 8:
                # Use these bands for prediction.
                if args.sync_bands_L7_L8 is False:
                    bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10', 'B11']  # 9
                else:
                    bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']

                # Use landsat8 Surface Reflectance data
                satellite = ee.ImageCollection("LANDSAT/LC08/C01/T1_SR")

                # Cloud masking function.
                def mask_func(data):
                    qa = data.select('pixel_qa')
                    cloud_shadow_bitmask = (1 << 3)
                    clouds_bitmask = (1 << 5)
                    _mask = qa.bitwiseAnd(cloud_shadow_bitmask).eq(0).And(qa.bitwiseAnd(clouds_bitmask).eq(0))
                    return data.updateMask(_mask).select(bands).divide(10000)
                mask = mask_func
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    with open(os.path.join(args.shape_path), encoding='utf-8') as f:
        gj = geojson.load(f)
        for i, polygon in enumerate(gj['features'][0]["geometry"]["coordinates"]):
            geometry = ee.Geometry.Polygon(polygon)

            # file_name = os.path.splitext(os.path.basename(args.shape_path))[0] + '_' + args.satellite
            # if args.only_panchromatic is True:
            #     file_name = file_name + '_panchromatic'
            # logging.info(file_name)

            if args.panshapern is False:
                _image = satellite.filterDate(args.start_date, args.end_date). \
                    filter(ee.Filter.lte(flag_clouds, 30)).map(mask).median()
                _scale = 15
            else:
                _image = satellite.filterDate(args.start_date, args.end_date).\
                    filter(ee.Filter.lte(flag_clouds, 30)).map(mask).map(pan_sharpen_panchhromatic).median()  # .float()
                _scale = 15
            task = ee.batch.Export.image.toDrive(image=_image.clip(geometry),
                                                 description=args.file_name,
                                                 folder='road_detection_dataset',
                                                 region=polygon,
                                                 scale=_scale,  # original 10
                                                 fileFormat='GeoTIFF',
                                                 skipEmptyTiles=True,
                                                 maxPixels=1e13)
            task.start()

    logging.info("End!")


if __name__ == '__main__':
    main()
