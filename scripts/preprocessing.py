import geotable
import os
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
#import earthpy as et
import shapefile
#from mpl_toolkits.basemap import Basemap
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.pyplot as plt
from osgeo import gdal
import glob
import os
import time
import zipfile
from rasterio.plot import show
from rasterio.plot import show_hist
from rasterio.mask import mask
from shapely.geometry import box
import geopandas as gpd
from fiona.crs import from_epsg
import pycrs
from osgeo import gdal, osr
import cv2


def main():

	#Transformando kmz a zip, descomprimir el zip
	# files_kmz=glob.glob("input/kmz/*.kmz")
	# for m in files_kmz:
	# 	try:
	# 		transform_kmz_to_shp(m)
	# 		time.sleep(12)
	# 	except:
	# 		print(f'Error {m}')
	# for m in files_kmz:
	# 	filename=m.split('/')[-1].split('.')[0]
	# 	with zipfile.ZipFile(os.getcwd()+'/output/shp/'+filename+'.zip', 'r') as zip_ref:
	# 	    zip_ref.extractall(os.getcwd()+'/output/shp/')

	#Reproyectando los tif para poder hacer el recorte utilizando los shapes
	files_tif=glob.glob("input/tif/*.tif")
	print(files_tif)
	for m in files_tif:
		try:
			chm_dataset = gdal.Open(m)
			filename=m.split('/')[-1].split('.')[0]
			proj=chm_dataset.GetProjection().split(',')[-1].replace('"','').replace(']','')
			if proj != '4326':
				reproject_et(m,'output/tif/'+filename+'.tif',new_crs = 'EPSG:4326')
			time.sleep(12)
		except:
			print(f'Error {m}')
	#Realizando los recortes por zona
	files_shape=glob.glob("output/shp/*.shp")
	print(files_shape)
	for m in files_shape:
		clip_raster('output/tif/NDRE.tif',m)
		clip_raster('output/tif/NDVI.tif',m)


	


#Transformar todos los archivos kmz a shapefile para mayor comodidad
def transform_kmz_to_shp(filetype_kmz):

	t = geotable.load(filetype_kmz)
	t.iloc[0]
	filename=filetype_kmz.split('/')[-1].split('.')[0]
	print('transform '+filename)
	t.save_shp('output/shp/'+filename+'.zip')

#Reproyectar todos los tif a ESPG:4326, la proyeccion de los shapefiles.
def reproject_et(inpath, outpath, new_crs):


    dst_crs = new_crs # CRS for web meractor 

    with rasterio.open(inpath) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        print(src.crs)
        print(dst_crs)
        print(src.width)
        print(src.height)
        print(*src.bounds)
        print(transform)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(outpath, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)


#Extension shapefile
def extension_shp(filetype_shp):
	sf = shapefile.Reader(filetype_shp)
	print(sf.shapeType)
	print(sf.bbox) #longitud, latitud, longitud, latitud
	return sf.bbox

#clip raster por shape
def clip_raster(fp,shape):
	
	filename=fp.split('/')[-1].split('.')[0]+'_'+shape.split('/')[-1].split('.')[0]

	out_tif='../output/clip_'+fp.split('/')[-1].split('.')[0]+'/'+filename+'.tif'
	sf = shapefile.Reader(shape)
	data = rasterio.open(fp)
	#show((data, [1,2,3]), cmap='terrain')
	minx, miny = sf.bbox[0], sf.bbox[1]
	maxx, maxy = sf.bbox[2], sf.bbox[3]
	bbox = box(minx, miny, maxx, maxy)
	print('**********+')
	print(bbox)
	geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=from_epsg(4326))
	geo = geo.to_crs(crs=data.crs.data)
	def getFeatures(gdf):
	    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
	    import json
	    return [json.loads(gdf.to_json())['features'][0]['geometry']]
	coords = getFeatures(geo)
	out_img, out_transform = mask(dataset=data, shapes=coords, crop=True)
	print(data)
	print(coords)
	out_meta = data.meta.copy()
	epsg_code = int(data.crs.data['init'][5:])
	out_meta.update({"driver": "GTiff",
	                 "height": out_img.shape[1],
	                 "width": out_img.shape[2],
	                 "transform": out_transform,
	                 "crs": pycrs.parse.from_epsg_code(epsg_code).to_proj4()})
	with rasterio.open(out_tif, "w", **out_meta) as dest:
	    dest.write(out_img)

	#clipped = rasterio.open(out_tif)
	#show((clipped, [1,2,3]), cmap='terrain')


def process_image(band_n, image):
	ds = gdal.Open(image)
	band = ds.GetRasterBand(band_n)
	myarray = np.array(band.ReadAsArray())
	if band_n ==1: 
		selection = np.logical_or(myarray >= 255, myarray <= 10)
	elif band_n ==2:
		selection = np.logical_or(myarray >= 245, myarray <= 80)

	new_array = [ [0 for i in range(band.XSize)] for j in range(band.YSize)]
	print(new_array)
	for i, item in enumerate(myarray):
	    for j, element in enumerate(item):
	        if selection[i][j] == True:
	            new_array[i][j] = myarray[i][j]
	        else:
	            new_array[i][j] = -999

	geotransform = ds.GetGeoTransform()
	wkt = ds.GetProjection()

	# Create gtif file
	driver = gdal.GetDriverByName("GTiff")
	output_file = "output/process_image_NDVI/B"+str(band_n)+"/"+image.split('/')[-1]
	dst_ds = driver.Create(output_file,
	                       band.XSize,
	                       band.YSize,
	                       band_n,
	                       gdal.GDT_Int16)

	new_array = np.array(new_array)
	#writting output raster
	dst_ds.GetRasterBand(band_n).WriteArray( new_array )
	#setting nodata value
	dst_ds.GetRasterBand(band_n).SetNoDataValue(-999)
	#setting extension of output raster
	# top left x, w-e pixel resolution, rotation, top left y, rotation, n-s pixel resolution
	dst_ds.SetGeoTransform(geotransform)
	# setting spatial reference of output raster
	srs = osr.SpatialReference()
	srs.ImportFromWkt(wkt)
	dst_ds.SetProjection( srs.ExportToWkt() )
	#Close output raster dataset

	ds = None
	dst_ds = None

#files_tif=glob.glob("output/clip_NDVI/*.tif")
#for m in files_tif:
#	process_image(2, m)

#main()
clip_raster('../output/tif/NDRE.tif','../output/shp/T0 P1.kmz.shp')
#reproject_et('../output/tif/NDRE.tif','test.tif',new_crs = 'EPSG:4326')

