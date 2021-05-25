#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import gdal
import ogr, osr
import matplotlib.pyplot as plt
from rasterio import plot
import earthpy.plot as ep
import scipy as sp
import scipy.interpolate
import glob

def area(path,band,tip):
    files_tif=glob.glob(path)
    area_verde_porcentual=[]
    for m in files_tif:
        chm_dataset = gdal.Open(m)


        # In[3]:


        #Display the dataset dimensions, number of bands, driver, and geotransform 
        cols = chm_dataset.RasterXSize; print('# of columns:',cols)
        rows = chm_dataset.RasterYSize; print('# of rows:',rows)
        print('# of bands:',chm_dataset.RasterCount)
        print('driver:',chm_dataset.GetDriver().LongName)


        # In[4]:


        print('projection:',chm_dataset.GetProjection())


        # In[5]:


        print('geotransform:',chm_dataset.GetGeoTransform())


        # In[6]:


        chm_mapinfo = chm_dataset.GetGeoTransform()
        xMin = chm_mapinfo[0]
        yMax = chm_mapinfo[3]

        xMax = xMin + chm_dataset.RasterXSize/chm_mapinfo[1] #divide by pixel width 
        yMin = yMax + chm_dataset.RasterYSize/chm_mapinfo[5] #divide by pixel height (note sign +/-)
        chm_ext = (xMin,xMax,yMin,yMax)
        print('chm raster extent:',chm_ext)


        # In[7]:


        chm_raster = chm_dataset.GetRasterBand(1)
        import rasterio
        dataset = rasterio.open(m)
        #chm_array=dataset.read((1,2,3,4)) 
        noDataVal = chm_raster.GetNoDataValue(); 
        print('no data value:',noDataVal)
        scaleFactor = chm_raster.GetScale(); 
        print('scale factor:',scaleFactor)
        chm_stats = chm_raster.GetStatistics(True,True)
        print('SERC CHM Statistics: Minimum=%.2f, Maximum=%.2f, Mean=%.3f, StDev=%.3f' % 
              (chm_stats[0], chm_stats[1], chm_stats[2], chm_stats[3]))



        data = dataset.read([band])




        chm_array = chm_dataset.GetRasterBand(band).ReadAsArray(0,0,cols,rows).astype(np.float)
        chm_array[chm_array==noDataVal]=np.nan #Assign CHM No Data Values to NaN
        chm_array=chm_array/scaleFactor
        print('SERC CHM Array:\n',chm_array) #display array values


        # In[12]:


        # Calculate the % of pixels that are NaN and non-zero:
        pct_nan = np.count_nonzero(np.isnan(chm_array))/(rows*cols)
        print('% NaN:',round(pct_nan*100,2))
        print('% non-zero:',round(100*np.count_nonzero(chm_array)/(rows*cols),2))


        # In[13]:


        def plot_spatial_array(band_array,spatial_extent,colorlimit,ax=plt.gca(),title='',cmap_title='',colormap=''):
            plot = plt.imshow(band_array,extent=spatial_extent,clim=colorlimit);
            print(spatial_extent)
            cbar = plt.colorbar(plot,aspect=40); 
            plt.set_cmap(colormap); 
            cbar.set_label(cmap_title,rotation=90,labelpad=20);
            plt.title(title); 
            ax = plt.gca(); 
            ax.ticklabel_format(useOffset=False, style='plain'); #do not use scientific notation #
            rotatexlabels = plt.setp(ax.get_xticklabels(),rotation=90); #rotate x tick labels 90 degrees


        # In[14]:


        
        # In[15]:
        if tip=='NDVI':
            #https://www.neonscience.org/classify-raster-thresholds-2018-py
            plt.hist(chm_array[~np.isnan(chm_array)],100);
            ax = plt.gca()
            ax.set_ylim([0,1e4]) #adjust the y limit to zoom in on area of interest
            plt.title('Distribution')
            plt.xlabel('pixel'); plt.ylabel('Relative Frequency')
            plt.savefig('output/image_processed/hist_'+str(m.split('/')[-1].split('.')[0])+'_'+str(band)+'.png', dpi=500)
            plt.close()




            plot_spatial_array(chm_array,
                            chm_ext,
                            (1,250),
                            title='Region de estudio',
                            cmap_title='NDVI BANDA 2',
                            colormap='BuGn')


            # In[44]:


            import copy
            chm_reclass = copy.copy(chm_array)
            #chm_reclass[np.where(chm_array==0)] = 1 # CHM = 0 : Class 1
            #chm_reclass[np.where((chm_array>0) & (chm_array<=20))] = 2 # 0m < CHM <= 10m - Class 2
            #chm_reclass[np.where((chm_array>20) & (chm_array<=60))] = 3 # 10m < CHM <= 20m - Class 3
            chm_reclass[np.where((chm_array>=0) & (chm_array<=230))] = 4 # 20m < CHM <= 30m - Class 4
            chm_reclass[np.where(chm_array>230)] = 5 # CHM > 30m - Class 5


            # In[46]:


            import matplotlib.colors as colors
            print(type(chm_reclass))
            plt.figure(); 
            cmapCHM = colors.ListedColormap(['green','red'])
            plt.imshow(chm_reclass,extent=chm_ext,cmap=cmapCHM)
            plt.title('Classification')
            ax=plt.gca(); ax.ticklabel_format(useOffset=False, style='plain') #do not use scientific notation 
            rotatexlabels = plt.setp(ax.get_xticklabels(),rotation=90) #rotate x tick labels 90 degrees

            # Create custom legend to label the four canopy height classes:
            import matplotlib.patches as mpatches
            
            class4_box = mpatches.Patch(color='green', label='100m < CHM <= 230m')
            class5_box = mpatches.Patch(color='red', label='CHM > 230m')

            ax.legend(handles=[class4_box,class5_box],
                      handlelength=0.7,bbox_to_anchor=(1.05, 0.4),loc='lower left',borderaxespad=0.)

            fig = ax.get_figure()
            fig.savefig('output/image_processed/'+str(m.split('/')[-1].split('.')[0])+'_'+str(band)+'.png', dpi=500)
            plt.close(fig)
            plt.close()
            print(chm_reclass[0][1])


            # In[32]:


            img = plt.imread(m) 
              
            # fetch the height and width 
            height, width, _ = img.shape 
              
            # area is calculated as “height x width” 
            area = height * width
            total = height + width
              
            # display the area 
            print("Area of the image is : ", area)
            print("total pixel: ", total)


            # In[34]:


            total=[]
            for n in range(chm_reclass.shape[0]):
                for m in range(chm_reclass.shape[1]):
                    if chm_reclass[n][m]== 4:
                        total.append(1)
                    else:
                        total.append(0)
            area_verde=sum(total)


            # In[35]:


            print("porcentaje color verde :", area_verde*100/area)
            total_area=area_verde*100/area
            area_verde_porcentual.append(total_area)
        elif tip == 'NDRE':
            plt.hist(chm_array[~np.isnan(chm_array)],100);
            ax = plt.gca()
            ax.set_ylim([0,1e4]) #adjust the y limit to zoom in on area of interest
            plt.title('Distribution')
            plt.xlabel('pixel'); plt.ylabel('Relative Frequency')
            plt.savefig('output/image_processed/hist_'+str(m.split('/')[-1].split('.')[0])+'_'+str(band)+'.png', dpi=500)
            plt.close()

            plot_spatial_array(chm_array,
                            chm_ext,
                            (0,1),
                            title='Region de estudio',
                            cmap_title='NDVI BANDA 2',
                            colormap='BuGn')


            # In[44]:


            import copy
            chm_reclass = copy.copy(chm_array)
            #chm_reclass[np.where(chm_array==0)] = 1 # CHM = 0 : Class 1
            #chm_reclass[np.where((chm_array>0) & (chm_array<=20))] = 2 # 0m < CHM <= 10m - Class 2
            #chm_reclass[np.where((chm_array>20) & (chm_array<=60))] = 3 # 10m < CHM <= 20m - Class 3
            chm_reclass[np.where((chm_array>=0) & (chm_array<=0.26))] = 4 # 20m < CHM <= 30m - Class 4
            chm_reclass[np.where(chm_array>0.26)] = 5 # CHM > 30m - Class 5


            # In[46]:


            import matplotlib.colors as colors
            print(type(chm_reclass))
            plt.figure(); 
            cmapCHM = colors.ListedColormap(['red','green'])
            plt.imshow(chm_reclass,extent=chm_ext,cmap=cmapCHM)
            plt.title('Classification')
            ax=plt.gca(); ax.ticklabel_format(useOffset=False, style='plain') #do not use scientific notation 
            rotatexlabels = plt.setp(ax.get_xticklabels(),rotation=90) #rotate x tick labels 90 degrees

            # Create custom legend to label the four canopy height classes:
            import matplotlib.patches as mpatches
            
            class4_box = mpatches.Patch(color='green', label='100m < CHM <= 230m')
            class5_box = mpatches.Patch(color='red', label='CHM > 230m')

            ax.legend(handles=[class4_box,class5_box],
                      handlelength=0.7,bbox_to_anchor=(1.05, 0.4),loc='lower left',borderaxespad=0.)

            fig = ax.get_figure()
            fig.savefig('output/image_processed/'+str(m.split('/')[-1].split('.')[0])+'_'+str(band)+'.png', dpi=500)
            plt.close(fig)
            plt.close()

            print(chm_reclass[0][1])


            # In[32]:


            img = plt.imread(m) 
              
            # fetch the height and width 
            height, width, _ = img.shape 
              
            # area is calculated as “height x width” 
            area = height * width
            total = height + width
              
            # display the area 
            print("Area of the image is : ", area)
            print("total pixel: ", total)


            # In[34]:


            total=[]
            for n in range(chm_reclass.shape[0]):
                for m in range(chm_reclass.shape[1]):
                    if chm_reclass[n][m]== 5:
                        total.append(1)
                    else:
                        total.append(0)
            area_verde=sum(total)


            # In[35]:


            print("porcentaje color verde :", area_verde*100/area)
            total_area=area_verde*100/area
            area_verde_porcentual.append(total_area)
    return area_verde_porcentual, files_tif
a1=area('output/clip_NDVI/*.tif',1,'NDVI')[0]
a2=area('output/clip_NDVI/*.tif',2,'NDVI')[0]
#a3=area('output/clip_NDVI/*.tif',3,'NDVI')[0]
a4=area('output/clip_NDRE/*.tif',1,'NDRE')[0]

files_tif1=glob.glob('output/clip_NDVI/*.tif')
files_tif4=glob.glob('output/clip_NDRE/*.tif')
files1=[  k.split('/')[-1].split('.')[0].split('_')[1] for k in files_tif1]
files4=[  k.split('/')[-1].split('.')[0].split('_')[1] for k in files_tif4]
#a5=area('output/clip_NDRE/*.tif',2)
#a6=area('output/clip_NDRE/*.tif',3)
import pandas as pd
areatotal=[]
df=pd.DataFrame()

df['files']=files1*2+files4

for n in a1:
    areatotal.append(n)
for n in a2:
    areatotal.append(n)
#for n in a3:
#    areatotal.append(n)
for n in a4:
    areatotal.append(n)
df['area']=areatotal
df['tipo']=['NDVI_B1']*24+['NDVI_B2']*24+['NDRE_B1']*24

df.to_csv('output/area.csv')