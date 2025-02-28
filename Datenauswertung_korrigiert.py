# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 16:24:40 2025

@author: danie
"""

import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.linear_model import LinearRegression

# basic directory
base_dir = r"D:\Hochschule\Auslandssemester Italien\Videoaufnahmen\11-02-25_neu"

#Function to convert the RGB pictures into grayscale pictures
def convert_rgb_to_grayscale():
    print("Konvertiere RGB-Bilder zu Graustufen und erstelle Mittelwertbilder...")

    # Algae types
    algae_types = ['Chlorella_vulgaris', 'Scenedesmus_sp']

    for algae in algae_types:
        algae_dir = os.path.join(base_dir, algae)

        #find all image_sequence_folders in algae_dir
        sequence_folders = [f for f in os.listdir(algae_dir) if f.startswith("image_sequence")]
        
        #find every folder in this iteration
        for seq in sequence_folders:
            rgb_dir = os.path.join(algae_dir, seq, 'RGB_picture') #RGB_pictures-folder
            gray_dir = os.path.join(algae_dir, seq, 'Grayscale_picture') #Grayscale_picture-folder
            mean_dir = os.path.join(algae_dir, seq, 'Mean_picture') #Mean_picture-folder
            subtracted_dir = os.path.join(algae_dir, seq, 'Subtracted_picture')# Subtracted_picture-folder

            # create folder, if they do not exist
            os.makedirs(gray_dir, exist_ok=True)
            os.makedirs(mean_dir, exist_ok=True)
            os.makedirs(subtracted_dir, exist_ok=True)

            # load all RGB-pictures
            rgb_image_files = glob.glob(os.path.join(rgb_dir, '*.tiff'))
            if not rgb_image_files:
                print(f"Keine RGB-Bilder in {rgb_dir} gefunden. Überspringe...") #check if it founds something
                continue

            # open first pictures to get the dimensions
            first_image = Image.open(rgb_image_files[0]).convert("L")
            width, height = first_image.size
            sum_array = np.zeros((height, width), dtype=np.float32)

            # convert RGB-pictures to Grayscale pictures and sum them up into the sum_array
            for file in rgb_image_files:
                img = Image.open(file).convert("L")
                img.save(os.path.join(gray_dir, os.path.basename(file)))  # save Grayscale-picture
                img_array = np.array(img, dtype=np.float32)
                sum_array += img_array
                
            print(f"Graustufenbilder gespeichert in {gray_dir}")
            
            
            # calculate a mean-picture and save it
            mean_image_array = (sum_array / len(rgb_image_files)).astype(np.uint8)
            mean_image = Image.fromarray(mean_image_array)
            mean_image_path = os.path.join(mean_dir, "Mittelwertbild.tiff")
            mean_image.save(mean_image_path)
            print(f"Mittelwertbild gespeichert: {mean_image_path}")

            # calculate the subtracted_pictures and save them
            for file in rgb_image_files:
                img = Image.open(file).convert("L")
                img_array = np.array(img, dtype=np.float32) #convert image into an array
                diff_array = img_array - mean_image_array #subtract the mean-image from the grayscale-pictures
                diff_array_normalized = np.clip(diff_array + 128, 0, 255).astype(np.uint8) #set boundaries, that no negative numbers can be used
                diff_image = Image.fromarray(diff_array_normalized)
                diff_image.save(os.path.join(subtracted_dir, f"sub_{os.path.basename(file)}")) #save the images

    print("Konvertierung, Mittelwertbildung und Differenzbildung abgeschlossen.")
    
    
#create the so-called "FinalMap_picture"
def create_visualization():
    print("Erstelle Visualisierung basierend auf minimalen Pixelwerten...")
    image_dict = {}

    for root, _, files in os.walk(base_dir):
        if "Subtracted_picture" in root:
            image_files = glob.glob(os.path.join(root, "*.tiff"))
            if not image_files:
                continue
        
            # Determine the parent directory (one level above ‘Subtracted_picture’)
            parent_dir = os.path.dirname(root)
            visualization_subfolder = os.path.join(parent_dir, "FinalMap_picture")
            os.makedirs(visualization_subfolder, exist_ok=True)
          
            img_slope = []
            for i, file in enumerate(image_files):
                img_array = np.array(Image.open(file).convert("L"))
                min_pixel_values = np.min(img_array, axis=1)
                img_slope.append(min_pixel_values)

                # save after each  500 pictures
                if (i + 1) % 500 == 0:
                    final_map = np.flip(np.array(img_slope), (0, 1))
                    vis_image = Image.fromarray(final_map.astype(np.uint8))
                    
                    # extract the relevant information from the directory
                    algentyp = root.split(os.sep)[-3]  # Algae type
                    sequence_number = root.split(os.sep)[-2]  # Image Sequence Number
                
                    # create new data-name with all relevant information
                    vis_filename = f"Visualization_{algentyp}_{sequence_number}.png"
                
                    # define the storage path and save image
                    vis_save_path = os.path.join(visualization_subfolder, vis_filename)
                    vis_image.save(vis_save_path)
                    
                    img_slope = []  # reset for the next sequence

            # determine the key for the mean-image
            algentyp = root.split(os.sep)[-3]
            key = (algentyp)
            image_dict.setdefault(key, []).append(final_map)

    # create the mean picture over the six sequences
    #visualization_folder = os.path.join(base_dir, "FinalMap_picture")
    visualization_folder = r"D:\Hochschule\Auslandssemester Italien\Videoaufnahmen\11-02-25_neu\visualization_map\gemittelte_Bilder"
    os.makedirs(visualization_folder, exist_ok=True)

    for (algentyp), images in image_dict.items():
        if len(images) == 6:
            mean_image = Image.fromarray(np.mean(images, axis=0).astype(np.uint8))
            save_path = os.path.join(visualization_folder, f"Gemitteltes_Bild_{algentyp}.png")
            mean_image.save(save_path)

    print("Visualisierung abgeschlossen.")
    
# colors for the different algae types
color_map = {
    "Chlorella_vulgaris": "orange",
    "Scenedesmus_sp": "blue"
}

#Function to plot one regression curve
def plot_regression_curve(image_path, time_x_axis, pixel_size, color = "gray"):
    """Hilfsfunktion, um eine einzelne Regressionsgerade zu berechnen und zu plotten."""
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Fehler beim Laden von {image_path}")
        return

    # extract algae type from the file path
    # Gehe davon aus, dass der Name der Algenart im Ordnernamen vor der "FinalMap_picture" erscheint
    parent_dir = os.path.dirname(image_path)
    algentyp = parent_dir.split(os.sep)[-2]  # algae type is second entry before "FinalMap_picture"
    
    # chose color based on the algae type
    color = color_map.get(algentyp,color)

    # Canny Edge detection method
    edges = cv2.Canny(image, 50, 150)
    y_indices, x_indices = np.where(edges > 0)

    if len(x_indices) == 0 or len(y_indices) == 0:
        print(f"Keine Kanten gefunden für {image_path}")
        return

    # Linear regression
    model = LinearRegression().fit(x_indices.reshape(-1, 1), y_indices.reshape(-1, 1))
    x_fit = np.linspace(min(x_indices), max(x_indices), 500).reshape(-1, 1)
    y_fit = model.predict(x_fit)
 
    # transformation of the coordinate system to set the origin to zero
    x_reference, y_reference = x_fit[0, 0], y_fit[0, 0]
    x_transformed = x_fit - x_reference
    y_transformed = y_fit - y_reference
    x_transformed_positive = x_transformed - x_transformed[0]
    y_transformed_positive = y_transformed - y_transformed[0]
    y_um = y_transformed_positive * pixel_size #change the pixel values to the micrometer size
    
    #print(len(y_um))
    
    #plot für regressionskurve im graphen
    # plt.figure(figsize=(10, 6))
    # plt.imshow(image, cmap='gray')
    # plt.scatter(x_indices, y_indices, s=2, color='red', alpha=0.1)  # Kantenpunkte
    # plt.plot(x_fit, y_fit, color = "yellow", linewidth = 2)
    # plt.show()
    
    plt.plot(time_x_axis, y_um, linewidth=2, label=image_path, color=color)
    plt.xlim(0, max(time_x_axis) + 50)  # set x-axis range
    plt.ylim(0, max(y_um) + 50) # set y-axis range
    
    # # Steigung berechnen und ausgeben
    # slope = model.coef_[0][0] * pixel_size
    # print(f"Steigung der Regressionslinie für {image_path}: {slope:.4f} µm/s")
    
    return max(y_um)

# def plot_regression_on_visualization(image_sequence_folder, pixel_size, time_x_axis):
#     final_map_folder = os.path.join(image_sequence_folder, "FinalMap_picture") #FinalMap_picture-Ordner finden
#     if not os.path.exists(final_map_folder):
#         print("2")
#     visualization_images = glob.glob(os.path.join(final_map_folder, "*.png")) #visualization_map-bild finden
    
#     visualization_image_path = visualization_images[0]
#     vis_image = cv2.imread(visualization_image_path, cv2.IMREAD_GRAYSCALE)
    
    
#Function for plotting each regression curve for both algae type into one plot
def plot_all_curves():
    """Plottet die Regressionsgeraden für jede Image-Sequence."""
    print("Erstelle Vergleichsdiagramm für einzelne Image-Sequenzen...")

    plt.figure(figsize=(10, 6))

    pixel_size = 3.45  # pixel size in µm
    time_x_axis = np.arange(0, 500)  # Time axis for x-values

    # collects all image paths from the Visualization-folder ( no mean-pictures)
    image_paths = []
    for root, _, files in os.walk(base_dir):
        if "FinalMap_picture" in root:
            image_paths.extend(glob.glob(os.path.join(root, "*.png")))

    # to determine the maximum y-value
    max_y_value = 0

    # counter for the number of lines for each algae type
    chlorella_count = 0
    scenedesmus_count = 0
    
    wert_500_frame_Chlorella_vulgaris = []
    wert_500_frame_Scenedesmus_sp = []
    for image_path in image_paths:
        # set color based on algae type
        if "Chlorella_vulgaris" in image_path:
            color = "orange"
            chlorella_count += 1
            save_list = wert_500_frame_Chlorella_vulgaris
        elif "Scenedesmus_sp" in image_path:
            color = "blue"
            scenedesmus_count += 1
            save_list = wert_500_frame_Scenedesmus_sp
        else:
            color = "gray"  # if there is a unknown file
            save_list = None

        # plot regression curve
        max_y = plot_regression_curve(image_path, time_x_axis,pixel_size, color)
        if max_y is not None and save_list is not None:
            max_y_value = max(max_y_value, max_y)
            save_list.append(max_y)
            
    #print(wert_500_frame_Scenedesmus_sp)
    
    # Überprüfen, ob die richtige Anzahl an Linien existiert
    #print(f"Chlorella vulgaris Linien: {chlorella_count} (Erwartet: 6)")
    #print(f"Scenedesmus sp. Linien: {scenedesmus_count} (Erwartet: 6)")
    
    # Caption
    legend_handler = [
        plt.Line2D([0], [0], linewidth=2, color="orange", label="Chlorella vulgaris"),
        plt.Line2D([0], [0], linewidth=2, color="blue", label="Scenedesmus sp.")
    ]
    # finalise the plot with adjusting fontsize, etc.
    plt.xlabel("time (in s)", fontsize=14)
    plt.ylabel("y-coordinates (in µm)", fontsize=14)
    plt.yticks(np.arange(0, float(max_y_value[0]) + 100, 100))#float(max_y_value[0]) gets the single value from the max_y_value-Array
    plt.legend(handles=legend_handler, fontsize=13)
    plt.grid(True)
    plt.show()
    #print(f"max_y_value Typ: {type(max_y_value)}, Wert: {max_y_value}")
    print("Diagramm für einzelne Image-Sequenzen abgeschlossen.")  
    
    #print(wert_500_frame_Chlorella_vulgaris)
    return wert_500_frame_Chlorella_vulgaris, wert_500_frame_Scenedesmus_sp
    
#function for plotting the mean curve for each algae type
def plot_mean_curve():
    """Plottet die gemittelten Regressionsgeraden für Chlorella vulgaris und Scenedesmus sp."""

    # recall the values from the function plot_all_curves()
    wert_500_frame_Chlorella_vulgaris, wert_500_frame_Scenedesmus_sp = plot_all_curves()
    
    #calculate the standard deviation
    std_Chlorella_vulgaris = np.std(wert_500_frame_Chlorella_vulgaris)
    std_Scenedesmus_sp = np.std(wert_500_frame_Scenedesmus_sp)
    
    #print(std_Chlorella_vulgaris)
    #print(std_Scenedesmus_sp)
    
    # time-axis
    #time_x_axis = np.arange(0, 500)  # Zeitachse für x-Werte
    time_x_axis = np.array([0, 500]) #we have to points for the line, since mean_chlorella and mean_scenedesmus are singel values
    pixel_size = 3.45  # pixel size in µm

    plt.figure(figsize=(10, 6))
    
    # calculate mean value for each algae type
    if wert_500_frame_Chlorella_vulgaris:
        mean_chlorella = np.mean(wert_500_frame_Chlorella_vulgaris)  #calculate mean value
        slope_chlorella = mean_chlorella / 500 # Steigung
        print(f"Steigung Chlorella vulgaris: {slope_chlorella:.4f} µm/s") #give me the velocity of each line
        plt.plot(time_x_axis, [0, mean_chlorella], color="orange", linewidth=2, label="mean Chlorella vulgaris") #y-values are 0 and the mean value at frame 500
        
        
        #plot the standard deviation into the plot (mean value +- std)
        plt.fill_between(time_x_axis, 
                         [0, mean_chlorella - std_Chlorella_vulgaris],
                         [0, mean_chlorella + std_Chlorella_vulgaris],
                         color = "orange", alpha = 0.2)
    
    #same thing for scenedesmus_sp
    if wert_500_frame_Scenedesmus_sp:
        mean_scenedesmus = np.mean(wert_500_frame_Scenedesmus_sp)  # Mittelwert berechnen
        slope_scenedesmus = mean_scenedesmus / 500 # Steigung
        print(f"Steigung Scenedesmus sp.:{slope_scenedesmus:.4f} µm/s")
        
        plt.plot(time_x_axis, [0, mean_scenedesmus], color="blue", linewidth=2, label="mean Scenedesmus sp.") 
        
        #Standardabweichungen reinzeichnen als der mittelwert +- die Standardabweichung
        plt.fill_between(time_x_axis,
                         [0, mean_scenedesmus - std_Scenedesmus_sp],
                         [0, mean_scenedesmus + std_Scenedesmus_sp],
                         color = "blue", alpha = 0.2)

    # format for x- and y-labels
    plt.xlabel("time (in s)", fontsize=14)
    plt.ylabel("y-coordinates (in µm)", fontsize=14)
    
    max_y_value = max(max(wert_500_frame_Chlorella_vulgaris), max(wert_500_frame_Scenedesmus_sp))
    
    plt.yticks(np.arange(0, float(max_y_value[0]) + 100, 100))  # y-axis in 100er steps
    plt.xlim(0, 500 + 50)  # x-axis from 0 to 550
    plt.ylim(0, float(max_y_value[0]) + 50)  # y-axis from 0 to the maximal value plus 50

    plt.legend(fontsize=13, loc ="upper left") #puts legend in top left place
    plt.grid(True)
    plt.title("Mean regression curves for Chlorella vulgaris and Scenedesmus sp.")
    
    
    plt.show()
    
    
#--------------------Code-Ausführung----------------------------  
    

#plot_regression_curve(r"D:\Hochschule\Auslandssemester Italien\Videoaufnahmen\11-02-25_neu\Chlorella_vulgaris\image_sequence_2\FinalMap_picture\Visualization_Chlorella_vulgaris_image_sequence_2.png", np.arange(0, 500), 3.45)
#plot_all_curves()
plot_mean_curve()


#-------------Calculation of the density of the algae
algae_diameter_chlorella = [24.395, 27.815, 24.395, 31.807, 26.274, 22.091, 17.592, 30.858, 13.800, 34.672]
algae_diameter_chlorella_in_meter = [x / 1000000 for x in algae_diameter_chlorella]
algae_diameter_without_magnification_chlorella = [x / 4 for x in algae_diameter_chlorella_in_meter]
R_chlorella = [x / 2 for x in algae_diameter_without_magnification_chlorella]
R_mean_chlorella = np.mean(R_chlorella)
print(R_mean_chlorella)

algae_diameter_scenedesmus =[60.252, 70.788, 66.987, 39.336, 56.584, 64.820, 65.912, 80.467, 73.348, 60.252]
algae_diameter_scenedesmus_in_meter = [x / 1000000 for x in algae_diameter_scenedesmus]
algae_diameter_without_magnification_scenedesmus = [x / 4 for x in algae_diameter_scenedesmus_in_meter]
R_scenedesmus = [x / 2 for x in algae_diameter_without_magnification_scenedesmus]
R_mean_scenedesmus = np.mean(R_scenedesmus)
print(R_mean_scenedesmus)
g = 9.81 #m/s^2
rho_water = 997  #kg/m^3
μ =  0.001002 #kg/(m*s)


# get the velocity from plot_mean_curve()
slope_chlorella = 1.2016e-6 #m/s ; 1.2016 um/s
slope_scenedesmus = 3.941e-7 #m/s ; 0.3941 um/s

#print density and mass for Chlorella vulgaris
rho_chlorella = ((slope_chlorella * μ) / (g*(R_mean_chlorella)**2)) * (9/2) + rho_water
volume_chlorella = 4/3 * np.pi * R_mean_chlorella**3
mass_chlorella =  volume_chlorella * rho_chlorella
print(f"Dichte von Chlorella: {rho_chlorella}")
print(f"Masse von Chlorella: {mass_chlorella}")

#print density and mass for Scenedesmus sp.
rho_scenedesmus = ((slope_scenedesmus * μ) / (g*(R_mean_scenedesmus)**2)) * (9/2) + rho_water
volume_scenedesmus = 4/3 * np.pi * R_mean_scenedesmus**3
mass_scenedesmus =  volume_scenedesmus * rho_scenedesmus
print(f"Dichte von Scenedesmus: {rho_scenedesmus}")
print(f"Masse von Scenedesmus:{mass_scenedesmus}")



