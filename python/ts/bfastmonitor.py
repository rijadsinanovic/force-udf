import numpy as np
import matplotlib.pyplot as plt
import copy
from bfast import BFASTMonitor
from bfast.monitor.utils import crop_data_dates
import pandas
import numpy

"""
Reference: Gieseke, F., Oancea, C., Serykh, D., Rosca, S., Henriksen, T. & Verbesselt, J.
(2021): Break Detection for Satellite Time Series Data. Break Detection
for Satellite Time Series Data — BFAST 0.7 documentation (Zugriff:
13.03.2023)
"""
def forcepy_init(dates, sensors, bandnames):

    bandnames = ['breakyear', 'magnitudes', 'means', 'breaks']
    return bandnames

def forcepy_pixel(inarray, outarray, dates, sensors, bandnames, nodata, nproc):
    
    #convert Epoch time in datetime.datetime object
    d = dates.tolist()

    from datetime import date, timedelta

    start = date(1970,1,1) #"Days Since" part
    from datetime import datetime
    dates3 = []
    for x in d:
        delta = timedelta(x)     # Creating a time delta object from the number of days
        offset = start + delta      # Addition of the specified number of days to 1990
        dates3.append(datetime(offset.year, offset.month, offset.day)) # date object -> datetime.datetime
    
    """
    # Save all recording data in a txt file, comment out lines only if a tile is being edited
    with open("dates_test_TSS_95.txt", "w") as file:
        for line in dates3:
            file.write(line.strftime("%d-%m-%Y") + "\n")
    """
    arr = np.reshape(inarray, (inarray.shape[0], inarray.shape[2], 1*inarray.shape[3])) #innarray to 3D
    
    start_hist = datetime(1980, 1, 1) # Start history period
    start_monitor = datetime(1990, 1, 1) # Start monitoring period
    end_monitor = datetime(2021, 1, 1) # Ende monitoring period
    data, dates3 = crop_data_dates(arr, dates3, start_hist, end_monitor) #Input arr and dates3 are tailored to each other
    
    # fit bfast monitor
    model = BFASTMonitor(
            start_monitor,
            freq=365,
            k=3, # Number of harmonic terms
            hfrac=0.25, 
            trend=False,
            level=0.05,
            backend='python',
            device_id=0,
    )
    
    model.fit(data, dates3, nan_value = -9999)
    
    # Prepare Break Array
    breaks = model.breaks
    breaks_plot = breaks.astype(np.float)
    breaks_plot[breaks == -2] = -1  #Set pixel with a value of -2 to -1 (-1 == NODATA)
    

    # Step 1: Collect data starting from the monitoring period
    dates_monitor = []

    for i in range(len(dates3)):
        if start_monitor <= dates3[i]:
            dates_monitor.append(dates3[i])
    dates_array = np.array(dates_monitor)
    
    # Step 2: Record the minimum and maximum year and create a list of the years in between.
    minyear = dates_monitor[0].year
    maxyear = dates_monitor[-1].year

    list_from_1_to_n = [] #List with the years
    for x in range(minyear,maxyear+1):
        list_from_1_to_n.append(x)
    dates_array2 = np.array(list_from_1_to_n)

    # Step 3: Counting how many times a year occurs in monitoring period
    dk = []
    for x in dates_array2:
        dk.append(np.argmax((dates_array >= datetime(x, 1, 1)) > False))
    
    # Step 4: break index is assigned to the year
    breaks_plot_years = copy.deepcopy(breaks_plot)
    i = 0
    while i < len(list_from_1_to_n):
        if i == 0:
            breaks_plot_years[np.where( (breaks_plot <= dk[1]) & (breaks_plot >= dk[0]) ) ] = 0
        elif i == len(list_from_1_to_n)-1:
            breaks_plot_years[np.where(dk[len(dk)-1] < breaks_plot)] = i
        else:
            breaks_plot_years[np.where(np.logical_and(dk[i] < breaks_plot, breaks_plot <= dk[i+1]))] = i
        i = i+1

    intk = breaks_plot_years.astype(np.int16)

    #Step 5: save results in Outarray
    outarray [:] = [intk, model.magnitudes, model.means, model.breaks]
    
    """
    # Collect data for the monitoring period for visualisation and save as a txt file.
    # Comment out lines only when a tile is edited, as the recording data is overwritten.
    with open("dates_test_monitor(95).txt", "w") as file:
        for line in dates_array:
            file.write(line.strftime("%d-%m-%Y") + "\n")
    """
