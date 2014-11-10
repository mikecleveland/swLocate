swLocate
========

##Earthquake relocation

by Mike Cleveland (07 Nov 2014)



##I. Introduction
The script is broken down into different parts, so you can choose exactly what is run
in the "workFlow" array. The basic steps are:

1. QC, Prepare, and Save *ALL* Data to HDF5 file 
2. Find *All* Viable Links, Compute CC, Write *ALL* to Text, Create Digital
3. Save (digital) log to Pickle format
4. Read in dataLog, ddArray, and myEventArray
5. Plot Correlation Values
6. Perform the iteration

There is also a step 7, but this was just used in the testing phase.

Covered in this overview are:

II. Modules 
	- required modules for the code
	
III. Parameters 
	- used defined parameters
	
IV. Workflow
	- The workflow of the code.


##II. Modules
II.a. pySACio - This is a waveform handling module that I wrote. It is included in the 
					gitHub set. This needs to be improved, but does the basics. I basically
					re-packages an Obspy trace into a format I like, then has some basic
					processing function, that are largely taken from Obspy. In addition
					to needing Obspy, this also needs h5py for storing data files.
					
II.b. h5py - Used to store the prepared waveforms in a compressed format that also reads
				quickly.

II.c. pickle - It is kind of silly that I store things as both h5py and pickle, but pickle
					is an easy way to dump dictionaries and arrays after data has been
					read in and processed.

##III. Parameters
The user defined parameters are all help within the dataStruct.settings dictionary. This
dictionary is made global so that all the functions can see it.

###Set Path 
1. dataStruct.settings['path'] = '/Users/mcleveland/Documents/Projects/Menard/EventSearch/Events/Graded/test2'
	* Define where the data are located
2. dataStruct.settings['pathPrefix'] = '/E`*'
	* The leading characters to the folders containing the data. I personally use "/E`*" for
		folders suchs as "E1995-09-15-04-50-20"
3. dataStruct.settings['dataSubDir'] = 'Dsp'
	* Inside my event folders, all the SAC files are in a subdirectory. Here, you just say the 
		name of the subdirectory folder. For example, 'Dsp' would relate to 
		E1995-09-15-04-50-20/Dsp
4. dataStruct.settings['fileSuffix'] = '.sac'
	* Describe what your SAC files end with. All of my end with '.sac'. If your files
		end with something that changes, such as 'LHZ' or 'LHT', you should be able to
		use a wildcard, like 'LH*'. I just use this so that it only tries to read in SAC
		files, instead of some other type of file held within the folder.

###HDF5 file save location 
1. directory = 'Waveforms/'
	* I think this is deprecated. It should save the HDF5 files to the same location
		as all of your event folders, but with '.h5' appended

###Define Period Band 
1. dataStruct.settings['shortPeriod'] = 30
2. dataStruct.settings['longPeriod']  = 80
	* Self explanatory

###Define Group Velocity Range (km/s) 
#####Rayleigh 
1. dataStruct.settings['rGvLow'] = 3
2. dataStruct.settings['rGvHi']  = 5
	* Self explanatory. This is used to window the waveforms.

#####Love 
1. dataStruct.settings['gGvLow'] = 3
2. dataStruct.settings['gGvHi']  = 5
	* This has not been implemented yet, however, it looks like the Love wave group
		velocity is close enough to the Rayleigh.
	
###Define Slowness 
1. dataStruct.settings['slowness'] = 0.24
	* Self explanatory

###Define Quality (False if not defined) 
1. dataStruct.settings['quality'] = 2
	* Minimum acceptable waveform quality. You can also use False if you don't want to use
		this.

###Define Channel 
1. dataStruct.settings['channels'] = ['lhz','lht']
	* Channels that are read in. I don't think this is case-sensitive.
	* Below are all the linking parameters.

###Define linking distance (km) 
1. dataStruct.settings['linkDist'] = 120

###Define minimum acceptable CC coefficient 
1. dataStruct.settings['minCC'] = 0.90

###Define minimum number of links 
1. dataStruct.settings['minLinks'] = 12

###Define minimum azimuthal coverage of links (degrees) 
1. dataStruct.settings['minAZ'] = 50

###*Below are weighting values for the inversion*
###Weight by distance (True/False) 
1. dataStruct.settings['weightByDistance'] = False

###Define zero centroid weight 
1. dataStruct.settings['zeroCentroidWt'] = 0.000

###Define minimum length weight 
1. dataStruct.settings['minLengthWt'] = 0.000

###Define GCarc to km conversion 
1. dataStruct.settings['gc2km'] = 111.19
	* Self explanatory



##IV. Workflow
1. QC, Prepare, and Save *ALL* Data to HDF5 file 
	This function reads in all of the waveforms, checks to see if they meet basic quality
	standards (e.g. desired channel(s), quality), processes the waveforms (e.g. filter, 
	taper, cut), then saves all of the processed data to HDF5 files and appends all of the
	the data to dataStruct. dataStruct will later be saved as a Pickle file. This includes
	all of the data and settings.
	
2. Find *All* Viable Links, Compute CC, Write *ALL* to Text, Create Digital
	This step finds all viable links (based only on matching stations) and computes the
	cross-correlation. This produces three structures:
	
	a. dataLog (dictionary): All of the observations are then stored to the dataLog dictionary. 
		The	observations are divided between 'accepted' and 'unused' based linking distance
		and minimum correlation coefficient (function: shouldLink). But, all observations 
		are stored, so at a later time the settings can be changed and the use 'accepted' 
		data updated without needing to re-process the data.
	
	b. ddArray (array): Array of DDObservation objects. Only the accepted observations
		are included in this array. However, the user can change linking parameters and
		rebuild this array with:
			ddArray = reBuildDifferenceArray(dataLog,myEventArray)
	
	c. myEventArray (object): EventArray object that is basically and array of Event objects,
		storing information about all of the events being investigated. Locations of these
		events are updated in the inversion processes.
	
3. Save (digital) log to Pickle format
	Dumps dataLog, ddArray, and myEventArray to a Pickle files. This way, unless you are 
	wanting to re-process your data, you just run steps 1-3 once, then you can simply read 
	in the Pickle files to perform the inversion. If additional data is added, this could
	also be a way of only processing the new data, then simply appending it on the existing
	measurements you have made. I have not worked out the exact procedure for this situation.
	
4. Read in dataLog, ddArray, and myEventArray
	Simply reads in the dataLog, ddArray, and myEventArray Pickle files.
	
5. Plot Correlation Values
	Plots all correlation coefficient vs azimuth plots.
	
6. Perform the iteration
	Performs the inversion.


