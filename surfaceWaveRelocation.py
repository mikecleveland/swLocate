#!/bin/python
# -*- coding: utf-8 -*- 

'''
	surfaceWaveRelocation.py version 1.0.1 (original 07 Mar, 2014)
				
		by Mike Cleveland
	
	Last edit: 14 Nov 2014 (KMC)
	
	14 Nov 2014 - Plot of traveltime difference versus azimuth has been improved to show
					a plot of the correlation coefficients of each point and the initial
					and optimized cosine curve. I don't think this curves are right yet,
					they seem like they could fit the data better.
					
					A histogram of the results and map of the improved locations needs to
					be added.
					
					Calculation of relative magnitudes should also be included.

'''
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
import sys
sys.path.insert(0, '/Users/mcleveland/Documents/Projects/pySACio')

import pySACio as ps
# reload(ps)

import os
import glob
import h5py
import numpy as np
from numpy.fft import rfft, irfft

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42	# Set font so it can be edited in Illustrator
mpl.rcParams['ps.fonttype'] = 42

from copy import deepcopy
from scipy.linalg import lapack 

import datetime

from obspy.core.util import gps2DistAzimuth
## The user is urged to install: http://geographiclib.sourceforge.net/
#	Without this package, gps2DistAzimuth uses Vincenty’s Inverse formulae which can
#	produce errors 

import pickle

import timeit

import pdb	# Debug with pdb.set_trace()

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
class Waveforms():
	def __init__(self):
		"Initialize an object and read in all waveforms."

		self.settings = {}

		self.settings['path'] = './'
		self.settings['pathPrefix'] = '/E*'
		self.settings['dataSubDir'] = 'Dsp'
		self.settings['fileSuffix'] = '.sac'

		self.settings['wavesDir'] = './'
		
		self.settings['shortPeriod'] = 30
		self.settings['longPeriod'] = 80

		self.settings['rGvLow'] = 3.0
		self.settings['rGvHi'] = 5.0
		self.settings['gGvLow'] = 3.0
		self.settings['gGvHi'] = 5.0
		
		self.settings['slowness'] = 0.25

		self.settings['quality'] = False

		self.settings['channels'] = []

		self.settings['linkDist'] = 0
		self.settings['minCC'] = 0
		self.settings['minLinks'] = 0
		self.settings['minAZ'] = 0
		
		self.settings['weightByDistance'] = False
		self.settings['zeroCentroidWt'] = 0.0
		self.settings['minLengthWt'] = 0.0

		self.settings['gc2km'] = 111.19
		
		self.waveforms = {}

class Event(ps.Trace):
	global settings
	
	def __init__(self,trace):		
		self.originTimeShift = 0.0

		self.stationID   = trace.stationID
		self.network     = trace.network
		self.station     = trace.station
		self.staLocation = trace.staLocation
		self.channel     = trace.channel
		self.staLat      = trace.staLat
		self.staLon      = trace.staLon
		self.staElev     = trace.staElev
		self.staDepth    = trace.staDepth
		self.calibration = trace.calibration
		self.componentAzNorth = trace.componentAzNorth
		self.componentIncidentAngleVertical = trace.componentIncidentAngleVertical
		
		self.origin  = trace.origin
		self.originStr = trace.origin.strftime('%Y-%m-%d-%H-%M-%S')
		self.evLat   = trace.evLat  
		self.evLon   = trace.evLon  
		self.evLatInitial = trace.evLat  
		self.evLonInitial = trace.evLon  
		self.evDepth = trace.evDepth
		self.mag     = trace.mag    

		self.distance = trace.distance
		self.gcarc    = trace.gcarc
		self.az       = trace.az
		self.baz      = trace.baz
		
		self.quality    = trace.quality   
		self.idep       = trace.idep      
		self.npts       = trace.npts      
		self.sampleRate = trace.sampleRate
		self.nyquist    = trace.nyquist   
		self.delta      = trace.delta     
		self.startTime  = trace.startTime 
		self.endTime    = trace.endTime   
		self.refTime    = trace.refTime   
		self.b          = trace.b         
		self.e          = trace.e         
		self.minAmp     = trace.minAmp    
		self.maxAmp     = trace.maxAmp    
		self.meanAmp    = trace.meanAmp   
		
		if int(trace.fileType) == 9:
			self.fileType = 'Reference time beginning of trace'
		elif int(trace.fileType) == 11:
			self.fileType = 'Reference origin'		
		else:
			self.fileType = trace.fileType

		self.processing = trace.processing
	def description(self):	
		myDescription = ' %9.3f %9.3f' % (self.evLat, self.evLon)
		
		return myDescription
	def gcarcOrigToLoc(self,theLat,theLon):
		(dist,az,baz) = gps2DistAzimuth(self.evLat,self.evLon,theLat,theLon)
		dist /= 1000.0
		gcarc = dist / settings['gc2km']
		
		return gcarc
	def azOrigToLoc(self,theLat,theLon):
		(dist,az,baz) = gps2DistAzimuth(self.evLat,self.evLon,theLat,theLon)
		
		return az
	def updateOriginTimeShift(value):
		self.originTimeShift += value
	def updateOrigin(self,newLat,newLon):
		(dist,az,baz) = gps2DistAzimuth(newLat,newLon,self.staLat,self.staLon)
		dist /= 1000.0
		gcarc = dist / settings['gc2km']
		
		self.evLat   = newLat  
		self.evLon   = newLon  

		self.distance = dist
		self.gcarc    = gcarc
		self.az       = az
		self.baz      = baz
		
class EventArray(object):
	def __init__(self):
		self.events = []
		self.nameArray = []
	def addEvent(self,eventsObj):
		self.events.append(eventsObj)
		self.nameArray.append(eventsObj.originStr)
		
		self.events = [x for [y,x] in sorted(zip(self.nameArray,self.events))]
		self.nameArray = sorted(self.nameArray)
	def setEventArray(self,valueArray):
		self.events = valueArray
		for a in self.events:
			self.nameArray = a.originStr

		self.events = [x for [y,x] in sorted(zip(self.nameArray,self.events))]
		self.nameArray = sorted(self.nameArray)
	def eventIndex(self,value):
		index = np.where(np.array(self.nameArray)==value.originStr)[0]
		
		return index
	def eventIndexStr(self,value):
		index = np.where(np.array(self.nameArray)==value)[0]
		
		return index

class DDObservation(object): 
	global settings
	  
	def __init__(self):
		self.aEvent  = None	# Event object
		self.bEvent  = None	# Event object
		self.phaseType = str()

		self.derivatives = np.zeros(8)
		self.weight = 1.0

		self.hSlowness = float()

		self.ccorAmplitude = float()
		self.ccorUnnormalizedAmplitude = float()
		self.powerSignalA = float()
		self.powerSignalB = float()
		
		self.dtObs = float()
		self.dtPredicted = float()

		self.qualityValue = float()

		self.powerSignalA = float()
		self.powerSignalB = float()
	def description(self):
		aE = self.aEvent
		bE = self.bEvent
		value = 'DD Obs: %5s %4s %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f | ' % \
				 (aE.station, aE.network, aE.staLat, trace.staLon,
				 aE.evLat, aE.evLon, bE.evLat, bE.evLon)

		value = value + '%8.3f %8.3f, %8.3f | %8.3f | %8.3f %8.3f %9.6e %9.6e' % \
				 (self.dtObs, self.dtPredicted, (self.dtObs-self.dtPredicted), 
				 self.weight, self.ccorAmplitude, self.ccorUnnormalizedAmplitude,
				 self.powerSignalA, self.powerSignalB)

		return value
	def updateDtObs(self, value):
		self.dtObs += value
	def setDerivatives(values):
		for i in np.arange(0,8):
			self.derivatives[i] = values[i]

	def computeDerivatives(self):
		'''
		Note that the derivative units are in the same units as hSlowness (s/km). Working 
		in these units keeps the colat and longitude derivatives ~1, which is comparable to 
		the origin time derivative, which is unity. That's better for the inversion stability.
		'''
		
		if self.aEvent and self.bEvent and self.hSlowness:
			aE = self.aEvent
			bE = self.bEvent
			hSlowness = self.hSlowness

			deg_to_rad = settings['deg_to_rad']
# 			deg_to_rad = np.arccos(-1.0) / 180.0
		
			## event 01 ##
			az          = aE.az
			colat       = deg_to_rad *(90.0 - aE.evLat)
			self.derivatives[0] =  hSlowness * np.cos(deg_to_rad * az) # dt/dcolat
			self.derivatives[1] = -hSlowness * np.sin(deg_to_rad * az) * np.sin(colat) # dt/dlon
			self.derivatives[2] = 0.0 # no depth partial
			self.derivatives[3] = 1.0 # dt/dT0
	
			## event 02 ##
			az          = bE.az
			colat       = deg_to_rad *(90.0 - bE.evLat)
			self.derivatives[4] =  hSlowness * np.cos(deg_to_rad * az); # dt/dcolat
			self.derivatives[5] = -hSlowness * np.sin(deg_to_rad * az) * np.sin(colat); # dt/dlon
			self.derivatives[6] = 0.0; # no depth partial, yet
			self.derivatives[7] = 1.0; # dt/dT0
					
		else:
			print 'DDObservation.computeDerivatives(): Variables not defined.'			
	def computePredictedDifference(self):
		'''
		Use the event and station locations and the hSlowness to compute the predicted 
		time shift (doesn't include depth right now). That would include the depth 
		difference times the vertical slowness.
		'''
		if self.aEvent and self.bEvent:
			aE = self.aEvent
			bE = self.bEvent
			
			dgcarc = aE.gcarc - bE.gcarc

	
			self.dtPredicted = dgcarc * settings['gc2km'] * self.hSlowness + \
								(aE.originTimeShift - bE.originTimeShift)

		else:
			print 'DDObservation.computePredictedDifference(): Variables not defined.'			
	def eventDistance(self):
		'''Returns the distance between the two events used in the difference.'''

		if self.aEvent and self.bEvent:
			aE = self.aEvent
			bE = self.bEvent

			gcarc = aE.gcarcOrigToLoc(bE.evLat,bE.evLon)
	
			return gcarc
		
		else:
			print 'DDObservation.eventDistance(): Variables not defined.'			

	def setWeight(self):
		self.weight = self.getDistanceDependentWeight()
	def getDistanceDependentWeight(self):
	
		distMax = settings['linkDist']
		dist = settings['gc2km'] * self.eventDistance()
			
		if (settings['weightByDistance'] == False):	
			return 1.0;
		
		wt = 0.0; # this will stay if distance > distMax
		
		if (dist <= distMax):
			wt = 0.125
	
		if (dist <= 0.5 * distMax):
			wt = 0.25
	
		if (dist <= 0.33 * distMax):
			wt = 0.50
	
		if (dist <= 0.25 * distMax):
			wt = 1.0
	
		return wt

class CCLog():
	def __init__(self,fileName='ccLog.txt'):
		"Initialize."

		self.ccLog = open(fileName,"w")
		header = 'Count Net  Sta Loc Chan:   staLat   staLon |   aEvLat   aEvLon,   bEvLat   bEvLon | '
		header = header + 'aGCarc   bGCarc,     aAz     bAz |dtObs dtPred (dtObs-dtPred)|'
		header = header + ' wt normCC unnormCC unnormCC_inverse | powerSignalA powerSignalB'
		print >>self.ccLog, header	
	def linking(self,aEventName,bEventName):
		linkingPhrase = 'Linking {} and {}'.format(aEventName,bEventName)
		print >>self.ccLog, linkingPhrase
		print linkingPhrase		
	def data(self,count,aTrace,bTrace,dtObs,dtPredicted,weight,nCC,unCC,unCC_i,powerA,powerB):
		 network = aTrace.network; station = aTrace.station
		 location= aTrace.staLocation; channel = aTrace.channel
		 
		 eventID = '   %03d %2s %5s %2s %3s : ' \
				% (count,network,station,location,channel)
		 staLoc = '%8.3f %8.3f | ' % (aTrace.staLat,aTrace.staLon)
		 eventLocs = '%8.3f %8.3f, %8.3f %8.3f | ' \
				% (aTrace.evLat,aTrace.evLon,bTrace.evLat,bTrace.evLon)
		 distAz = '%7.3f %7.3f, %7.3f %7.3f | ' \
				% (aTrace.gcarc,bTrace.gcarc,aTrace.az,bTrace.az)
		 dtValue = '%6.2f %6.2f %6.2f | ' % (dtObs,dtPredicted,(dtObs-dtPredicted))
		 results = '%0.3f %5.3f %5.3f %5.3f | ' % (weight,nCC,unCC,unCC_i)
		 results = results + '%9.6e %9.6e' % (powerA,powerB)

		 print >>self.ccLog, eventID + staLoc + eventLocs + distAz + dtValue+ results
	def close(self):
		self.ccLog.close()

class Linefit():
	'''
	 lineFit.c
 
	 Created by Charles Ammon on Fri Apr 16 2004.
	 Copyright (c) 2004 Charles Ammon. All rights reserved.
		(converted to Python by Mike Cleveland, 14 Nov 2014)
 
	 Simple line fit using perpendicular distances.
	'''
	def __init__(self,xDataArray=[],yDataArray=[],weightArray=[]):
		self.slope = None
		self.intercept = None
		self.covariance = np.zeros(shape=(2,2))
		self.misfit = None
		
		if (len(xDataArray) == len(yDataArray)) and (len(xDataArray) == len(weightArray)):
			self.xData = np.array(xDataArray,dtype=np.float64)
			self.yData = np.array(yDataArray,dtype=np.float64)
		
			self.weights = np.array(weightArray,dtype=np.float64)
					
		else:
			print 'xDataArray, yDataArray, and/or weightArray not equal lengths'

			self.xData = np.array([],dtype=np.float64)
			self.yData = np.array([],dtype=np.float64)
		
			self.weights = np.array([],dtype=np.float64)

		self.n = len(self.xData)

		self.misfit = None
	def computePerpLineFit(self):
		n = self.n
	
		sumx2= np.sum(self.xData * self.xData)
		sumy2= np.sum(self.yData * self.yData)
	
		sumxy= np.sum(self.xData * self.yData)

		xbar = np.mean(self.xData)
		ybar = np.mean(self.yData)
		
		B = 0.5 * (sumy2-n*ybar*ybar - sumx2-n*xbar*xbar) / ( n*xbar*ybar - sumxy )
	
		## Check both branches of sqrt ##
		b1 = -B + np.sqrt(B*B+1)
		a1 = ybar - b1 * xbar
		b2 = -B - np.sqrt(B*B+1)
		a2 = ybar - b2 * xbar
	
		mf1 = self.lfMisfit(b1,a1)
		mf2 = self.lfMisfit(b2,a2)
	
		if (mf1 < mf2):
			self.slope = b1
			self.intercept = a1
			self.misfit = mf1
		
		else:
			self.slope = b2
			self.intercept = a2
			self.misfit = mf2
	## this function uses the formulas for the vertical misfits ##
	def computeWtLineFit(self):
		sumx = np.sum(self.weights * self.xData)
		sumx2= np.sum(self.weights * self.xData * self.xData)

		sumy = np.sum(self.weights * self.yData)
		sumy2= np.sum(self.weights * self.yData * self.yData)

		sumxy= np.sum(self.weights * self.xData * self.yData)

		sumw = np.sum(self.weights)
	
		D = sumw * sumx2 - sumx*sumx
	
		A = (sumx2*sumy - sumx * sumxy) / D
		B = (sumw*sumxy - sumx * sumy ) / D
	
		self.slope     = B
		self.intercept = A
		self.misfit    = self.lfMisfit(B,A)
	def computeItWtLineFit(self, nIter=3):
	
		self.computeWtLineFit()
	
		for i in np.arange(0,nIter):
			self.adjustWeights()
			self.computeWtLineFit()
	def lfMisfit(self, slope, intercept):
		vmisfit = self.weights * (self.yData - (slope*self.xData + intercept))
		pmisfit = vmisfit*vmisfit
		sumwt = np.sum(self.weights)
	
		return np.sum(pmisfit) / sumwt
	def adjustWeights(self, cutoff = 1.0):
		cutoff = 1.0
	
		self.weights = np.ones(len(self.xData))
	
		vmisfit = np.abs(self.yData - (self.slope*self.xData + self.intercept))
	
		index = np.where(vmisfit > cutoff)[0]
	
		self.weights[index] = 1.0 / vmisfit[index]
		
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
## 1. Prep Data ##
# prepDataSets.py
def prepData(dataStruct):
	'''
	Performs quality control, filters, cuts, and saves waveforms to HDF5 files organized
		by event origin
	'''
	global settings
	
	##~~~ QC, Prepare, and Save Data to HDF5 file ~~~##
	events = glob.glob(settings['path']+settings['wavesDir']+settings['pathPrefix'])

	for aEvent in events:
		for aChan in settings['channels']:
			dataStruct.waveforms[aChan.upper()] = []

		if aEvent.split('.')[-1] != 'h5':
			eventID = aEvent.split('/')[-1]
			print 'Reading in ' + eventID
		
			aEvent = aEvent + '/' + settings['dataSubDir'] + '/*' + settings['fileSuffix']

			waves = ps.readDirectory(aEvent)
	
			for aTrace in waves:
				ok = True
			
				##~~~ Check if waveform is acceptable ~~~##
				ok,shortTime,longTime = checkWaveform(aTrace)
			
				##~~~ Prepare waveform ~~~##
				if ok:
					##~ Bandpass each waveform ~##
					shortP = settings['shortPeriod']
					longP = settings['longPeriod']
				
					aTrace.bandpass(1/float(longP),1/float(shortP))
		
					##~ Cut waveform around Rayleigh waves ~##
					aTrace.cut(starttime=shortTime,endtime=longTime,cutFrom='o')

					##~ Remove Mean and Taper ~##
					aTrace.detrend(type='constant')
					aTrace.cosTaper(p=0.15)
			
					##~ Add to list of waveforms ~##
					dataStruct.waveforms[aTrace.channel.upper()].append(aTrace)
			
			##~~~ Write waveforms to HDF5 file ~~~##			
			print 'Saving ' + eventID
			save2HDF5(dataStruct)		
def checkWaveform(trace):
	'''
	Checks to a specific trace for see if it qualifies based on:
		1. Quality rating
		2. Desired channel
		3. Length of time series
	'''
	
	global settings

	##~~~ Check if waveform is acceptable ~~~##
	ok = True
	
	#~ Quality ~#
	if settings['quality']:
		if trace.quality < settings['quality']:
			ok = False

	#~ Channel ~#
	if ok:
		if trace.channel.lower() in settings['channels']:
			if trace.channel.lower()[-1] == 'z':
				gvHi  = settings['rGvHi']
				gvLow = settings['rGvLow']
			elif trace.channel.lower()[-1] == 't':
				gvHi  = settings['gGvHi']
				gvLow = settings['gGvLow']
			else:
				print 'checkWaveform(): Unknown group velocity'
		else:
			ok = False

	#~ Length ~#
	if ok:
		shortTime = trace.gcarc * settings['gc2km'] / gvHi
		longTime  = trace.gcarc * settings['gc2km'] / gvLow
	
		if (trace.origin + shortTime) < trace.startTime:
			ok = False

		if (trace.origin + longTime) > trace.endTime:
			ok = False

	#~ Some method to remove "broken" waveforms ~#
	##
	
	if not ok:
		shortTime = None
		longTime = None
		
	return ok,shortTime,longTime

## 2. Calculate Cross-Correlation Values for All Pairs ##
def matchComputeCC(dataStruct):
	global settings

	## Initialize (digital) log file ##
	dataLog = {}
	ddArray = []
	myEventArray = EventArray()
		
	## Initialize (text) log file ##
	ccLog = CCLog(fileName=settings['path'] + '/ccLog-All.txt')

	## Read list of events ##
	events = glob.glob(settings['path']+settings['wavesDir']+settings['pathPrefix']+'.h5')

	##~~~ Calculate cross-correlations ~~~##
	## Loop over master events ## 
	index = 0
	for aEvent in events[:-1]:
		index += 1

		aEventName = aEvent.split('/')[-1].split('.')[0][1:]
		aEvent = h5py.File(aEvent, 'r')

		## Add master event to (digital) log file ##
		dataLog[aEventName] = {}

		## Loop over secondary events ##
		for bEvent in events[index:]:

			tempDDArray = []

			bEventName = bEvent.split('/')[-1].split('.')[0][1:]
			bEvent = h5py.File(bEvent, 'r')

			## Add secondary event to (digital) log file ##
			dataLog = updateDict(dataLog,aEventName,bEventName,step=1)

			## Update (text) log file ##
			count = 0
			ccLog.linking(aEventName,bEventName)

			## Look for viable links between master and secondary events ##
			for aChan in aEvent:
			 if aChan not in ['Settings']:
			  if aChan in bEvent:

			   ## Add all viable channels to (digital) log ##
			   if 'total' not in dataLog[aEventName][bEventName]['links']['totalLinks']:
			   		dataLog = updateDict(dataLog,aEventName,bEventName,channel='total',step=2)

			   if aChan not in dataLog[aEventName][bEventName]['links']['totalLinks']:
			   		dataLog = updateDict(dataLog,aEventName,bEventName,channel=aChan,step=2)

			   ## Continue looking for viable links ##
			   for aNet in aEvent[aChan]:
			    if aNet in bEvent[aChan]:

			     for aSta in aEvent[aChan][aNet]:
			      if aSta in bEvent[aChan][aNet]:

			       for aStaID in aEvent[aChan][aNet][aSta]:
			        if aStaID in bEvent[aChan][aNet][aSta]:

			         ## Read in traces ##
			         aTrace = ps.readOneHDF5Event(aEvent[aChan][aNet][aSta][aStaID])
			         bTrace = ps.readOneHDF5Event(bEvent[aChan][aNet][aSta][aStaID])

			         eventA = Event(aTrace)
			         if eventA.originStr not in myEventArray.nameArray:
			         	myEventArray.addEvent(eventA)

			         eventB = Event(bTrace)
			         if eventB.originStr not in myEventArray.nameArray:
			         	myEventArray.addEvent(eventB)

			         ## Cross-Correlate master and secondary events ##
# 			         lag,nCC,unCC,unCC_i,powerA,powerB = calcAllCC(aTrace,bTrace)
			         lag,nCC,unCC,unCC_i,powerA,powerB = getOptimalShift(aTrace,bTrace)
			         lag = lag + (aTrace.startTime-aTrace.origin) - (bTrace.startTime-bTrace.origin)

			         ## Calculate weight of cross-correlation ##
			         weight = min([aTrace.quality,bTrace.quality])
			         weight = weight / 4.0

			         ## Write results to (text) log ##
			         dtObs = lag
			         dtPredicted = (aTrace.gcarc - bTrace.gcarc) * settings['gc2km'] * settings['slowness']
			         
			         count += 1
			         ccLog.data(count,aTrace,bTrace,dtObs,dtPredicted,weight,nCC,unCC,unCC_i,powerA,powerB)
			         
			         ## Write results to (digital) log ##
			         resultValues = [weight,lag,nCC,unCC,unCC_i,powerA,powerB]
			         dataLog = writeResults2Dict(dataLog,aEventName,bEventName, \
			         				settings,aTrace,bTrace,aChan,resultValues)

			         ## New Methods ##		             		             
			         ok = shouldLink(aTrace,bTrace,nCC)
			         if ok:
			         	buildDifferenceArray(tempDDArray,eventA,eventB,weight,lag,nCC,unCC,powerA,powerB)

			   ## Channel Level ##
			   azLog = dataLog[aEventName][bEventName]['links']['azCover'][aChan]
			   azLog['cover1'],azLog['cover2'] = calcAzCover(azLog['az'])

			## B Event Level ##
			azLog = dataLog[aEventName][bEventName]['links']['azCover']['total']
			azLog['cover1'],azLog['cover2'] = calcAzCover(azLog['az'])
			links = dataLog[aEventName][bEventName]['links']['totalLinks']['total']

			if (azLog['cover1'] >= settings['minAZ']) and (links >= settings['minLinks']):
				ddArray = ddArray + tempDDArray

	ccLog.close()
	
	return dataLog,ddArray,myEventArray
def getOptimalShift(aTrace,bTrace):
	''' The two signals are cross correlated and the maximum correlation (in an absolute 
		sense) is returned along with the normalized amplitude of the cross correlation. 
		Right now, no checking of sample rates is performed, you can do weird things if 
		they are not equal. The shift is relative to each traces start time (no 
		accounting for b or absolute time differences).
	'''
	
	aSeis = aTrace.data
	bSeis = bTrace.data
	
	npts = max([aTrace.npts,bTrace.npts])
	npwr2 = int(pow(2, np.ceil(np.log(npts) / np.log(2))))
	
	halfpts = (npwr2 / 2)

	df = 1.0 / (npwr2 * aTrace.delta)
	dw = 2*np.pi*df

	##~~~ Allocate a dummy array for the spectrum ~~~##
	aSpec = np.zeros(npwr2)
	aSpec[:len(aSeis)] = aSeis
	
	bSpec = np.zeros(npwr2)
	bSpec[:len(bSeis)] = bSeis
	
	##~~~ Fourier Transform the Seismogram ~~~##
	aSPEC = rfft(aSpec)
	bSPEC = rfft(bSpec)
	
	##~~~ Now cross-correlate ~~~##
	CCOR = np.conj(bSPEC) * aSPEC
	
	##~~~ Now go back to the time domain ~~##
		# Fourier Transform the Seismogram #
	ccor = irfft(CCOR)

		# Scale the cross correlation #
# 	scalefactor = 2 * df * aTrace.delta
# 	ccor = ccor * scalefactor
	
	maxCC = max(abs(ccor))
	maxIndex = np.where(abs(ccor)==maxCC)[0][0]
	timeShift = maxIndex * aTrace.delta

	if (maxIndex < halfpts):
		timeShift = maxIndex * aTrace.delta
	else:
		timeShift = (maxIndex - npwr2) * aTrace.delta
	
	## Normalized and Unnormalized Correlation Coefficient
	aSum = np.sum(aSeis*aSeis); bSum = np.sum(bSeis*bSeis)
	ccNorm = maxCC / (np.sqrt(aSum)*np.sqrt(bSum))
	ccUnNorm = maxCC / aSum
	ccUnNormInv = maxCC / bSum
	
# 	## Plot for QC ##
# 	tempCC = np.append(ccor[halfpts:],ccor[:halfpts])
# 	tempTime = np.arange(-halfpts,halfpts)
# 	
# 	plt.subplot(3,1,1)
# 	plt.plot(tempTime,tempCC)
# 	plt.axis('tight')
# 	plt.grid('on')
# 	plt.subplot(3,1,2)
# 	plt.plot(aTrace.data)
# 	plt.axis('tight')
# 	plt.grid('on')
# 	plt.subplot(3,1,3)
# 	plt.plot(bTrace.data)
# 	plt.axis('tight')
# 	plt.grid('on')
# 	plt.suptitle('{} Lag: {}'.format(aTrace.stationID,timeShift))
# 	plt.show()
# 	
# 	pdb.set_trace()
	
	return timeShift,ccNorm,ccUnNorm,ccUnNormInv,aSum,bSum
def updateDict(log,aName,bName,channel=None,step=0):
	if step == 1:

		log[aName][bName] = {}
		log[aName][bName]['accepted'] = {}
		log[aName][bName]['unused'] = {}

		log[aName][bName]['links'] = {}
		log[aName][bName]['links']['totalLinks'] = {}
		log[aName][bName]['links']['azCover'] = {}

		if bName not in log:
			log[bName] = {}

		log[bName][aName] = {}
		log[bName][aName]['accepted'] = {}
		log[bName][aName]['unused'] = {}

		log[bName][aName]['links'] = {}
		log[bName][aName]['links']['totalLinks'] = {}
		log[bName][aName]['links']['azCover'] = {}

	elif step == 2:
		log[aName][bName]['links']['totalLinks'][channel] = 0
		log[aName][bName]['links']['azCover'][channel] = {}
		log[aName][bName]['links']['azCover'][channel]['az'] = []
		log[aName][bName]['links']['azCover'][channel]['cover1'] = None
		log[aName][bName]['links']['azCover'][channel]['cover2'] = None

		log[bName][aName]['links']['totalLinks'][channel] = 0
		log[bName][aName]['links']['azCover'][channel] = {}
		log[bName][aName]['links']['azCover'][channel]['az'] = []
		log[bName][aName]['links']['azCover'][channel]['cover1'] = None
		log[bName][aName]['links']['azCover'][channel]['cover2'] = None
	
	return log
def writeResults2Dict(masterLog,aName,bName,settings,aTrace,bTrace,channel,results):
	
	## Read in results ##
	[weight,lag,normCC,unnormCC,unnormCC_i,powerA,powerB] = results
			
	## Define a-master and b-master logs
	aLog = masterLog[aName][bName]
	bLog = masterLog[bName][aName]
	
	## Check linking criteria ##
	ok = shouldLink(aTrace,bTrace,normCC)
	
	if ok:
		group = 'accepted'
		aLog['links']['totalLinks'][channel] += 1
		aLog['links']['azCover'][channel]['az'].append(aTrace.az)

		aLog['links']['totalLinks']['total'] += 1
		aLog['links']['azCover']['total']['az'].append(aTrace.az)

		bLog['links']['totalLinks'][channel] += 1
		bLog['links']['azCover'][channel]['az'].append(bTrace.az)

		bLog['links']['totalLinks']['total'] += 1
		bLog['links']['azCover']['total']['az'].append(aTrace.az)
	else:
		group = 'unused'
	
	aTempLog = aLog[group][aTrace.stationID] = {}
	bTempLog = bLog[group][aTrace.stationID] = {}
	
	## Populate dictionary ##
	aTempLog = populateResultsDict(aTempLog,aTrace,bTrace,weight,lag,normCC,unnormCC,powerA,powerB)
	bTempLog = populateResultsDict(bTempLog,bTrace,aTrace,weight,-lag,normCC,unnormCC_i,powerA,powerB)
	
	return masterLog	
def populateResultsDict(log,aTrace,bTrace,weight,lag,normCC,unnormCC,powerA,powerB):

	(dist,az,baz) = gps2DistAzimuth(aTrace.evLat,aTrace.evLon,bTrace.evLat,bTrace.evLon)		             
	dist = dist * 0.001

	log['network'] = aTrace.network
	log['station'] = aTrace.station
	log['location'] = aTrace.staLocation
	log['channel'] = aTrace.channel

	log['abEvDist'] = dist
	log['abEvAz'] = az

	log['staLoc'] = {}
	log['staLoc']['lat'] = aTrace.staLat
	log['staLoc']['lon'] = aTrace.staLon
	log['staLoc']['elev'] = aTrace.staElev
	log['staLoc']['depth'] = aTrace.staDepth

	log['aEvent'] = {}
	log['aEvent']['lat'] = aTrace.evLat
	log['aEvent']['lon'] = aTrace.evLon
	log['aEvent']['mag'] = aTrace.mag
	log['aEvent']['gcarc'] = aTrace.gcarc
	log['aEvent']['distance'] = aTrace.distance
	log['aEvent']['az'] = aTrace.az
	log['aEvent']['baz'] = aTrace.baz
	log['aEvent']['depth'] = aTrace.evDepth

	log['bEvent'] = {}
	log['bEvent']['lat'] = bTrace.evLat
	log['bEvent']['lon'] = bTrace.evLon
	log['bEvent']['mag'] = bTrace.mag
	log['bEvent']['gcarc'] = bTrace.gcarc
	log['bEvent']['distance'] = bTrace.distance
	log['bEvent']['az'] = bTrace.az
	log['bEvent']['baz'] = bTrace.baz
	log['bEvent']['depth'] = bTrace.evDepth

	log['weight'] = weight
	log['lag'] = lag
	log['normCC'] = normCC
	log['unnormCC'] = unnormCC
	log['powerA'] = powerA
	log['powerB'] = powerB
	
	return log
##
def testParse(myEventArray,ddArray,matchLoc=True):
	global settings

	fname = '/Users/mcleveland/Documents/Projects/Menard/EventSearch/Events/Graded/'
	fname = fname + 'Test/eqLocateResults/Observations.txt'
# 	fname = 'Observations.txt'	# name of file with list of events
	Obs = open(fname).read()

	observations = {}
	aEvent = None
	bEvent = None
	Obs = Obs.split('\n' )
	for aOb in Obs:
		aOb = aOb.split()
	
		if len(aOb) > 0:
	
			if aOb[0] == 'Linking':
				aEvent = int(aOb[2])
				bEvent = int(aOb[4])
				if aEvent not in observations.keys():
					observations[aEvent] = {}
				observations[aEvent][bEvent] = {}
	
			else:
				shift = 0
				observations[aEvent][bEvent][aOb[3]] = {}
				if len(aOb) > 18:
					obSet = observations[aEvent][bEvent][aOb[3]][aOb[4]] = {}
					shift = 1
				else:
					obSet = observations[aEvent][bEvent][aOb[3]]['--'] = {} 
				obSet['staLat'] = float(aOb[4+shift])
				obSet['staLon'] = float(aOb[5+shift])
				obSet['aLat'] =  float(aOb[6+shift])
				obSet['aLon'] =  float(aOb[7+shift])
				obSet['bLat'] =  float(aOb[8+shift])
				obSet['bLon'] =  float(aOb[9+shift])

				obSet['dtObs'] = float(aOb[10+shift])
				obSet['dtPred'] = float(aOb[11+shift].split(',')[0])
				obSet['dtDiff'] = float(aOb[12+shift])

				obSet['wt'] = float(aOb[13+shift])

				obSet['cc_norm'] = float(aOb[14+shift])
				obSet['cc_un'] = float(aOb[15+shift])
		
				obSet['powerSignalA'] = float(aOb[16+shift])
				obSet['powerSignalB'] = float(aOb[17+shift])

	tempDDArray = []
	slowness = settings['slowness']
	ccLog = CCLog(fileName=settings['path'] + '/ccLog-All.txt')
	masterA = ''
	masterB = ''
	
	tMatch1 = ddArray[-1].aEvent.stationID
	tMatch2 = ddArray[-1].aEvent.originStr
	tMatch3 = ddArray[-1].aEvent.originStr
	
	totalObs = 0
	for aOb in observations:
	 for bOb in observations[aOb]:
	  for aSta in observations[aOb][bOb]:
	   for aLoc in observations[aOb][bOb][aSta]:
		
		ok = False
		
		totalObs += 1
		
		aEventName = myEventArray.nameArray[aOb-1]
		bEventName = myEventArray.nameArray[bOb-1]
		
		for aDD in ddArray:
			aEvTest = aDD.aEvent.originStr
			bEvTest = aDD.bEvent.originStr
			staTest = aDD.aEvent.station.lower()
			locTest = aDD.aEvent.staLocation
			chanTest = aDD.aEvent.channel
			
			if (aEvTest == aEventName) and (bEvTest == bEventName):
				
				if (masterA != aEventName) or (masterB != bEventName):
					masterA = aEventName; masterB = bEventName
					ccLog.linking(aEventName,bEventName)
					count = 0
				else:
					count += 1

				match = False
				if matchLoc:
					if (staTest == aSta.lower()) and (locTest == aLoc) and (chanTest == 'LHZ'):
						match = True
				else:
					if (staTest == aSta.lower()) and (chanTest == 'LHZ'):
						match = True
				
				if match:
						ok = True
						
						eventA = aDD.aEvent
						eventB = aDD.bEvent
					
						anObservation = observations[aOb][bOb][aSta][aLoc]
										
						theDD = DDObservation()

						theDD.aEvent = eventA
						theDD.bEvent = eventB

						theDD.hSlowness = slowness
						theDD.ccorAmplitude = anObservation['cc_norm']
						theDD.ccorUnnormalizedAmplitude = anObservation['cc_un']

						theDD.dtObs = anObservation['dtObs']
					# 	theDD.dtPredicted = anObservation['dtPred']
						theDD.computePredictedDifference()
						theDD.computeDerivatives()
						theDD.qualityValue = anObservation['cc_norm']

						theDD.powerSignalA = anObservation['powerSignalA']
						theDD.powerSignalB = anObservation['powerSignalB']

						tempDDArray.append(theDD)

						ccLog.data(count,eventA,eventB,theDD.dtObs,theDD.dtPredicted,0, \
								theDD.ccorAmplitude,theDD.ccorUnnormalizedAmplitude,0, \
								theDD.powerSignalA,theDD.powerSignalB)
				
						break

# 		if not ok:
# 			print '  ',aOb,aEventName,bOb,bEventName,aSta,aLoc,observations[aOb][bOb][aSta][aLoc]['cc_norm']
# 			print "dataLog['{}']['{}']['accepted'].keys()".format(aEventName,bEventName)
# 			pdb.set_trace()
			
	ccLog.close()
	print '\nTotal Obs:  {}\nTotal Read: {}\n'.format(totalObs,len(tempDDArray))
	return tempDDArray
##
def shouldLink(trace01,trace02,normCC):
	global settings

		## Calculate distance and azimuth between events ##
	(dist,az,baz) = gps2DistAzimuth(trace01.evLat,trace01.evLon,trace02.evLat,trace02.evLon)		             
	dist = dist * 0.001
	
	if (dist <= settings['linkDist']) and (normCC >= settings['minCC']):
		return True
	
	else:
		return False
#
def doIteration(myEventArray,ddArray):
	'''
	Perform inversion.
	'''
	global settings
	
	nEvents = len(myEventArray.events)
	nDiffs  = len(ddArray)

	if (nEvents < 1) or (nDiffs == 0): return

	##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
	##~~~ A. The first call is to compute the optimal work vector size ~~~##	
	##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
	nCols = nEvents * 4; nRows = nDiffs + nCols + 4; lda = nRows; ldb = nRows

	nrhs = 1; lwork = -1; rcond = 0.01	# same as cond ??

	A = np.zeros([nRows,nCols])
	b = np.zeros([ldb,nrhs])

	[A,b,s,rank,lwork,info] = lapack.clapack.sgelss(A,b,rcond,lwork) 
	
	theInitialDDResidualsText = 'Initial DD Residuals\nIndex DD_Obs DD_Pred DD_Obs-DD_Pred\n'

	##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
	##~~~ B. Print out the initial DD's ~~~##
	##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
	for i in np.arange(0,nDiffs):
		dd = ddArray[i]
		theInitialDDResidualsText += ('%03d %9.3f %9.3f %9.3f\n' % \
			(i+1, dd.dtObs, dd.dtPredicted,(dd.dtObs-dd.dtPredicted)))		
	theInitialDDResidualsText += '************************************************\n'

	##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
	##~~~ C. Set the initial data weights ~~~##
	##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
	for i in np.arange(0,nDiffs):
	
		ddArray[i].setWeight()

	##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
	##~~~ D. Iteratively reweight the dd's in the inversion ~~~##
	##~~~		to decrease the impact of outliers			~~~##
	##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
	nIter = 3
	for iter in np.arange(0,nIter):

		##~~~ D.1. Build the inversion matrix and vectors ~~~##
		A,b,p,x,dt,wtA = \
			buildInversionMatrixVectors(ddArray,myEventArray,nRows,nCols,ldb,nrhs,nDiffs)
					
		##~~~ D.2. The minimum length constraint ~~~##
		row = nDiffs

		for i in np.arange(0,nCols):
			A[row+i][i] = settings['minLengthWt']

		##~~~ D.3. The constant centroid constraint ~~~##
		row += nCols

		for i in np.arange(0,nEvents):
			col = 4 * i
			A[row][col]     = settings['zeroCentroidWt']
			A[row+1][col+1] = settings['zeroCentroidWt']
			A[row+2][col+2] = settings['zeroCentroidWt']
			A[row+3][col+3] = settings['zeroCentroidWt']
		
		##~~~ D.4. Perform inversion of A matrix ~~~##
		[V,x,s,rank,work,info] = lapack.clapack.sgelss(A,b,cond=rcond,lwork=lwork)
# 		[V,x,s,rank,work,info] = lapack.clapack.sgelss(A,b,cond=0.001,lwork=lwork)
		
		##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Begin Testing ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
# 		[xT,residT,rankT,sT] = np.linalg.lstsq(A,b,rcond)
# 		[xT,residT,rankT,sT] = np.linalg.lstsq(A,b,rcond=0.001)
		
# 		plotInversionResults(nEvents,iter,A,x,b,xT,dt,wtA,pStyle='split')		
		plotInversionResults2(nEvents,iter,A,x,b,wtA,pStyle='split')		
		##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ End Testing ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
		
		##~~~ D.5. Compute the new weights based on the residuals ~~~##
		'''
		I consider anything fitting within 2 seconds to be a good observation perhaps
			this should be a multiplicative factor, since the original weight may reflect
			the data quality.
		'''
		unweightedMisfit = 0
		weightedMisfit = 0
		for i in np.arange(0,nDiffs):

			dd = ddArray[i]
			
			predicted = np.dot(A[i], x[:nCols])
			p[i] = predicted
			
			##~~~ A has the weights built in. I want the weight computed from the raw 
			##~~~ 	DD time misfit, so I remove the weights here
			absMisfit  = np.abs(predicted - b[i])
			
			weightedMisfit += absMisfit*absMisfit
			if dd.weight != 0.0:
				absMisfit /= dd.weight
			else:
				absMisfit = 0
			unweightedMisfit += absMisfit*absMisfit
			
			##~~~ Only update the weights if we are going to do another iteration ~~~##
			wt_cutoff = 3.0;
			if (iter < (nIter - 1)):

				if (absMisfit < wt_cutoff):
					wt = dd.getDistanceDependentWeight()
				else:
					if absMisfit != 0.0:
						wt = dd.getDistanceDependentWeight() * (wt_cutoff / absMisfit)
					else:
						wt = 0.0

				dd.weight = wt

	##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
	##~~~ End of the iteratively reweighted inversion ~~~##
	##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
	wtMisfitText = np.sqrt(weightedMisfit/nDiffs)
	unwtMisfitText = np.sqrt(unweightedMisfit/nDiffs)
	
	theSingularValuesText = ('Singular Values (Rank:%03d)\n' % rank)
	for i in np.arange(0,nCols):

		theSingularValuesText += ('%03d %9.3f\n' % (i+1,s[i]))
    
	
	theLocationPerturbationsText = 'Location perturbations:\n'

	theLocationsText = 'Locations:\n'
	theLocationsText += \
		'index eventName         evLatInit evLonInit |    evLat    evLon   |    newLat   newLon  | kmMv_tot   azMv_tot |   OTshift   kmMoved   azMoved\n'

	for i in np.arange(0,nEvents):

		j = 4 * i
		theLocationPerturbationsText += ('%03d %9.3f %9.3f %9.3f %9.3f\n' % \
                                     (i+1, x[j], x[j+1], x[j+2], x[j+3]))
    	
		''' UPDATE THE EVENTS
		 note the scaling of the longitude value. I am treating the problem
		 as if I had simply weighted the partial by 111.19 - not really doing
		 a change in projection, so I don't use cos(lat)*111.19 in the conversion,
		 I am only doing a scaling.
		'''
		event  = myEventArray.events[i]
		newLat = event.evLat - (x[j]  /settings['gc2km'])
		newLon = event.evLon + (x[j+1]/settings['gc2km'])
		
		##~~~ Should compute the distance it has moved here ~~~##
		(distI,azI,baz) = gps2DistAzimuth(event.evLatInitial,event.evLonInitial,newLat,newLon)
		distI /= 1000.0
		
# 		if i == 5:
# 			pdb.set_trace()

		(dist,az,baz) = gps2DistAzimuth(event.evLat,event.evLon,newLat,newLon)
		dist /= 1000.0
		
		theLocationsText += \
			('%03d %s %9.3f %9.3f | %9.3f %9.3f | %9.3f %9.3f | %9.3f %9.3f | %9.3f %9.1f %9.1f\n' % \
			(i+1,event.originStr,event.evLatInitial,event.evLonInitial, \
			event.evLat,event.evLon,newLat,newLon, \
			distI,azI, \
			(event.originTimeShift + x[j+3]),dist,az ))

		##~~~ Update the locations in the event array ~~~##
		event.updateOrigin(newLat,newLon)
		event.evDepth = 0.0
		##~~ Update the origin timeShift in the event array ~~~##
		event.originTimeShift += x[j+3]
		
# ->	[myController mapEvents];

	##~~~~~~~~~~~~~~~~~~~~~~~##
	##~~~ UPDATE THE DD's ~~~##
	##~~~~~~~~~~~~~~~~~~~~~~~##
	theDDInfoText = updateDDs(ddArray,myEventArray,b,p)
			
# 	print theInitialDDResidualsText
# 	print theSingularValuesText
# 	print theLocationPerturbationsText
# 	print theDDInfoText
	print theLocationsText
	
	printInversionResults(theInitialDDResidualsText,theSingularValuesText, \
							theLocationPerturbationsText,theDDInfoText,theLocationsText, \
							wtMisfitText,unwtMisfitText)	
def buildInversionMatrixVectors(ddArray,myEventArray,nRows,nCols,ldb,nrhs,nDiffs):
	##~~~ Zero the arrays ~~~##
	A = np.zeros([nRows,nCols])
	b = np.zeros([ldb,nrhs])
	p = np.zeros(nRows)
	x = np.zeros([ldb,nrhs])
	
	dt = np.zeros([ldb,nrhs])
	wtA = np.zeros([ldb,nrhs])

	##~~~ Build the inversion matrix and vectors ~~~##
	for row in np.arange(0,nDiffs):
		
		dd = ddArray[row]
		
		i01 = myEventArray.eventIndex(dd.aEvent)
		i02 = myEventArray.eventIndex(dd.bEvent)
		
		dt[row] = dd.dtObs - dd.dtPredicted
		wtA[row] = dd.weight
		
		b[row] = (dd.dtObs - dd.dtPredicted) * dd.weight
					
		'''##~~~ this if the jumping part - not implemented yet ~~~##
		 b[row] += [[dd firstEvent]  originTimeShift] -
		 [[dd secondEvent] originTimeShift]; '''

		derivatives = dd.derivatives
					
		col = i01 * 4
		A[row][col] =  derivatives[0] * dd.weight
		col += 1
		A[row][col] =  derivatives[1] * dd.weight
		col += 1
		A[row][col] =  derivatives[2] * dd.weight
		col += 1
		A[row][col] =  derivatives[3] * dd.weight

		col = i02 * 4;
		A[row][col] = -derivatives[4] * dd.weight
		col += 1
		A[row][col] = -derivatives[5] * dd.weight
		col += 1
		A[row][col] = -derivatives[6] * dd.weight
		col += 1
		A[row][col] = -derivatives[7] * dd.weight
	
	return A,b,p,x,dt,wtA
def updateDDs(ddArray,eventArray,b,p):

	theDDInfoText = 'Index Observed Predicted (obs-pred) weight DD_Obs DD_Pred:\n'

	for i,dd in enumerate(ddArray):
	
		i01 = eventArray.eventIndex(dd.aEvent)
		e01 = eventArray.events[i01]
		dd.aEvent.updateOrigin(e01.evLat,e01.evLon)
		dd.aEvent.originTimeShift = e01.originTimeShift
		
		i02 = eventArray.eventIndex(dd.bEvent)
		e02 = eventArray.events[i02]
		dd.bEvent.updateOrigin(e02.evLat,e02.evLon)
		dd.bEvent.originTimeShift = e02.originTimeShift

		theDDInfoText += ('%03d %9.3f %9.3f %9.3f %8.3f %9.3f %9.3f\n' % \
							(i+1,b[i],p[i],(b[i]-p[i]),dd.weight,dd.dtObs,dd.dtPredicted))

		dd.computePredictedDifference()
		dd.weight = 1.0
		
		dd.computeDerivatives()

	return theDDInfoText
#
def buildDifferenceArray(ddArray,eventA,eventB,weight,lag,nCC,unCC,powerA,powerB):
	global settings
	
	slowness = settings['slowness']
	
	predictedShift = slowness * (eventA.gcarc - eventB.gcarc) * settings['gc2km'] + \
		(eventA.originTimeShift - eventB.originTimeShift)

	theDD = DDObservation()
	
	theDD.aEvent = eventA
	theDD.bEvent = eventB

	theDD.hSlowness = slowness
	theDD.ccorAmplitude = nCC
	theDD.ccorUnnormalizedAmplitude = unCC

	theDD.dtObs = lag
# 	theDD.dtPredicted = predictedShift
	theDD.computePredictedDifference()
	theDD.computeDerivatives()
	theDD.qualityValue = nCC

	theDD.powerSignalA = powerA
	theDD.powerSignalB = powerB
	
	ddArray.append(theDD)
def printInversionResults(initDDResid,singValues,locPert,ddInfo,finalLocations,wtMisfit,unwtMisfit):
	global settings
	
	currentTime = str(datetime.datetime.now())
	
	##~~~ Make Results Directory ~~~##
	textPath = settings['path'] + '/Results/'
	if not os.path.isdir(textPath):
		os.mkdir(textPath)
	
	##~~~ Initial DD Residuals ~~~##
	fname = textPath + 'theInitialDDResiduals.txt'
	if os.path.isfile(fname):
		f = open(fname,'a')
	else:	
		f = open(fname,'w')		
	f.write(currentTime)
	f.write('\nwtMisfit: %0.6f, unwtMisfit: %0.6f\n' % (wtMisfit,unwtMisfit))
	f.write(initDDResid)
	f.write('\n')
	f.close()

	##~~~ Singular Values ~~~##
	fname = textPath + 'theSingularValues.txt'
	if os.path.isfile(fname):
		f = open(fname,'a')
	else:	
		f = open(fname,'w')		
	f.write(currentTime)
	f.write('\nwtMisfit: %0.6f, unwtMisfit: %0.6f\n' % (wtMisfit,unwtMisfit))
	f.write(singValues)
	f.write('\n')
	f.close()

	##~~~ Location Perturbations ~~~##
	fname = textPath + 'theLocationPerturbations.txt'
	if os.path.isfile(fname):
		f = open(fname,'a')
	else:	
		f = open(fname,'w')		
	f.write(currentTime)
	f.write('\nwtMisfit: %0.6f, unwtMisfit: %0.6f\n' % (wtMisfit,unwtMisfit))
	f.write(locPert)
	f.write('\n')
	f.close()

	##~~~ DD Information ~~~##
	fname = textPath + 'theDDInfo.txt'
	if os.path.isfile(fname):
		f = open(fname,'a')
	else:	
		f = open(fname,'w')		
	f.write(currentTime)
	f.write('\nwtMisfit: %0.6f, unwtMisfit: %0.6f\n' % (wtMisfit,unwtMisfit))
	f.write(ddInfo)
	f.write('\n')
	f.close()

	##~~~ Final Locations ~~~##
	fname = textPath + 'theLocationsText.txt'
	if os.path.isfile(fname):
		f = open(fname,'a')
	else:	
		f = open(fname,'w')		
	f.write(currentTime)
	f.write('\nwtMisfit: %0.6f, unwtMisfit: %0.6f\n' % (wtMisfit,unwtMisfit))
	f.write(finalLocations)
	f.write('\n')
	f.close()	
def plotInversionResults(nEvents,iter,A,x,b,xT,dt,wtA,pStyle='split'):
	'''
	Plots 2 figures.
	
	This function requires the following command to be called at some point after this
	function is called:
		>> fig1.savefig('InversionResults_A-Iter_{}.pdf'.format(i), bbox_inches='tight')
		>> fig2.savefig('InversionResults_B-Iter_{}.pdf'.format(i), bbox_inches='tight')
		>> plt.clf()
		>> plt.close()
		>> plt.show()
	'''
	global fig1,fig2
	
	Ax  = np.mat(A)*np.mat(x[:(nEvents*4)])
	AxT = np.mat(A)*np.mat(xT)

	if iter == 0:
		tempMax1 = np.ceil(max(Ax[:(nEvents*4)])); tempMin1 = np.floor(min(Ax[:(nEvents*4)]))
		tempMax2 = np.ceil(max(AxT[:(nEvents*4)])); tempMin2 = np.floor(min(AxT[:(nEvents*4)]))
		minY = min(tempMin1,tempMin2); maxY = max(tempMax1,tempMax2)
	else:
		plt.figure(1)
		plt.subplot(2, 2, 1)
		limits = plt.axis()
		minY = limits[2]; maxY = limits[3]

		tempMax1 = np.ceil(max(Ax[:(nEvents*4)])); tempMin1 = np.floor(min(Ax[:(nEvents*4)]))
		tempMax2 = np.ceil(max(AxT[:(nEvents*4)])); tempMin2 = np.floor(min(AxT[:(nEvents*4)]))
		
		minY = min(minY,tempMin1,tempMin2); maxY = max(maxY,tempMax1,tempMax2)
	
	if pStyle == 'split':
		for i in np.arange(0,nEvents):
			colors1 = ['go','bo','ro']
			colors2 = ['gv','bv','rv']
			colors3 = ['g+','b+','r+']
			colors4 = ['gs','bs','rs']

			j = 4 * i
			
			fig1 = plt.figure(1)
			plt.subplot(2, 2, 1)

			plt.plot(b[j],Ax[j],colors1[iter],label='colat')
			plt.plot(b[j+1],Ax[j+1],colors2[iter],label='lon')
			plt.plot(b[j+2],Ax[j+2],colors3[iter],label='depth')
			plt.plot(b[j+3],Ax[j+3],colors4[iter],label='time')
			plt.xlabel('b')
			plt.ylabel('Ax')
			plt.title('lapack.clapack.sgelss')
			plt.ylim((minY,maxY))
# 			if (iter == 0) and (i == 0):
# 				legend = plt.legend(loc=2,ncol=1,handlelength=1,borderpad=0.5,labelspacing=0.5,columnspacing=0.5)
# 				for label in legend.get_texts():
# 					label.set_fontsize('xx-small')	

			plt.subplot(2, 2, 2)
			plt.plot(b[j],AxT[j],colors1[iter])
			plt.plot(b[j+1],AxT[j+1],colors2[iter])
			plt.plot(b[j+2],AxT[j+2],colors3[iter])
			plt.plot(b[j+3],AxT[j+3],colors4[iter])
			plt.xlabel('b')
			plt.ylabel('Ax')
			plt.title('numpy.linalg.lstsq')		
			plt.ylim((minY,maxY))

			plt.subplot(2, 2, 3)
			plt.plot(x[j],xT[j],colors1[iter])
			plt.plot(x[j+1],xT[j+1],colors2[iter])
			plt.plot(x[j+2],xT[j+2],colors3[iter])
			plt.plot(x[j+3],xT[j+3],colors4[iter])
			plt.xlabel('Lapack')
			plt.ylabel('Numpy')
			plt.title('unweighted x values')		

			plt.subplot(2, 2, 4)
			plt.plot(Ax[j],AxT[j],colors1[iter])
			plt.plot(Ax[j+1],AxT[j+1],colors2[iter])
			plt.plot(Ax[j+2],AxT[j+2],colors3[iter])
			plt.plot(Ax[j+3],AxT[j+3],colors4[iter])
			plt.xlabel('Lapack')
			plt.ylabel('Numpy')
			plt.title('Ax')

			plt.tight_layout()

			fig2 = plt.figure(2)
			plt.subplot(2, 2, 1)
			plt.plot(dt[j],Ax[j],colors1[iter])
			plt.plot(dt[j+1],Ax[j+1],colors2[iter])
			plt.plot(dt[j+2],Ax[j+2],colors3[iter])
			plt.plot(dt[j+3],Ax[j+3],colors4[iter])
			plt.xlabel('dt (unweighted b)')
			plt.ylabel('Ax')
			plt.title('lapack.clapack.sgelss')		

			plt.subplot(2, 2, 2)
			plt.plot(dt[j],AxT[j],colors1[iter])
			plt.plot(dt[j+1],AxT[j+1],colors2[iter])
			plt.plot(dt[j+2],AxT[j+2],colors3[iter])
			plt.plot(dt[j+3],AxT[j+3],colors4[iter])
			plt.xlabel('dt (unweighted b)')
			plt.ylabel('Ax')
			plt.title('numpy.linalg.lstsq')		
	
			plt.subplot(2, 2, 3)
			plt.plot(x[j]*wtA[j],xT[j]*wtA[j],colors1[iter])
			plt.plot(x[j+1]*wtA[j+1],xT[j+1]*wtA[j+1],colors2[iter])
			plt.plot(x[j+2]*wtA[j+2],xT[j+2]*wtA[j+2],colors3[iter])
			plt.plot(x[j+3]*wtA[j+3],xT[j+3]*wtA[j+3],colors4[iter])
			plt.xlabel('Lapack')
			plt.ylabel('Numpy')
			plt.title('weighted x values')		
	
			plt.subplot(2, 2, 4)
			plt.plot(Ax[j]*wtA[j],AxT[j]*wtA[j],colors1[iter])
			plt.plot(Ax[j+1]*wtA[j+1],AxT[j+1]*wtA[j+1],colors2[iter])
			plt.plot(Ax[j+2]*wtA[j+2],AxT[j+2]*wtA[j+2],colors3[iter])
			plt.plot(Ax[j+3]*wtA[j+3],AxT[j+3]*wtA[j+3],colors4[iter])
			plt.xlabel('Lapack')
			plt.ylabel('Numpy')
			plt.title('Ax')
	
	else:			
		colors = ['go','bo','ro']

		fig1 = plt.figure(1)
		plt.subplot(2, 2, 1)
		Ax = np.array(np.mat(A)*np.mat(x[:(nEvents*4)]))
		plt.plot(b[:nEvents*4],Ax[:nEvents*4],colors[iter])
		plt.xlabel('b')
		plt.ylabel('Ax')
		plt.title('Lapack')		

		plt.subplot(2, 2, 2)
		AxT = np.array(np.mat(A)*np.mat(xT))
		plt.plot(b[:nEvents*4],AxT[:nEvents*4],colors[iter])
		plt.xlabel('b')
		plt.ylabel('Ax')
		plt.title('Numpy')		

		plt.subplot(2, 2, 3)
		plt.plot(x[:(nEvents*4)],xT,colors[iter])
		plt.xlabel('Lapack')
		plt.ylabel('Numpy')
		plt.title('unweighted x values')		

		plt.subplot(2, 2, 4)
		plt.plot(Ax[:nEvents*4],AxT[:nEvents*4],colors[iter])
		plt.xlabel('Lapack')
		plt.ylabel('Numpy')
		plt.title('Ax')

		plt.tight_layout()

		fig2 = plt.figure(2)
		plt.subplot(2, 2, 1)
		plt.plot(dt[:nEvents*4],Ax[:nEvents*4],colors[iter])
		plt.xlabel('dt (unweighted b)')
		plt.ylabel('Ax')
		plt.title('Lapack')		

		plt.subplot(2, 2, 2)
		plt.plot(dt[:nEvents*4],AxT[:nEvents*4],colors[iter])
		plt.xlabel('b')
		plt.ylabel('Ax')
		plt.title('Numpy')		
	
		plt.subplot(2, 2, 3)
		plt.plot(x[:(nEvents*4)]*wtA[:(nEvents*4)],xT*wtA[:(nEvents*4)],colors[iter])
		plt.xlabel('Lapack')
		plt.ylabel('Numpy')
		plt.title('weighted x values')		
	
		plt.subplot(2, 2, 4)
		plt.plot(Ax[:(nEvents*4)]*wtA[:(nEvents*4)],AxT[:(nEvents*4)]*wtA[:(nEvents*4)],colors[iter])
		plt.xlabel('Lapack')
		plt.ylabel('Numpy')
		plt.title('Ax')
	
	plt.tight_layout()
def plotInversionResults2(nEvents,iter,A,x,b,wtA,pStyle='split'):
	'''
	Plots a single figure
	
	This function requires the following command to be called at some point after this
	function is called:
		>> fig1.savefig('InversionResults_A-Iter_{}.pdf'.format(i), bbox_inches='tight')
		>> plt.clf()
		>> plt.close()
		>> plt.show()
	'''
	global fig1
	
	Ax  = np.mat(A)*np.mat(x[:(nEvents*4)])

	if iter == 0:
		maxY = np.ceil(max(Ax[:(nEvents*4)])); minY = np.floor(min(Ax[:(nEvents*4)]))
	else:
		plt.figure(1)
		plt.subplot(2, 2, 1)
		limits = plt.axis()
		minY = limits[2]; maxY = limits[3]

		tempMax1 = np.ceil(max(Ax[:(nEvents*4)])); tempMin1 = np.floor(min(Ax[:(nEvents*4)]))
		
		minY = min(minY,tempMin1); maxY = max(maxY,tempMax1)
	
	if pStyle == 'split':
		events = np.arange(0,nEvents)

		colors1 = ['go','bo','ro']
		colors2 = ['gv','bv','rv']
		colors3 = ['g+','b+','r+']
		colors4 = ['gs','bs','rs']

		j = 4 * iter
		
		fig1 = plt.figure(1)
		plt.subplot(2, 2, 1)

		j = events * 4
		plt.plot(b[j],Ax[j],colors1[iter],label='colat')
		plt.plot(b[j+1],Ax[j+1],colors2[iter],label='lon')
		plt.plot(b[j+2],Ax[j+2],colors3[iter],label='depth')
		plt.plot(b[j+3],Ax[j+3],colors4[iter],label='time')
		plt.xlabel('b')
		plt.ylabel('Ax')
		plt.title('lapack.clapack.sgelss')
		plt.ylim((minY,maxY))
		plt.grid('on')

		plt.subplot(2, 2, 2)
		data = Ax[j] - b[j]
		plt.plot(j,data,colors1[iter])
		plt.xlabel('Index')
		plt.ylabel('Ax - b')
		plt.title('Resid: Colatitude: %0.4f' % (np.std(data)) )		
		plt.ylim((-2,2))
		plt.grid('on')

		plt.subplot(2, 2, 3)
		data = Ax[j+1] - b[j+1]
		plt.plot(j+1,data,colors2[iter])
		plt.xlabel('Index')
		plt.ylabel('Ax - b')
		plt.title('Resid: Longitude: %0.4f' % (np.std(data)) )		
		plt.ylim((-2,2))
		plt.grid('on')

		plt.subplot(2, 2, 4)
		data = Ax[j+3] - b[j+3]
		plt.plot(j+3,data,colors4[iter])
		plt.xlabel('Index')
		plt.ylabel('Ax - b')
		plt.title('Resid: Time: %0.4f' % (np.std(data)) )	
		plt.ylim((-2.5,2.5))
		plt.grid('on')
		
	else:			
		colors = ['go','bo','ro']

		fig1 = plt.figure(1)
		plt.subplot(2, 2, 1)
		Ax = np.array(np.mat(A)*np.mat(x[:(nEvents*4)]))
		plt.plot(b[:nEvents*4],Ax[:nEvents*4],colors[iter])
		plt.xlabel('b')
		plt.ylabel('Ax')
		plt.title('Lapack')		

		plt.subplot(2, 2, 2)
		AxT = np.array(np.mat(A)*np.mat(xT))
		plt.plot(b[:nEvents*4],AxT[:nEvents*4],colors[iter])
		plt.xlabel('b')
		plt.ylabel('Ax')
		plt.title('Numpy')		

		plt.subplot(2, 2, 3)
		plt.plot(dt[:nEvents*4],Ax[:nEvents*4],colors[iter])
		plt.xlabel('dt (unweighted b)')
		plt.ylabel('Ax')
		plt.title('Lapack')		

		plt.subplot(2, 2, 4)
		plt.plot(dt[:nEvents*4],AxT[:nEvents*4],colors[iter])
		plt.xlabel('b')
		plt.ylabel('Ax')
		plt.title('Numpy')		
		
	plt.tight_layout()
##
def reBuildDifferenceArray(dataLog,myEventArray):
	global settings
	
	newDDArray = []
	
	for a in sorted(dataLog):

		for b in sorted(dataLog[a]):
			
			azCover = dataLog[a][b]['links']['azCover']['total']['cover1']
			links = dataLog[a][b]['links']['totalLinks']['total']
			if (azCover >= settings['minAZ']) and (links >= settings['minLinks']):
			
				for aSet in ['unused','accepted']:
					for link in sorted(dataLog[a][b][aSet]):
						aLink = dataLog[a][b][aSet][link]
						
						eventA = myEventArray.events[myEventArray.eventIndexStr(a)]
						eventB = myEventArray.events[myEventArray.eventIndexStr(b)]

						if shouldLink(eventA,eventB,aLink['normCC']):
														
							slowness = settings['slowness']
	
							predictedShift = slowness * (eventA.gcarc - eventB.gcarc) * \
								settings['gc2km'] + \
								(eventA.originTimeShift - eventB.originTimeShift)

							theDD = None
							theDD = DDObservation()

							theDD.aEvent = deepcopy(eventA)
							theDD.bEvent = deepcopy(eventB)
							theDD.aEvent = reBuildEventInformation(theDD.aEvent,aLink,link)
							theDD.bEvent = reBuildEventInformation(theDD.bEvent,aLink,link)

							theDD.hSlowness = slowness
							theDD.ccorAmplitude = aLink['normCC']
							theDD.ccorUnnormalizedAmplitude = aLink['unnormCC']

							theDD.dtObs = aLink['lag']
# 							theDD.dtPredicted = predictedShift
							theDD.computePredictedDifference()
		
							theDD.computeDerivatives()
							theDD.qualityValue = aLink['normCC']

							theDD.powerSignalA = aLink['powerA']
							theDD.powerSignalB = aLink['powerB']
	
							newDDArray.append(theDD)

	return newDDArray
def reBuildEventInformation(ddEvent,newEvent,staID):
	ddEvent.stationID   = staID
	ddEvent.network     = newEvent['network']
	ddEvent.station     = newEvent['station']
	ddEvent.staLocation = newEvent['location']
	ddEvent.channel     = newEvent['channel']
	ddEvent.staLat      = newEvent['staLoc']['lat']
	ddEvent.staLon      = newEvent['staLoc']['lon']
	ddEvent.staElev     = newEvent['staLoc']['elev']
	ddEvent.staDepth    = newEvent['staLoc']['depth']

	ddEvent.updateOrigin(ddEvent.evLat,ddEvent.evLon)
		
	'''
	The event information from myEventArray may not be the same station information
		so unrequired information will be removed.
	'''
	ddEvent.calibration = None
	ddEvent.componentAzNorth = None
	ddEvent.componentIncidentAngleVertical = None
	ddEvent.quality    = None
	ddEvent.idep       = None
	ddEvent.npts       = None
	ddEvent.sampleRate = None
	ddEvent.nyquist    = None
	ddEvent.delta      = None
	ddEvent.startTime  = None
	ddEvent.endTime    = None
	ddEvent.refTime    = None
	ddEvent.b          = None
	ddEvent.e          = None
	ddEvent.minAmp     = None
	ddEvent.maxAmp     = None
	ddEvent.meanAmp    = None
	ddEvent.fileType   = None
	ddEvent.processing = None

	return ddEvent
	
## 3. Plot Correlation Values ##
from matplotlib.widgets import RadioButtons
from matplotlib.widgets import CheckButtons

def findAzLag(event_A,event_B,log,set='accepted'):
	azimuth = {}; lag = {}; cc = {}
	evDist = None; evAz = None
	
	for a in log[event_A][event_B][set]:
		tempAz  = log[event_A][event_B][set][a]['aEvent']['az']
		tempLag = log[event_A][event_B][set][a]['lag']
		tempCC = log[event_A][event_B][set][a]['normCC']

		evDist = log[event_A][event_B][set][a]['abEvDist']
		evAz = log[event_A][event_B][set][a]['abEvAz']
		
		chan = log[event_A][event_B][set][a]['channel']
		if chan not in azimuth:
			azimuth[chan] = []
			lag[chan] = []
			cc[chan] = []
		azimuth[chan].append(tempAz)
		lag[chan].append(tempLag)
		cc[chan].append(tempCC)
	
	if set=='accepted':
		links = {}
		for aChan in log[event_A][event_B]['links']['totalLinks']:
			links[aChan] = log[event_A][event_B]['links']['totalLinks'][aChan]
	else:
		links = []
	
	evLocs = []
	if len(log[event_A][event_B]['accepted']) > 0:
		tSet = 'accepted'
	else:
		tSet = 'unused'
	temp = log[event_A][event_B][tSet].keys()
	evLocs.append(log[event_A][event_B][tSet][temp[0]]['aEvent']['lat'])
	evLocs.append(log[event_A][event_B][tSet][temp[0]]['aEvent']['lon'])
	evLocs.append(log[event_A][event_B][tSet][temp[0]]['bEvent']['lat'])
	evLocs.append(log[event_A][event_B][tSet][temp[0]]['bEvent']['lon'])
		
	return azimuth,lag,cc,links,evLocs,evDist,evAz
def updatePlot(aEvent,bEvent,log):
	
	if bEvent in log[aEvent]:
		azimuth_pick,lag_pick,links,locs = findAzLag(aEvent,bEvent,log,set='accepted')
		azimuth_un,lag_un,dummy,dummy = findAzLag(aEvent,bEvent,log,set='unused')
		
		for aChan in l_pick:
			if aChan in azimuth_pick:
				
				l_pick[aChan].set_xdata(azimuth_pick[aChan])
				l_pick[aChan].set_ydata(lag_pick[aChan])
			else:
				l_pick[aChan].set_xdata([])
				l_pick[aChan].set_ydata([])

			if aChan in azimuth_un:
				l_un[aChan].set_xdata(azimuth_un[aChan])
				l_un[aChan].set_ydata(lag_un[aChan])
			else:
				l_un[aChan].set_xdata([])
				l_un[aChan].set_ydata([])
					
		linkT = ''
		for aChan in links:
			linkT = linkT+aChan+': '+str(links[aChan])+' '
		titleText = aEvent+' & '+bEvent + '\nLinks: '+linkT
		aTitle.set_text(titleText)
				
	else:
		for aChan in l_pick:
			l_pick[aChan].set_xdata([])
			l_pick[aChan].set_ydata([])

			l_un[aChan].set_xdata([])
			l_un[aChan].set_ydata([])

		titleText = aEvent+' & '+bEvent + '\nNot Linked'
		aTitle.set_text(titleText)
	
	fig.canvas.draw()		
def aDataSet(event):
	global aGlobal
	aEvent = aGlobal = event
	bEvent = bGlobal
	
	updatePlot(aEvent,bEvent,dataLog)
def bDataSet(event):
	global bGlobal
	aEvent = aGlobal
	bEvent = bGlobal = event

	updatePlot(aEvent,bEvent,dataLog)

def plotAllCorrValues(log):
	
	plotPath = settings['path'] + '/CorrPlots/'
	makeDir(plotPath)
	
	azimuthArray = np.arange(0, 380, 30)
		
	events = sorted(log.keys())
	for aEvent in events:
		
		print 'Plotting all {} pairs'.format(aEvent)
		
		plotEventPath = plotPath + aEvent + '/'
		makeDir(plotEventPath)

		for bEvent in events:
			if aEvent != bEvent:

				##~~~ ax1: Plot Traveltime Difference (s) vs Azimuth (deg) ~~~##
				##~~~ ax2: Plot Traveltime Difference (s) vs Azimuth (deg) ~~~##
				ax1 = plt.subplot2grid((5,1),(0,0),rowspan=4)
				ax2 = plt.subplot2grid((5,1),(4,0))

				if bEvent in log[aEvent]:
					azimuth_pick,lag_pick,cc_pick,links,locs,evDist1,evAz1 = \
										findAzLag(aEvent,bEvent,log,set='accepted')
					azimuth_un,lag_un,cc_un,dummy,dummy,evDist2,evAz2 = \
										findAzLag(aEvent,bEvent,log,set='unused')
					
					## Define distance and azimuth between two events ##
					if evDist1:
						evDist = evDist1
						evAz   = evAz1
					else:
						evDist = evDist2
						evAz   = evAz2
						
					## Plot measurements ##
					obsAz_P = []; obsLag_P = []
					obsAz_A = []; obsLag_A = []
					symbols = ['o','v','*']
					index = 0
					for aChan in settings['channels']:
						aChan = aChan.upper()
						
						if aChan in azimuth_pick:
							pick_x = azimuth_pick[aChan]
							pick_y = lag_pick[aChan]
							pick_y2 = cc_pick[aChan]
							
							obsAz_P += pick_x
							obsLag_P += pick_y

							obsAz_A += pick_x
							obsLag_A += pick_y
			
						else:
							pick_x = []
							pick_y = []
							pick_y2 = []
							
						if aChan in azimuth_un:
							unpick_x = azimuth_un[aChan]
							unpick_y = lag_un[aChan]
							unpick_y2 = cc_un[aChan]

							obsAz_A += unpick_x
							obsLag_A += unpick_y
							
						else:
							unpick_x = []
							unpick_y = []
							unpick_y2 = []

						ax1.plot(pick_x,pick_y,symbols[index],lw=2,color='red')
						ax1.plot(unpick_x,unpick_y,symbols[index],lw=0.5,color='gray')

						ax2.plot(pick_x,pick_y2,symbols[index],lw=2,color='red')
						ax2.plot(unpick_x,unpick_y2,symbols[index],lw=0.5,color='gray')

						index += 1

					## Plot initial locations sine curve ##
					azRange = np.arange(0, 380, 2)
					initAmp   = settings['slowness'] * evDist
					initPhase = settings['deg_to_rad'] * evAz
					initXdata,initYdata = calcSineCurve(azRange,initAmp,initPhase,0)
					ax1.plot(initXdata,initYdata,'gray')
					
					## Plot improved optimized locations sine curve for accepted observations ##
					if len(obsAz_P) > 0:
						(optX_P,optY_P,phaseEst_P,amplitudeEst_P,biasEst_P) = \
								fitSine(obsAz_P,obsLag_P)						
						ax1.plot(optX_P,optY_P,'k')
						stats =  '%-17s Distance  Azimuth\n' % ('')
						stats += '%-12s %5.1f km  %5.1f$^{\circ}$\n' % ('Initial',evDist,evAz)
						stats += '%-8s %5.1f km  %5.1f$^{\circ}$\n' % \
							('Optimal',amplitudeEst_P*settings['slowness'],phaseEst_P)
						stats += 'OT Shift    %5.1f s\n' % (biasEst_P)
# 						stats += 'RMS Misfit  %0.1f s' % (0)
						stats += 'gr=Inititial, blk=Optimal'
# 						stats += 'gr=Init,blk=OptPick,gr=OptAll'
						
						bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.75)
						ax1.text(.02,.98,stats,fontsize=8,bbox=bbox_props, \
									transform=ax1.transAxes,ha='left',va='top')

					## Plot improved optimized locations sine curve for all observations ##
# 					if len(obsAz_A) > 0:
# 						(optX_A,optY_A,phaseEst_A,amplitudeEst_A,biasEst_A) = \
# 							fitSine(obsAz_A,obsLag_A)						
# 						ax1.plot(optX_A,optY_A,':g')
						
					## Write titles with number of links ##
					if evDist <= settings['linkDist']:
						linkT = ''
						for aChan in links:
							if aChan != 'total':
								linkT = linkT+aChan+': '+str(links[aChan])+' '
						linkT = linkT+'total'+': '+str(links['total'])+' '
						titleText = aEvent+' & '+bEvent + '\nLinks: '+linkT
					
					else:
						titleText = aEvent+' & '+bEvent + \
								'\nNot Linked: Event distance %0.2f km.' % (evDist)

				else:
					ax1.plot([],[])
					ax2.plot([],[])

					titleText = aEvent+' & '+bEvent + '\nNot Linked: No common links.'
					aTitle.set_text(titleText)

				##~~~ ax1: Plot Traveltime Difference (s) vs Azimuth (deg) ~~~##
				ax1.set_xlim([0,360])
				ax1.xaxis.set_ticks(azimuthArray)
				ax1.xaxis.set_ticklabels([])
				
				ax1.set_ylim([-50,50])
				
				ax1.grid(True)
				
				zeroLine = ax1.plot([0,360],[0,0],lw=1,color='black')

				aTitle = ax1.text(.5,1.01,titleText,fontsize=15,transform=ax1.transAxes,ha='center')
				
				ax1.set_ylabel('Traveltime Difference (s)')

				##~~~ ax2: Plot Traveltime Difference (s) vs Azimuth (deg) ~~~##				
				ax2.set_xlim([0,360])
				ax2.xaxis.set_ticks(azimuthArray)
				
				ax2.set_ylim([0.0,1.1])
				ax2.yaxis.set_ticks(np.arange(0, 1.2, 0.5))
				
				ax2.grid(True)

				ccThreshLine = ax2.plot([0,360],[settings['minCC'],settings['minCC']],lw=1,color='black')

				ax2.set_ylabel('CCorrelation')
				ax2.set_xlabel('Azimuth ($^{\circ}$)')

				##~~~ Plot ~~~##
				plt.savefig(plotEventPath+'observations_{}-{}.pdf'.format(aEvent,bEvent))
				plt.clf()
				plt.close()

def fitSine(azList,yList):
	
	deg_to_rad = settings['deg_to_rad']
	azList = np.array(azList) * deg_to_rad
	
	b = np.matrix(yList).T
	rows = [ [np.sin(az), np.cos(az), 1] for az in azList]
	A = np.matrix(rows)
	
	(w,residuals,rank,sing_vals) = np.linalg.lstsq(A,b)
	
	phase = np.arctan2(w[1,0],w[0,0])*180.0/np.pi
	amplitude = np.linalg.norm([w[0,0],w[1,0]],2)
	shift = w[2,0]

	azRange = np.arange(0, 380, 2)
	optXdata,optYdata = calcSineCurve(azRange,amplitude,phase,shift)
	
	return optXdata,optYdata,phase,amplitude,shift
def calcSineCurve(azRange,amplitude,phase,shift):
	deg_to_rad = settings['deg_to_rad']
	
	xdata = azRange
	ydata = amplitude * np.sin(deg_to_rad *(azRange - phase)) + shift

	return xdata,ydata
	
## io ##
def save2HDF5(dataStruct):
	global settings
	
# 	traceObject = dataStruct.waveforms['LHZ']
# 	ps.save2HDF5(traceObject)
	
	ok = False
	
	for aChan in dataStruct.waveforms:

		if isinstance(dataStruct.waveforms[aChan][0],ps.Trace):
			for aTrace in dataStruct.waveforms[aChan]:
				
				eventName = aTrace.origin.strftime('E%Y-%m-%d-%H-%M-%S')
				fname = settings['path'] + settings['wavesDir'] + '/' + eventName + '.h5'
# 				fname = eventName + '.h5'
				
				if not ok:
					ok = True
										
					if os.path.isfile(fname):
						os.remove(fname)
						hdf5File = h5py.File(fname, 'w')
					else:
						hdf5File = h5py.File(fname, 'w')
				
				save_a_HDF5(aTrace,hdf5File)
	
	k = hdf5File.require_group('Settings')		
	for aSetting in settings:
		k.attrs[aSetting] = settings[aSetting]

	#~~ Close HDF5 objects ~~#
	hdf5File.close()				
def save_a_HDF5(trace,fname):
	try:
		g = fname.require_group(trace.channel)
		h = g.require_group(trace.network)
		j = h.require_group(trace.station)
	except TypeError:
		pass
			
	j.create_dataset(trace.stationID, data=trace.data, compression=None)	
	ps.populateHDF5Attributes(trace,j[trace.stationID])
def readHDF5(h5pyFile,path=False):
	'''
	eventList = readHDF5(h5pyFile,path=False)
	
	path = True/False
		if True: include a path address to the processing record.
	'''
	dataStruct = Waveforms()
	
	hdf5File = h5py.File(h5pyFile, 'r')

	if path == True:
		path = h5pyFile
	
	for aChannel in hdf5File.keys():
		if aChannel not in ['Settings']:
			if aChannel not in dataStruct.waveforms:
				dataStruct.waveforms[aChannel] = []
			for aNetwork in hdf5File[aChannel].keys():
				for aStation in hdf5File[aChannel][aNetwork].keys():
					for aWave in hdf5File[aChannel][aNetwork][aStation].keys():
						waveform = hdf5File[aChannel][aNetwork][aStation][aWave]
						newTrace = ps.readOneHDF5Event(waveform,path=path)
						dataStruct.waveforms[aChannel].append(newTrace)
	
	for aSetting in hdf5File['Settings'].attrs.keys():
		dataStruct.settings[aSetting] = hdf5File['Settings'].attrs[aSetting]
		
	return dataStruct	
def readDataTextFile():
	print 'To be done.'
def readPickle(path):
	file = open(path,'rb')
	data = pickle.load(file)
	
	return data

## Utilities ##
def makeDir(directory):
	try:
		os.makedirs(directory)
	except OSError:
		if os.path.exists(directory):
			pass
		else:
			raise
def calcAzCover(array):
	if len(array) < 2:
		azCover1 = 0
		azCover2 = 0
	
	else:
		tempCover = []
		
		index = 1
		for a in array[:-1]:
			for b in array[index:]:
				aMeasure = abs(a - b)
				bMeasure = 360 - max([a,b]) + min([a,b])
				
				tempCover.append(min([aMeasure,bMeasure]))
			index += 1
					
		azCover1 = sorted(tempCover,reverse=True)[0]
		
		if len(tempCover) > 1:
			azCover2 = sorted(tempCover,reverse=True)[1]
		else:
			azCover2 = azCover1
	
	return azCover1,azCover2

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
dataStruct = Waveforms()

##~~~ Set Path ~~~##
# dataStruct.settings['path'] = '/Users/mcleveland/Documents/Projects/Menard/EventSearch/Events/Graded/Test'
# dataStruct.settings['path'] = '/Users/mcleveland/Documents/Projects/Menard/EventSearch/Events/Graded/test2'
dataStruct.settings['path'] = '/Users/mcleveland/Documents/Projects/Menard/EventSearch/Events/Graded/Both'

dataStruct.settings['pathPrefix'] = '/E*'
# dataStruct.settings['dataSubDir'] = 'Dsp_Use'
dataStruct.settings['dataSubDir'] = 'Dsp'
dataStruct.settings['fileSuffix'] = '.sac'

##~~~ SAC and HDF5 file save location ~~~##
dataStruct.settings['wavesDir']  = '/Waveforms'

##~~~ Define Period Band ~~~##
dataStruct.settings['shortPeriod'] = 30
dataStruct.settings['longPeriod']  = 80

##~~~ Define Group Velocity Range (km/s) ~~~##
	# Rayleigh #
dataStruct.settings['rGvLow'] = 3.0
dataStruct.settings['rGvHi']  = 5.0

	# Love #
dataStruct.settings['gGvLow'] = 3.0
dataStruct.settings['gGvHi']  = 5.0

##~~~ Define Slowness ~~~##
dataStruct.settings['slowness'] = 0.24

##~~~ Define Quality (False if not defined) ~~~##
dataStruct.settings['quality'] = 2

##~~~ Define Channel ~~~##
dataStruct.settings['channels'] = ['lhz','lht']

##~~~ Define linking distance (km) ~~~##
dataStruct.settings['linkDist'] = 120

##~~~ Define minimum acceptable CC coefficient ~~~##
dataStruct.settings['minCC'] = 0.90

##~~~ Define minimum number of links ~~~##
dataStruct.settings['minLinks'] = 12

##~~~ Define minimum azimuthal coverage of links (degrees) ~~~##
dataStruct.settings['minAZ'] = 50

##~~~ Weight by distance (True/False) ~~~##
dataStruct.settings['weightByDistance'] = False

##~~~ Define zero centroid weight ~~~##
dataStruct.settings['zeroCentroidWt'] = 0.000

##~~~ Define minimum length weight ~~~##
dataStruct.settings['minLengthWt'] = 0.000

##~~~ Define GCarc to km conversion ~~~##
dataStruct.settings['gc2km'] = 111.19

##~~~ Define degrees to radians conversion ~~~##
dataStruct.settings['deg_to_rad'] = np.arccos(-1.0) / 180.0

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
##~~~ Work Flow (which steps to include) ~~~##
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
# workFlow = [1,2,3]
workFlow = [4,5,6]
# workFlow = [1,2,3,4,6]
# workFlow = [1,2,3,4,5,6]

global settings
settings = dataStruct.settings

pdb.set_trace()

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##

##~~~ 1. QC, Prepare, and Save *ALL* Data to HDF5 file ~~~##
if 1 in workFlow:
	print '\nPreparing data\n'
	prepData(dataStruct)

##~~~ 2. Find *All* Viable Links, Compute CC, Write *ALL* to Text, Create Digital ~~~##
##~~~		Log With Links Filtered by Linking Distance and Minimum CC Criteria   ~~~##
##~~~		(reads data in from HDF5 files generated in prepData()   ~~~##
if 2 in workFlow:
	print '\nComputing correlation values\n'
	dataLog,ddArray,myEventArray = matchComputeCC(dataStruct)
	
##~~~ 3. Save (digital) log to Pickle format ~~~##
if 3 in workFlow:
	print '\nSaving data structure(s) as a Pickle file(s)\n'
	pickle.dump( dataLog, open( dataStruct.settings['path']+'/dataLog.pickle', 'wb' ) )
	pickle.dump( ddArray, open( dataStruct.settings['path']+'/ddArray.pickle', 'wb' ) )
	pickle.dump( myEventArray, open( dataStruct.settings['path']+'/myEventArray.pickle', 'wb' ) )

##~~~ 4. Read in dataLog, ddArray, and myEventArray ~~~##
if 4 in workFlow:
	print '\nReading in data structure(s) from Pickle file(s)\n'
	dataLog = readPickle(dataStruct.settings['path']+'/dataLog.pickle')
	ddArray = readPickle(dataStruct.settings['path']+'/ddArray.pickle')
	myEventArray = readPickle(dataStruct.settings['path']+'/myEventArray.pickle')

##~~~ 5. Plot Correlation Values ~~~##
if 5 in workFlow:
	print '\nPlotting data\n'
	plotAllCorrValues(dataLog)

##~~~ 6. Perform the iteration ~~~##
if 6 in workFlow:
	print '\nPerforming iteration(s)\n'

	nIter = 3
	for i in np.arange(0,nIter):
		i += 1
		print '\nIteration: {}\n'.format(i)
		doIteration(myEventArray,ddArray)
		
		fig1.savefig('InversionResults_A-Iter_{}.pdf'.format(i), bbox_inches='tight')
# 		fig2.savefig('InversionResults_B-Iter_{}.pdf'.format(i), bbox_inches='tight')
		plt.clf()
		plt.close()
		plt.show()

##~~~ 7. ~~~##
if 7 in workFlow:
	print '\nPerforming test iteration(s)\n'

	dataStruct.settings['minCC'] = 0.85
	ddArray = reBuildDifferenceArray(dataLog,myEventArray)
	
	tempDDArray = testParse(myEventArray,ddArray,matchLoc=False)
	
	for i in np.arange(0,5):
		doIteration(myEventArray,tempDDArray)
		plt.clf()
		plt.close()
		
# pdb.set_trace()
