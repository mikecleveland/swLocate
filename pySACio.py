#!/bin/python
# -*- coding: utf-8 -*- 

'''
	pySACio.py version 1.0 (original 07 Mar, 2014)
				
		by Mike Cleveland (with contributions from many other sources)

'''
#====================================================================
#====================================================================
import os
import sys
import glob
import datetime as dt
from copy import deepcopy
import numpy as np
from scipy import signal
from scipy.fftpack import hilbert

from obspy.core import read, UTCDateTime
from obspy.core.util import gps2DistAzimuth
from obspy.signal.invsim import cosTaper
from obspy.signal.util import nextpow2
from obspy.sac import SacIO

import h5py

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42	# Set font so it can be edited in Illustrator
mpl.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import matplotlib.dates as md
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from mpl_toolkits.basemap import Basemap

import pdb	# Debug with pdb.set_trace()

from obspy.signal import seisSim

#====================================================================
#====================================================================
class Trace():
	"Trace class."
	
	def __init__(self,a,path=None):
		"Initialize an object: a = [waveform]"	
		
		#~~~~~~~~~~~~~~ Obspy Trace ~~~~~~~~~~~~~~#
		if str(type(a)) == "<class 'obspy.core.trace.Trace'>":		
			
			#~~ Station Information ~~#
			self.network  = a.stats['network']
			self.station  = a.stats['station']
			self.channel  = a.stats['channel']
			if len(self.channel) > 3:
				self.channel = self.channel[-3:]
			self.staLocation = a.stats['location']
			if self.staLocation == '':
				self.staLocation = '--'
			self.stationID = '%s-%s-%s.%s' % (self.network,self.station,self.staLocation,self.channel)
			self.staLat = a.stats['sac']['stla']
			self.staLon = a.stats['sac']['stlo']
			self.staElev = a.stats['sac']['stel']
			self.staDepth = a.stats['sac']['stdp']
			self.calibration = a.stats['calib']
			self.componentAzNorth = a.stats['sac']['cmpaz']
			self.componentIncidentAngleVertical = a.stats['sac']['cmpinc']
		
			#~~ Event Information ~~#
			if a.stats['sac']['iztype'] == 9:
				if a.stats['sac']['o'] != -12345:
					self.origin = UTCDateTime(round(a.stats.starttime + a.stats['sac']['o'],3))
				else:
					print '{}: Uncertain origin time (Trace.__init__)'.format(self.stationID)
					self.origin = -12345				
			elif a.stats['sac']['iztype'] == 11:
				self.origin = a.stats.starttime - a.stats.sac.b
			else:
				print '{}: Uncertain origin time (Trace.__init__)'.format(self.stationID)
				self.origin = -12345
			self.evLat = a.stats['sac']['evla']
			self.evLon = a.stats['sac']['evlo']
			self.evDepth = a.stats['sac']['evdp']
			self.mag   = a.stats['sac']['mag']

			#~~ Distance, GCarc, Azimuth, Back-Azimuth ~~#
			(d,az,baz) = gps2DistAzimuth(a.stats['sac']['evla'],a.stats['sac']['evlo'] \
				,a.stats['sac']['stla'],a.stats['sac']['stlo'])
			self.distance = d/1000.0
			self.gcarc = self.distance/111.19
		
			if a.stats['sac']['az'] != -12345:
				self.az = a.stats['sac']['az']
			else:
				self.az = az
		
			if a.stats['sac']['baz'] != -12345:
				self.baz = a.stats['sac']['baz']
			else:
				self.baz = baz
			
			#~~ Waveform Information ~~#
			self.fileType = a.stats['sac']['iztype']
		
			self.npts = a.data.size
			self.sampleRate = a.stats.sampling_rate
			self.nyquist = a.stats.sampling_rate * 0.5
			self.delta = a.stats.delta
		
			self.idep = -12345
		
			self.startTime = a.stats.starttime
			self.endTime = a.stats.endtime
			if a.stats['sac']['iztype'] == 9:
				self.refTime = self.startTime
			elif a.stats['sac']['iztype'] == 11:
				self.refTime = self.origin
			else:
				self.refTime = -12345
			self.b = a.stats['sac']['b']
			self.e = a.stats['sac']['e']
		
			self.minAmp = a.stats['sac']['depmin']
			self.maxAmp = a.stats['sac']['depmax']
			self.meanAmp= a.stats['sac']['depmen']

			self.quality = a.stats['sac']['iqual']
						
			#~~ Phase Arrivals ~~#
			if a.stats['sac']['iztype'] == 9:
				if a.stats['sac']['ko'].strip() != '-12345':
					self.marker_ko = a.stats['sac']['ko'].strip()
					self.marker_o  = 0
				else:
					self.marker_ko = 'O'
					self.marker_o  = 0

				if a.stats['sac']['ka'].strip() != '-12345':
					self.marker_ka = a.stats['sac']['ka'].strip()
					self.marker_a  = a.stats['sac']['a'] - a.stats['sac']['o']
				else:
					self.marker_ka = ''
					if a.stats['sac']['a'] != -12345:
						self.marker_a  = a.stats['sac']['a'] - a.stats['sac']['o']
					else:
						self.marker_a  = a.stats['sac']['a']
						
				if a.stats['sac']['kt0'].strip() != '-12345':
					self.marker_kt0 = a.stats['sac']['kt0'].strip()
					self.marker_t0  = a.stats['sac']['t0'] - a.stats['sac']['o']
				else:
					self.marker_kt0 = ''
					if a.stats['sac']['t0'] != -12345:
						self.marker_t0  = a.stats['sac']['t0'] - a.stats['sac']['o']
					else:
						self.marker_t0  = a.stats['sac']['t0']
			
				if a.stats['sac']['kt1'].strip() != '-12345':
					self.marker_kt1 = a.stats['sac']['kt1'].strip()
					self.marker_t1  = a.stats['sac']['t1'] - a.stats['sac']['o']
				else:
					self.marker_kt1 = ''
					if a.stats['sac']['t1'] != -12345:
						self.marker_t1  = a.stats['sac']['t1'] - a.stats['sac']['o']
					else:
						self.marker_t1  = a.stats['sac']['t1']

				if a.stats['sac']['kt2'].strip() != '-12345':
					self.marker_kt2 = a.stats['sac']['kt2'].strip()
					self.marker_t2  = a.stats['sac']['t2'] - a.stats['sac']['o']
				else:
					self.marker_kt2 = ''
					if a.stats['sac']['t2'] != -12345:
						self.marker_t2  = a.stats['sac']['t2'] - a.stats['sac']['o']
					else:
						self.marker_t2  = a.stats['sac']['t2']

			if a.stats['sac']['iztype'] == 11:
				if a.stats['sac']['ko'].strip() != '-12345':
					self.marker_ko = a.stats['sac']['ko'].strip()
					self.marker_o  = a.stats['sac']['o']
				else:
					self.marker_ko = 'O'
					self.marker_o  = a.stats['sac']['o']

				if a.stats['sac']['ka'].strip() != '-12345':
					self.marker_ka = a.stats['sac']['ka'].strip()
					self.marker_a  = a.stats['sac']['a']
				else:
					self.marker_ka = ''
					self.marker_a  = a.stats['sac']['a']

				if a.stats['sac']['kt0'].strip() != '-12345':
					self.marker_kt0 = a.stats['sac']['kt0'].strip()
					self.marker_t0  = a.stats['sac']['t0']
				else:
					self.marker_kt0 = ''
					self.marker_t0  = a.stats['sac']['t0']
			
				if a.stats['sac']['kt1'].strip() != '-12345':
					self.marker_kt1 = a.stats['sac']['kt1'].strip()
					self.marker_t1  = a.stats['sac']['t1']
				else:
					self.marker_kt1 = ''
					self.marker_t1  = a.stats['sac']['t1']

				if a.stats['sac']['kt2'].strip() != '-12345':
					self.marker_kt2 = a.stats['sac']['kt2'].strip()
					self.marker_t2  = a.stats['sac']['t2']
				else:
					self.marker_kt2 = ''
					self.marker_t2  = a.stats['sac']['t2']
		
			#~~ Data ~~#
			self.data = a.data

			#~~ Other ~~#
			self.lovrok = bool(a.stats['sac']['lovrok'])
			self.lpspol = bool(a.stats['sac']['lpspol'])
			self.lcalda = bool(a.stats['sac']['lcalda'])
			self.nvhdr  = a.stats['sac']['nvhdr']		

			#~~ Other ~~#
			self.processing = {}
			if path:
				cwd = os.getcwd()
				timeStamp = dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S:%m')
				self.processing[timeStamp] = {}
				self.processing[timeStamp][1] = {}
				self.processing[timeStamp][1] = {'Read':(cwd+'/'+path)}

		#~~~~~~~~~~~~~~ Obspy Trace ~~~~~~~~~~~~~~#
		elif isinstance(a,h5py._hl.dataset.Dataset):		
			
			#~~ Station Information ~~#
			self.network     = a.attrs['network']
			self.station     = a.attrs['station']
			self.channel     = a.attrs['channel']
			self.staLocation = a.attrs['staLocation']
			self.stationID   = a.attrs['stationID']
			self.staLat      = a.attrs['staLat']
			self.staLon      = a.attrs['staLon']
			self.staElev     = a.attrs['staElev']
			self.staDepth    = a.attrs['staDepth']
			self.calibration = a.attrs['calibration']
			self.componentAzNorth = a.attrs['componentAzNorth']
			self.componentIncidentAngleVertical = a.attrs['componentIncidentAngleVertical']
		
			#~~ Event Information ~~#
			self.origin = UTCDateTime(a.attrs['origin'])
			self.evLat  = a.attrs['evLat']
			self.evLon  = a.attrs['evLon']
			self.evDepth= a.attrs['evDepth']
			self.mag    = a.attrs['mag']

			#~~ Distance, GCarc, Azimuth, Back-Azimuth ~~#
			self.distance = a.attrs['distance']
			self.gcarc    = a.attrs['gcarc']
			self.az       = a.attrs['az']
			self.baz      = a.attrs['baz']
			
			#~~ Waveform Information ~~#
			self.fileType = a.attrs['fileType']
		
			self.npts      = a.attrs['npts']
			self.sampleRate= a.attrs['sampleRate']
			self.nyquist   = a.attrs['nyquist']
			self.delta     = a.attrs['delta']
		
			self.idep = a.attrs['idep']
		
			self.startTime = UTCDateTime(a.attrs['startTime'])
			self.endTime   = UTCDateTime(a.attrs['endTime'])
			self.refTime = UTCDateTime(a.attrs['refTime'])
			self.b         = a.attrs['b']
			self.e         = a.attrs['e']
		
			self.minAmp = a.attrs['minAmp']
			self.maxAmp = a.attrs['maxAmp']
			self.meanAmp= a.attrs['meanAmp']

			self.quality = a.attrs['quality']
						
			#~~ Phase Arrivals ~~#
			self.marker_ko  = a.attrs['marker_ko']
			self.marker_o   = a.attrs['marker_o']

			self.marker_ka  = a.attrs['marker_ka']
			self.marker_a   = a.attrs['marker_a']

			self.marker_kt0 = a.attrs['marker_kt0']
			self.marker_t0  = a.attrs['marker_t0']
		
			self.marker_kt1 = a.attrs['marker_kt1']
			self.marker_t1  = a.attrs['marker_t1']

			self.marker_kt2 = a.attrs['marker_kt2']
			self.marker_t2  = a.attrs['marker_t2']
		
			#~~ Data ~~#
			self.data = a.value

			#~~ Other ~~#
			self.lovrok = a.attrs['lovrok']
			self.lpspol = a.attrs['lpspol']
			self.lcalda = a.attrs['lcalda']
			self.nvhdr  = a.attrs['nvhdr']

			#~~ Other ~~#
			self.processing = {}
			
			if len(a.attrs['processing']) > 0:
				processEvents = a.attrs['processing'].split('|')[1:]
				
				for aEntry in processEvents:
					aEntry = aEntry.split(',')
					
					timeStamp = aEntry[0]
					aStep = int(aEntry[1])
					aProcess = aEntry[2]
					aSetting = aEntry[3]					

					if timeStamp in self.processing:
						if aStep in self.processing[timeStamp]:
							if aProcess not in self.processing[timeStamp][aStep]:
								self.processing[timeStamp][aStep][aProcess] = {}	
						
						else:
							self.processing[timeStamp][aStep] = {}
							self.processing[timeStamp][aStep][aProcess] = {}
					
					else:
						self.processing[timeStamp] = {}
						self.processing[timeStamp][aStep] = {}
						self.processing[timeStamp][aStep][aProcess] = {}
					
					if len(aEntry) == 4:
						self.processing[timeStamp][aStep][aProcess] = [aSetting]

					else:
						aDetail = aEntry[4]
						self.processing[timeStamp][aStep][aProcess][aSetting] = aDetail	
				
			if path:
				cwd = os.getcwd()
				timeStamp = dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S:%m')
				self.processing[timeStamp] = {}
				self.processing[timeStamp][1] = {}
				self.processing[timeStamp][1] = {'Read':(cwd+'/'+path)}				
						
		#~~~~~~~~~~~~~~ Other ~~~~~~~~~~~~~~#
		else:
			print 'Unidentified object type: {}'.format(type(a))
			print '   See Trace __init__'
	def listHeader(self):
		print '  Station: %s-%s-%s.%s' % (self.network,self.station,self.staLocation,self.channel)
		print '  Sta Lat: %0.4f' % self.staLat
		print '  Sta Lon: %0.4f' % self.staLon
		print ' Sta Elev: %0.2f' % self.staElev
		print 'Sta Depth: %0.2f' % self.staDepth
		print '   Calibr: %0.2f' % self.calibration
		print '  Comp Az: %0.4f (N)' % self.componentAzNorth
		print ' Comp Inc: %0.4f (Vertical)' % self.componentIncidentAngleVertical
		
		print '\n   Origin: %s' % self.origin
		print '   Ev Lat: %0.4f' % self.evLat
		print '   Ev Lon: %0.4f' % self.evLon
		print ' Ev Depth: %0.2f' % self.evDepth
		print '      Mag: %0.2f' % self.mag

		print '\n     Dist: %0.4f' % self.distance
		print '    GCARC: %0.4f' % self.gcarc
		print '       AZ: %0.4f' % self.az
		print '      BAZ: %0.4f' % self.baz
		
		print '\n Quality: %d' % self.quality
		print '   Units: %s' % self.idep
		print '    Npts: %d' % self.npts
		print 'SampRate: %0.2e' % self.sampleRate
		print ' Nyquist: %0.2e' % self.nyquist
		print '   Delta: %0.2e' % self.delta
		print '   Start: %s' % self.startTime
		print '     End: %s' % self.endTime
		print '     Ref: %s' % self.refTime
		print '       b: %0.2f' % self.b
		print '       e: %0.2f' % self.e
		print ' min Amp: %0.4e' % self.minAmp 
		print ' max Amp: %0.4e' % self.maxAmp 
		print 'mean Amp: %0.4e' % self.meanAmp
		if int(self.fileType) == 9:
			print 'FileType: Reference time beginning of trace'
		elif int(self.fileType) == 11:
			print 'FileType: Reference origin'		
		else:
			print 'FileType: %s' % self.fileType

		print '\no Marker: %0.4e (%s)' % (self.marker_o,self.marker_ko)		
		print 'a Marker: %0.4e (%s)' % (self.marker_a,self.marker_ka)		
		print 't0Marker: %0.4e (%s)' % (self.marker_t0,self.marker_kt0)		
		print 't1Marker: %0.4e (%s)' % (self.marker_t1,self.marker_kt1)		
		print 't2Marker: %0.4e (%s)' % (self.marker_t2,self.marker_kt2)		

		print '\n  LOVROK: %s' % self.lovrok
		print '  LPSPOL: %s' % self.lpspol
		print '  LCALDA: %s' % self.lcalda
		print '   NVHDR: %d' % self.nvhdr
		
		print '\n Signal Processing'
		for aDate in sorted(self.processing, key=self.processing.get, reverse=False):
			print aDate, self.processing[aDate]
	def listFields(self):
		fields = dir(self)
		for f in fields:
			if f[0] != '_':
				print f
	def copy(self):
		'''Copy Trace object'''
		
		a = deepcopy(self)
		
		a.recordProcessing('Copy','')
		return a	

	def recordProcessing(self,process,details):
		timeStamp = dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S:%m')
		if timeStamp in self.processing:
			ii = self.processing[timeStamp].keys()[-1] + 1
			self.processing[timeStamp][ii] = {}
		else:
			ii = 1
			self.processing[timeStamp] = {}
			self.processing[timeStamp][ii] = {}
		self.processing[timeStamp][ii][process] = {}
		self.processing[timeStamp][ii][process] = details
		
	def cosTaper(self,p=0.1,freqs=None,flimit=None,halfcosine=True,sactaper=False):
		'''
		See obspy.signal.invsim.cosTaper
		
		:type p: Float
		:param p: Decimal percentage of cosine taper (ranging from 0 to 1). Default
			is 0.1 (10%) which tapers 5% from the beginning and 5% form the end.
		:type freqs: NumPy ndarray
		:param freqs: Frequencies as, for example, returned by fftfreq
		:type flimit: List or tuple of floats
		:param flimit: The list or tuple defines the four corner frequencies
			(f1, f2, f3, f4) of the cosine taper which is one between f2 and f3 and
			tapers to zero for f1 < f < f2 and f3 < f < f4.
		:type halfcosine: Boolean
		:param halfcosine: If True the taper is a half cosine function. If False it
			is a quarter cosine function.
		:type sactaper: Boolean
		:param sactaper: If set to True the cosine taper already tapers at the
			corner frequency (SAC behaviour). By default, the taper has a value
			of 1.0 at the corner frequencies.
		'''
		
		self.data = self.data * cosTaper(self.data.size,p=p,freqs=freqs,flimit=flimit,halfcosine=halfcosine,sactaper=sactaper)
		
		details = {'p':p,'freqs':freqs,'flimit':flimit,'halfcosine':halfcosine,
					'sactaper':sactaper}
		self.recordProcessing('Cosine taper',details)
	def detrend(self,axis=-1,type='linear',bp=0):
		'''
		See scipy.signal.detrend
		
		:param data: array_like
						The input data.
		:param axis: int, optional
						The axis along which to detrend the data. By default this is the 
						last axis (-1).
		:param type: {‘linear’, ‘constant’}, optional
						The type of detrending. If type == 'linear' (default), the result 
						of a linear least-squares fit to data is subtracted from data. If 
						type == 'constant', only the mean of data is subtracted.
		:param bp: array_like of ints, optional
						A sequence of break points. If given, an individual linear fit is 
						performed for each part of data between two break points. Break 
						points are specified as indices into data.
		'''
		
		self.data = signal.detrend(self.data,axis=axis,type=type,bp=bp)
		
		self.recordProcessing('Detrend',{'axis':axis,'type':type,'bp':bp})
	def smooth(self, window_size=1):
		'''
		Convolves a boxcar of desired length to smooth the waveform (effectively averaging
			as a function of the window_size. Original timeseries length is preserved.
		
		:param window: Length of unit area boxcar to convolve with timeseries
		'''
		
		window = np.ones(int(window_size)) / float(window_size)
		self.data = np.convolve(self.data, window, 'same')
		
		details = {'window size':window_size}
		self.recordProcessing('Smooth',details)
	
	##~~ All cutting routines below are all copied from Obspy. I take no credit.
	##~~ 	(http://docs.obspy.org/_modules/obspy/core/trace.html)				
	def _ltrim(self, starttime, pad=False, nearest_sample=True,fill_value=None):
		"""
		Cuts current trace to given start time. For more info see
		:meth:`~obspy.core.trace.Trace.trim`.

		.. rubric:: Example

		>>> tr = Trace(data=np.arange(0, 10))
		>>> tr.stats.delta = 1.0
		>>> tr._ltrim(tr.stats.starttime + 8)  # doctest: +ELLIPSIS
		<...Trace object at 0x...>
		>>> tr.data
		array([8, 9])
		>>> tr.stats.starttime
		UTCDateTime(1970, 1, 1, 0, 0, 8)
		"""
		org_dtype = self.data.dtype
		if isinstance(starttime, float) or isinstance(starttime, int):
			starttime = UTCDateTime(self.startTime) + starttime
		elif not isinstance(starttime, UTCDateTime):
			raise TypeError
		# check if in boundary
		if nearest_sample:
			delta = round((starttime - self.startTime) * self.sampleRate)
			# due to rounding and npts starttime must always be right of
			# self.stats.starttime, rtrim relies on it
			if delta < 0 and pad:
				npts = abs(delta) + 10  # use this as a start
				newstarttime = self.startTime - npts / float(self.sampleRate)
				newdelta = round((starttime - newstarttime) * self.sampleRate)
				delta = newdelta - npts
			delta = int(delta)
		else:
			delta = int(math.floor(round((self.startTime - starttime) * self.sampleRate, 7))) * -1
		# Adjust starttime only if delta is greater than zero or if the values
		# are padded with masked arrays.
		if delta > 0 or pad:
			self.startTime += delta * self.delta
		if delta == 0 or (delta < 0 and not pad):
			return
		elif delta < 0 and pad:
			try:
				gap = createEmptyDataChunk(abs(delta), self.data.dtype,
										   fill_value)
			except ValueError:
				# createEmptyDataChunk returns negative ValueError ?? for
				# too large number of points, e.g. 189336539799
				raise Exception("Time offset between starttime and trace.starttime too large")
			self.data = np.ma.concatenate((gap, self.data))
			return
		elif starttime > self.endTime:
			self.data = np.empty(0, dtype=org_dtype)
			return
		elif delta > 0:
			self.data = self.data[delta:]
			
		# Update header information
		self.npts = len(self.data)
	def _rtrim(self,endtime,pad=False,nearest_sample=True,fill_value=None):
		"""
		Cuts current trace to given end time. For more info see
		:meth:`~obspy.core.trace.Trace.trim`.

		.. rubric:: Example

		>>> tr = Trace(data=np.arange(0, 10))
		>>> tr.stats.delta = 1.0
		>>> tr._rtrim(tr.stats.starttime + 2)  # doctest: +ELLIPSIS
		<...Trace object at 0x...>
		>>> tr.data
		array([0, 1, 2])
		>>> tr.stats.endtime
		UTCDateTime(1970, 1, 1, 0, 0, 2)
		"""
		org_dtype = self.data.dtype
		if isinstance(endtime, float) or isinstance(endtime, int):
			endtime = UTCDateTime(self.endTime) - endtime
		elif not isinstance(endtime, UTCDateTime):
			raise TypeError
		# check if in boundary
		if nearest_sample:
			delta = round((endtime - self.startTime) *
						self.sampleRate) - self.npts + 1
			delta = int(delta)
		else:
			# solution for #127, however some tests need to be changed
			#delta = -1*int(math.floor(round((self.stats.endtime - endtime) * \
			#                       self.stats.sampling_rate, 7)))
			delta = int(math.floor(round((endtime - self.endTime) *
								   self.sampleRate, 7)))
		
		if delta == 0 or (delta > 0 and not pad):
			return
		if delta > 0 and pad:
			try:
				gap = createEmptyDataChunk(delta, self.data.dtype, fill_value)
			except ValueError:
				# createEmptyDataChunk returns negative ValueError ?? for
				# too large number of pointes, e.g. 189336539799
				raise Exception("Time offset between starttime and " +
									"trace.starttime too large")
			self.data = np.ma.concatenate((self.data, gap))
			return
		elif endtime < self.startTime:
# 			self.startTime = self.endTime + delta * self.delta
# 			self.data = np.empty(0, dtype=org_dtype)
# 			return
			print ("Error (pySACio.Trace._rtrim): {}: End ".format(self.stationID) +
							"time earlier than trace start time. No end cut performed")
			return
		# cut from right
		delta = abs(delta)
		total = len(self.data) - delta
				
		if endtime == self.startTime:
			total = 1
		self.data = self.data[:total]

		# Update header information
		self.endTime = self.startTime + total * self.delta
		self.npts = len(self.data)
	def trim(self,starttime=None,endtime=None,pad=False,nearest_sample=True,fill_value=None):
		"""
		Cuts current trace to given start and end time.

		:type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
		:param starttime: Specify the start time.
		:type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
		:param endtime: Specify the end time.
		:type pad: bool, optional
		:param pad: Gives the possibility to trim at time points outside the
			time frame of the original trace, filling the trace with the
			given ``fill_value``. Defaults to ``False``.
		:type nearest_sample: bool, optional
		:param nearest_sample: If set to ``True``, the closest sample is
			selected, if set to ``False``, the next sample containing the time
			is selected. Defaults to ``True``.

				Given the following trace containing 4 samples, "|" are the
				sample points, "A" is the requested starttime::

					|        A|         |         |

				``nearest_sample=True`` will select the second sample point,
				``nearest_sample=False`` will select the first sample point.

		:type fill_value: int, float or ``None``, optional
		:param fill_value: Fill value for gaps. Defaults to ``None``. Traces
			will be converted to NumPy masked arrays if no value is given and
			gaps are present.

		.. note::

			This operation is performed in place on the actual data arrays. The
			raw data is not accessible anymore afterwards. To keep your
			original data, use :meth:`~obspy.core.trace.Trace.copy` to create
			a copy of your trace object.

		.. rubric:: Example

		>>> tr = Trace(data=np.arange(0, 10))
		>>> tr.stats.delta = 1.0
		>>> t = tr.stats.starttime
		>>> tr.trim(t + 2.000001, t + 7.999999)  # doctest: +ELLIPSIS
		<...Trace object at 0x...>
		>>> tr.data
		array([2, 3, 4, 5, 6, 7, 8])
		"""	
		# check time order and swap eventually
		if starttime and endtime and starttime > endtime:
			raise ValueError("startime is larger than endtime")
		# cut it
		if starttime:
			self._ltrim(starttime,pad=pad,nearest_sample=nearest_sample,fill_value=fill_value)
		if endtime:
			self._rtrim(endtime,pad=pad,nearest_sample=nearest_sample,fill_value=fill_value)
		# if pad=True and fill_value is given convert to NumPy ndarray
		if pad is True and fill_value is not None:
			try:
				self.data = self.data.filled()
			except AttributeError:
				# numpy.ndarray object has no attribute 'filled' - ignoring
				pass
	def cut(self,starttime=None,endtime=None,cutFrom='abs'):
		"""
		Returns a new Trace object with data going from start to end time.

		:type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
		:param starttime: Specify the start time of slice.
		:type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
		:param endtime: Specify the end time of slice.
		:return: New :class:`~obspy.core.trace.Trace` object. Does not copy
			data but just passes a reference to it.

		.. rubric:: Example

		>>> startCut = test.origin + test.marker_a - 5
		>>> endCut   = startCut + 50	
		>>> test.cut(startCut,endCut)
		
		:type cutFrom: string
		:param cutFrom:	'abs' = provide ~obspy.core.utcdatetime.UTCDateTime of absolute
									UTC time cuts
						'o' = provide int/float of seconds from origin
						'a' = provide int/float of seconds from a-marker
						't0' = provide int/float of seconds from t0-marker
						't1' = provide int/float of seconds from t1-marker
						't2' = provide int/float of seconds from t2-marker
		"""
		if cutFrom == 'abs':
			if isinstance(starttime,UTCDateTime) and isinstance(endtime,UTCDateTime):
				self.trim(starttime=starttime, endtime=endtime)
			else:
				print 'Trace.cut(): starttime/endtime are not UTCDateTime for absolute cut'
				print '   Cut not performed'
				sys.exit()
		
		elif cutFrom == 'o':
			if isinstance(starttime,(int,float)) and isinstance(endtime,(int,float)):
				starttime = self.origin + starttime
				endtime   = self.origin + endtime
				
				if starttime < self.startTime:
					starttime = self.startTime
					print 'Start cut prior to start of trace. Start cut moved to start of trace.'

				if endtime > self.endTime:
					starttime = self.startTime
					print 'End cut after end of trace. End cut moved to end of trace.'
				
				self.trim(starttime=starttime, endtime=endtime)
			else:
				print 'Trace.cut(): starttime/endtime are not float/int for origin cut'
				print '   Cut not performed'
				sys.exit()

		elif cutFrom == 'a':
			if isinstance(starttime,(int,float)) and isinstance(endtime,(int,float)):
				starttime = self.origin + self.marker_a + starttime
				endtime   = self.origin + self.marker_a + endtime
				self.trim(starttime=starttime, endtime=endtime)
			else:
				print 'Trace.cut(): starttime/endtime are not float/int for a-marker cut'
				print '   Cut not performed'
				sys.exit()

		elif cutFrom == 't0':
			if isinstance(starttime,(int,float)) and isinstance(endtime,(int,float)):
				starttime = self.origin + self.marker_t0 + starttime
				endtime   = self.origin + self.marker_t0 + endtime
				self.trim(starttime=starttime, endtime=endtime)
			else:
				print 'Trace.cut(): starttime/endtime are not float/int for t0-marker cut'
				print '   Cut not performed'
				sys.exit()

		elif cutFrom == 't1':
			if isinstance(starttime,(int,float)) and isinstance(endtime,(int,float)):
				starttime = self.origin + self.marker_t1 + starttime
				endtime   = self.origin + self.marker_t1 + endtime
				self.trim(starttime=starttime, endtime=endtime)
			else:
				print 'Trace.cut(): starttime/endtime are not float/int for t1-marker cut'
				print '   Cut not performed'
				sys.exit()

		elif cutFrom == 't2':
			if isinstance(starttime,(int,float)) and isinstance(endtime,(int,float)):
				starttime = self.origin + self.marker_t2 + starttime
				endtime   = self.origin + self.marker_t2 + endtime
				self.trim(starttime=starttime, endtime=endtime)
			else:
				print 'Trace.cut(): starttime/endtime are not float/int for t2-marker cut'
				print '   Cut not performed'
				sys.exit()
		
		else:
			print 'Unknown cut parameter for "cutFrom" in Trace.cut()'
			print '   Cut no performed'
			sys.exit()

		details = {'cutFrom':cutFrom,'start':starttime.strftime('%Y-%m-%d %H:%M:%S:%f'),
						'end':endtime.strftime('%Y-%m-%d %H:%M:%S:%f')}
		self.recordProcessing('Cut',details)
				
	def envelope(self):
		"""
		see obspy.signal.filter.envelope
		
		Envelope of a function.

		Computes the envelope of the given function. The envelope is determined by
		adding the squared amplitudes of the function and it's Hilbert-Transform
		and then taking the square-root. (See [Kanasewich1981]_)
		The envelope at the start/end should not be taken too seriously.
		"""
		
		hilb = hilbert(self.data)
		self.data = (self.data ** 2 + hilb ** 2) ** 0.5
		
		self.recordProcessing('Envelope','')
	def decimate(self, decimation_factor, filter=True):
		"""
		See obspy.signal.filter.integerDecimation 
				and obspy.core.trace.Trace.decimate (modified)
		
		Downsampling by applying a simple integer decimation.

		:param decimation_factor: Integer decimation factor. Range is 2 to 7. This 
									command may be applied several times if a larger 
									decimation factor is required.
		:param filter: If true, an anti-aliasing lowpass filter is applied  
		"""

		if not isinstance(decimation_factor, int):
			msg = "Decimation_factor must be an integer!"
			raise TypeError(msg)
		
		# Low-pass
		if filter:
			if decimation_factor > 7:
				msg = "Too large of a decimation factor. Use values between 2 and 7."
				msg = msg + "\nThis process can be applied multiple times if larger decimation"
				msg = mgs + "\nfactor is required."
				raise TypeError(msg)
			
			else:
				antiAliasFreq = self.sampleRate * 0.5 / float(decimation_factor)
				self.lowpassCheby2(freq=antiAliasFreq, maxorder=12)

		# Decimate the data
		self.data = np.array(self.data[::decimation_factor])
		
		# Update header information
		self.npts = len(self.data)
		self.sampleRate = self.sampleRate / decimation_factor
		self.delta = 1.0 / self.sampleRate
				
		# Update Processing
		details = {'decimation factor':decimation_factor,'filter':filter}
		self.recordProcessing('Decimate',details)

	def bandpass(self, freqmin, freqmax, corners=4, zerophase=True):
		"""
		from obspy.signal.filter.bandpass
		
		Butterworth-Bandpass Filter.

		Filter data from ``freqmin`` to ``freqmax`` using ``corners`` corners.

		:param freqmin: Pass band low corner frequency.
		:param freqmax: Pass band high corner frequency.
		:param corners: Filter corners. Note: This is twice the value of PITSA's
			filter sections
		:param zerophase: If True, apply filter once forwards and once backwards.
			This results in twice the number of corners but zero phase shift in
			the resulting filtered trace.
		"""
		
		fe = 0.5 * self.sampleRate
		low = freqmin / fe
		high = freqmax / fe
		# raise for some bad scenarios
		if high > 1:
			high = 1.0
			msg = "Selected high corner frequency is above Nyquist. " + \
				  "Setting Nyquist as high corner."
			warnings.warn(msg)
		if low > 1:
			msg = "Selected low corner frequency is above Nyquist."
			raise ValueError(msg)
		[b, a] = signal.iirfilter(corners, [low, high], btype='band',
						   ftype='butter', output='ba')
		if zerophase:
			firstpass = signal.lfilter(b, a, self.data)
			self.data = signal.lfilter(b, a, firstpass[::-1])[::-1]
		else:
			self.data = signal.lfilter(b, a, self.data)
		
		details = {'freqmin':freqmin,'freqmax':freqmax,'corners':corners,'zerophase':zerophase}
		self.recordProcessing('Band-Pass',details)
	def lowpass(self, freq, corners=4, zerophase=True):
		"""
		see obspy.signal.filter.lowpass
		
		Butterworth-Lowpass Filter.

		Filter data removing data over certain frequency ``freq`` using ``corners``
		corners.

		:param freq: Filter corner frequency.
		:param corners: Filter corners. Note: This is twice the value of PITSA's
			filter sections
		:param zerophase: If True, apply filter once forwards and once backwards.
			This results in twice the number of corners but zero phase shift in
			the resulting filtered trace.
		"""
		
		fe = 0.5 * self.sampleRate
		f = freq / fe
		# raise for some bad scenarios
		if f > 1:
			f = 1.0
			msg = "Selected corner frequency is above Nyquist. " + \
				  "Setting Nyquist as high corner."
			warnings.warn(msg)
		[b, a] = signal.iirfilter(corners, f, btype='lowpass', ftype='butter',
						   output='ba')
		if zerophase:
			firstpass = signal.lfilter(b, a, self.data)
			self.data = signal.lfilter(b, a, firstpass[::-1])[::-1]
		else:
			self.data = signal.lfilter(b, a, self.data)

		details = {'freq':freq,'corners':corners,'zerophase':zerophase}
		self.recordProcessing('Low-Pass',details)
	def highpass(self, freq, corners=4, zerophase=True):
		"""
		see obspy.signal.filter.highpass
		
		Butterworth-Highpass Filter.

		Filter data removing data below certain frequency ``freq`` using
		``corners`` corners.

		:param freq: Filter corner frequency.
		:param corners: Filter corners. Note: This is twice the value of PITSA's
			filter sections
		:param zerophase: If True, apply filter once forwards and once backwards.
			This results in twice the number of corners but zero phase shift in
			the resulting filtered trace.
		"""
		
		fe = 0.5 * self.sampleRate
		f = freq / fe
		# raise for some bad scenarios
		if f > 1:
			msg = "Selected corner frequency is above Nyquist."
			raise ValueError(msg)
		[b, a] = signal.iirfilter(corners, f, btype='highpass', ftype='butter',
						   output='ba')
		if zerophase:
			firstpass = signal.lfilter(b, a, self.data)
			self.data = signal.lfilter(b, a, firstpass[::-1])[::-1]
		else:
			self.data = signal.lfilter(b, a, data)

		details = {'freq':freq,'corners':corners,'zerophase':zerophase}
		self.recordProcessing('High-Pass',details)
	def lowpassCheby2(self, freq, maxorder=12, ba=False, freq_passband=False):
		"""
		See obspy.signal.filter.lowpassCheby2
		
		Cheby2-Lowpass Filter

		Filter data by passing data only below a certain frequency.
		The main purpose of this cheby2 filter is downsampling.

		This method will iteratively design a filter, whose pass
		band frequency is determined dynamically, such that the
		values above the stop band frequency are lower than -96dB.

		:param freq: The frequency above which signals are attenuated
			with 95 dB
		:param maxorder: Maximal order of the designed cheby2 filter
		:param ba: If True return only the filter coefficients (b, a) instead
			of filtering
		:param freq_passband: If True return additionally to the filtered data,
			the iteratively determined pass band frequency
		"""
		
		nyquist = 0.5 * self.sampleRate
		
		# rp - maximum ripple of passband, rs - attenuation of stopband
		rp, rs, order = 1, 96, 1e99
		ws = freq / nyquist  # stop band frequency
		wp = ws              # pass band frequency
		
		# raise for some bad scenarios
		if ws > 1:
			ws = 1.0
			msg = "Selected corner frequency is above Nyquist. " + \
				  "Setting Nyquist as high corner."
			warnings.warn(msg)
		
		while True:
			if order <= maxorder:
				break
			wp = wp * 0.99
			order, wn = signal.cheb2ord(wp, ws, rp, rs, analog=0)
		
		b, a = signal.cheby2(order, rs, wn, btype='low', analog=0, output='ba')
		
		if ba:
			return b, a
		
		else:
			self.data = signal.lfilter(b, a, self.data)

			details = {'freq':freq,'maxorder':maxorder}
			self.recordProcessing('Low-Pass Cheby2',details)
			
		if freq_passband:
			return wp * nyquist	

	def plot(self,xLabel='clock',limits=False,showXLabel=True,showTitle=True,title=None,
				picks=True,grid=True,singlePlot=True,plot='y',saveFile=None):
		'''
		xlabel: 'clock' (UTC time), 'sec' (seconds from origin)
		showXlabel: False to suppress showing labels
		picks: True shows phase picks in header markers
		
		'''
		##~ Set data to be plotted ~##
		values=self.data

		start = md.date2num(self.startTime) * 60 * 60 * 24
		end = md.date2num(self.endTime) * 60 * 60 * 24
		datenums = np.linspace(start, end, len(self.data)) / float(60 * 60 * 24)
		
		##~ Set figure dimensions ~##
		if singlePlot:
			fig = plt.figure(figsize=(10,4), dpi=100)

		##~ Set figure labels and axis ~##
		ax=plt.gca()
		ax.ticklabel_format(style='sci',scilimits=(-3,4),axis='y')

		if xLabel == 'clock':
			plt.subplots_adjust(bottom=0.2)
			plt.xticks( rotation=45 )
# 			xfmt = md.DateFormatter('%H:%M:%S\n%Y-%m-%d')
			xfmt = md.DateFormatter('%H:%M:%S')
			ax.xaxis.set_major_formatter(xfmt)

			st = min(datenums)
			en = max(datenums)
			
			plt.xticks(np.arange(st,en,(en-st)/15))
	
			plt.plot(datenums,values,'black')
		
		elif xLabel == 'sec':
			datenums = (datenums - md.date2num(self.origin)) * 60 * 60 * 24
			plt.plot(datenums,values,'black')
			plt.xlabel('Time Since Origin (s)', fontsize=14)	
		
		else:
			print 'Unknown x-label scale. See Trace.plot()'
			sys.exit()
			
		if not showXLabel:
			ax.get_xaxis().set_visible(False)
	
		if self.idep != -12345:
			yLabel = self.idep
			plt.ylabel(yLabel, fontname='sans-serif', fontsize=14)
		if showTitle:
			if not title:
				aTitle = '%s - GC:%0.2f Az:%0.1f - T: %s' % \
					(self.stationID,self.gcarc,self.az,self.origin.strftime('%Y (%j) %H:%M:%S'))
			else:
				aTitle = title
			plt.title(aTitle, fontname='sans-serif', fontsize=12)
		
		##~ Set figure plot limits ~##
		if limits:
			if limits[1] > 0:
				if xLabel == 'clock':
					endDate = datenums[0] + (limits[1]/(60.0*60*24))
					plt.xlim(datenums[0],endDate)
				elif xLabel == 'sec':
					endDate = datenums[0] + limits[1]
					plt.xlim(datenums[0],endDate)
				else:
					plt.xlim(datenums[0],datenums[-1])
				
			else:
				plt.xlim(datenums[0],datenums[-1])
				
			if (limits[3]-limits[2]) > 0:
				plt.ylim(limits[2],limits[3])
		else:	
			plt.xlim(datenums[0],datenums[-1])
			plt.axis('tight')
		
		##~ Grid ~##
		if grid:
			if isinstance(grid,float):
				plt.grid(True, which="both",linestyle=':',linewidth=0.25, color=str(grid))
			else:
				plt.grid(True, which="both",linestyle=':',linewidth=0.25, color='0.65')
		
		##~ Phase picks ~##
		if picks:
			pltLimits = plt.axis()
			ymin = 0.1 * (pltLimits[3] - pltLimits[2]) + pltLimits[2]
			ymax = 0.9 * (pltLimits[3] - pltLimits[2]) + pltLimits[2]
		
			ylabel = 0.06 * (pltLimits[3] - pltLimits[2]) + pltLimits[2]
		
			color = '0.65'
			
			if xLabel == 'clock':
				origin = md.date2num(self.origin) * 60 * 60 * 24
				if self.marker_o != -12345:
					x = (origin + self.marker_o) / float(60 * 60 * 24)
					label = self.marker_ko
					plt.vlines(x, ymin, ymax, colors=color, linestyles='solid')
					plt.text(x,ylabel,label)

				if self.marker_a != -12345:
					x = (origin + self.marker_a) / float(60 * 60 * 24)
					label = self.marker_ka
					plt.vlines(x, ymin, ymax, colors=color, linestyles='solid')
					plt.text(x,ylabel,label)

				if self.marker_t0 != -12345:
					x = (origin + self.marker_t0) / float(60 * 60 * 24)
					label = self.marker_kt0
					plt.vlines(x, ymin, ymax, colors=color, linestyles='solid')
					plt.text(x,ylabel,label)

				if self.marker_t1 != -12345:
					x = (origin + self.marker_t1) / float(60 * 60 * 24)
					label = self.marker_kt1
					plt.vlines(x, ymin, ymax, colors=color, linestyles='solid')
					plt.text(x,ylabel,label)
		
				if self.marker_t2 != -12345:
					x = (origin + self.marker_t2) / float(60 * 60 * 24)
					label = self.marker_kt2
					plt.vlines(x, ymin, ymax, colors=color, linestyles='solid')
					plt.text(x,ylabel,label)

			elif xLabel == 'sec':
				if self.marker_o != -12345:
					x = self.marker_o
					label = self.marker_ko
					plt.vlines(x, ymin, ymax, colors=color, linestyles='solid')
					plt.text(x,ylabel,label)

				if self.marker_a != -12345:
					x = self.marker_a
					label = self.marker_ka
					plt.vlines(x, ymin, ymax, colors=color, linestyles='solid')
					plt.text(x,ylabel,label)

				if self.marker_t0 != -12345:
					x = self.marker_t0
					label = self.marker_kt0
					plt.vlines(x, ymin, ymax, colors=color, linestyles='solid')
					plt.text(x,ylabel,label)

				if self.marker_t1 != -12345:
					x = self.marker_t1
					label = self.marker_kt1
					plt.vlines(x, ymin, ymax, colors=color, linestyles='solid')
					plt.text(x,ylabel,label)
		
				if self.marker_t2 != -12345:
					x = self.marker_t2
					label = self.marker_kt2
					plt.vlines(x, ymin, ymax, colors=color, linestyles='solid')
					plt.text(x,ylabel,label)

		##~ Show or save plot ~##
		if (plot[0].lower() == 'y') or saveFile:
			if not saveFile:
				plt.show()
				plt.close()
	
			else:
				F = plt.gcf()
				DefaultSize = F.get_size_inches()
				F.set_size_inches( (DefaultSize[0]*2, DefaultSize[1]) )

				plt.savefig((saveFile+'.pdf'))
				plt.close('all')		

##~~ IO ~~##
def readSAC(path):
	st = read(path, format='SAC')
	waveform = st[0]
	newTrace = Trace(waveform,path=path)
	
	return newTrace
def readDirectory(path,sort='complete',splitByDate=False):
	'''
	path = 'Waveforms/E*/Dsp_LP/*.lh?'
	'''	
	eventList = []
	waveforms = glob.glob(path)
	
	for aTrace in waveforms:
		newTrace = readSAC(aTrace)
		eventList.append(newTrace)
		
	if splitByDate:
		eventList = splitDirectoryByDate(eventList)
			
	else:
		eventList = sortDirectory(eventList,sortDate=True,sort=sort,reverse=False)
							
	return eventList

def save2HDF5(traceObject,directory=''):
	'''
	traceObject = The trace that will be saved. This can either be a single waveform
					(type = 'instance') or a set of waveforms (type = 'list')
	directory = Location where the file will be saved. By default, the file will be saved
					in the current working directory.
	'''
	hdf5File = {}
	
	#~~ List of traces ~~#
	if type(traceObject) is list:
		if len(traceObject) > 0:
			if isinstance(traceObject[0],Trace):
				eventList = []
				for aTrace in traceObject:
					eventName = aTrace.origin.strftime('E%Y-%m-%d-%H-%M-%S')
			
					if eventName not in eventList:
						eventList.append(eventName)
				
						fname = directory + eventName + '.h5'
	
						if os.path.isfile(fname):
							os.remove(fname)
							hdf5File[eventName] = h5py.File(fname, 'w')
						else:
							hdf5File[eventName] = h5py.File(fname, 'w')
						
					save_a_HDF5(aTrace,hdf5File[eventName])
			
			else:
				print 'Unidentified trace object type.'
				print '   See save2HDF5'
				sys.exit()
		else:
			print 'Trace list is empty.'
			print '   See save2HDF5'
			sys.exit()
			
		
	#~~ Single trace ~~#
	elif isinstance(traceObject,Trace):
		eventName = raw[0].origin.strftime('E%Y-%m-%d-%H-%M')
		fname = directory + eventName + '.h5'
	
		if os.path.isfile(fname):
			os.remove(fname)
			hdf5File[eventName] = h5py.File(fname, 'w')
		else:
			hdf5File[eventName] = h5py.File(fname, 'w')
			
		save_a_HDF5(traceObject,hdf5File[eventName])
	
	#~~ Unidentified format ~~#
	else:
		print 'Unidentified trace object type.'
		print '   See save2HDF5'
		sys.exit()
	
	#~~ Close HDF5 objects ~~#
	for aFile in hdf5File:
		hdf5File[aFile].close()
def save_a_HDF5(trace,fname):
	try:
		g = fname.require_group(trace.network)
		h = g.require_group(trace.station)
	except TypeError:
		pass
	h.create_dataset(trace.stationID, data=trace.data, compression=None)

	populateHDF5Attributes(trace,h[trace.stationID])
def populateHDF5Attributes(traceObject,h5pyObject):
	processingAttrs = ''
	for attr, value in traceObject.__dict__.iteritems():				
		if attr != 'data':
			if str(type(value)) == "<class 'obspy.core.utcdatetime.UTCDateTime'>":
				h5pyObject.attrs[attr] = str(value)
			elif attr == 'processing':
				for aTime in value:
					for aStep in value[aTime]:
						for aProcess in value[aTime][aStep]:
							params = value[aTime][aStep][aProcess]
							if isinstance(params,str):
								aSet = '{},{},{},{}'.format(
											aTime,str(aStep),aProcess,params)
								processingAttrs = processingAttrs + '|' + aSet
							else:
								for aParam in params.keys():
									aValue = params[aParam]
									aSet = '{},{},{},{},{}'.format(
											aTime,str(aStep),aProcess,aParam,str(aValue))
									processingAttrs = processingAttrs + '|' + aSet
			else:	
				h5pyObject.attrs[attr] = value
				
			h5pyObject.attrs['processing'] = processingAttrs

def readOneHDF5Event(h5pyEvent,path=None):
	'''
	This requires the HDF5 to be open in read and you know the location of the waveform
		that you want.
		
		example:	h = h5py.File('E2013-02-12-02-57.h5','r')
					readOneHDF5Event(h['IU']['YSS']['IU-YSS-00.LHR'])
	'''
	newTrace = Trace(h5pyEvent,path=path)
	
	return newTrace
def readHDF5(h5pyFile,path=False):
	'''
	eventList = readHDF5(h5pyFile,path=False)
	
	path = True/False
		if True: include a path address to the processing record.
	'''
	hdf5File = h5py.File(h5pyFile, 'r')
	
	if path == True:
		path = h5pyFile
	
	eventList = []
	for aNetwork in hdf5File.keys():
		for aStation in hdf5File[aNetwork].keys():
			for aChannel in hdf5File[aNetwork][aStation]:
				waveform = hdf5File[aNetwork][aStation][aChannel]
				newTrace = readOneHDF5Event(waveform,path=path)
				eventList.append(newTrace)
	
	return eventList
				
def save2sac(trace,directory=''):
	#
# 	fname = directory + 'E' + trace.origin.strftime('%Y-%m-%d-%H-%M') + '.sac'		
	fname = directory + trace.station + '-' + trace.network + '-' + trace.staLocation + \
		'-' + trace.channel + '-' + trace.origin.strftime('%Y-%m-%d-%H-%M') + '.sac'		
	#
	h = SacIO()
# 	h.readTrace(tr)
	h.fromarray(trace.data, begin=trace.b, delta=trace.delta, starttime=trace.startTime)
	h.SetHvalue('lcalda',True)
	#
	h.SetHvalue('stlo',trace.staLon)
	h.SetHvalue('stla',trace.staLat)
	h.SetHvalue('stel',trace.staElev)
	h.SetHvalue('stdp',trace.staDepth)
	h.SetHvalue('cmpaz',trace.componentAzNorth)
	h.SetHvalue('cmpinc',trace.componentIncidentAngleVertical)
	h.SetHvalue('kstnm',trace.station)
	h.SetHvalue('knetwk',trace.network)
	locChan = trace.staLocation + trace.channel
	h.SetHvalue('kcmpnm',locChan)
	#
	h.SetHvalue('evlo',trace.evLon)
	h.SetHvalue('evla',trace.evLat)
	h.SetHvalue('evdp',trace.evDepth)
	h.SetHvalue('mag',trace.mag)
	#
	h.SetHvalue('o',trace.marker_o)
	#
	h.SetHvalue('a',trace.marker_a)
	h.SetHvalue('ka',trace.marker_ka)
	h.SetHvalue('T0',trace.marker_t0)
	h.SetHvalue('kt0',trace.marker_kt0)
	h.SetHvalue('T1',trace.marker_t1)
	h.SetHvalue('kt1',trace.marker_kt1)
	h.SetHvalue('T2',trace.marker_t2)
	h.SetHvalue('kt2',trace.marker_kt2)
	#
	h.SetHvalue('nzyear', trace.origin.year)
	h.SetHvalue('nzjday', trace.origin.julday)
	h.SetHvalue('nzhour', trace.origin.hour)
	h.SetHvalue('nzmin', trace.origin.minute)
	h.SetHvalue('nzsec', trace.origin.second)
	h.SetHvalue('nzmsec', trace.origin.microsecond)
	h.SetHvalue('b',trace.b)
	h.SetHvalue('e',trace.e)
	h.SetHvalue('iztype',trace.fileType)
	#
	h.SetHvalue('iqual',trace.quality)	
	#
	h.WriteSacBinary(fname)

##~~ Dataset Handling ~~##
def sortDirectory(waveformSet,sortDate=True,sort='complete',reverse=False):
	'''
	sort:	'complete'	sort by network,station,channel
			'origin'
			'network'
			'station'
			'channel'
			'gcarc'		sort by great circle arc
			'az'		sort by azimuth
			'baz'		sort by back azimuth
			'quality'	sort by quality
	'''
	try:
		if sort == 'complete':
			sortedSet = sorted(waveformSet, key=lambda trace: trace.stationID, reverse=reverse)
			if sortDate:
				sortedSet = sorted(sortedSet, key=lambda trace: trace.origin, reverse=reverse)
		if sort == 'origin':
			sortedSet = sorted(waveformSet, key=lambda trace: trace.origin, reverse=reverse)
		if sort == 'network':
			sortedSet = sorted(waveformSet, key=lambda trace: trace.network, reverse=reverse)
			if sortDate:
				sortedSet = sorted(sortedSet, key=lambda trace: trace.origin, reverse=reverse)
		if sort == 'station':
			sortedSet = sorted(waveformSet, key=lambda trace: trace.station, reverse=reverse)
			if sortDate:
				sortedSet = sorted(sortedSet, key=lambda trace: trace.origin, reverse=reverse)
		if sort == 'channel':
			sortedSet = sorted(waveformSet, key=lambda trace: trace.channel, reverse=reverse)
			if sortDate:
				sortedSet = sorted(sortedSet, key=lambda trace: trace.origin, reverse=reverse)
		if sort == 'gcarc':
			sortedSet = sorted(waveformSet, key=lambda trace: trace.stationID, reverse=reverse)
			sortedSet = sorted(sortedSet, key=lambda trace: trace.gcarc, reverse=reverse)
			if sortDate:
				sortedSet = sorted(sortedSet, key=lambda trace: trace.origin, reverse=reverse)
		if sort == 'az':
			sortedSet = sorted(waveformSet, key=lambda trace: trace.stationID, reverse=reverse)
			sortedSet = sorted(sortedSet, key=lambda trace: trace.az, reverse=reverse)
			if sortDate:
				sortedSet = sorted(sortedSet, key=lambda trace: trace.origin, reverse=reverse)
		if sort == 'baz':
			sortedSet = sorted(waveformSet, key=lambda trace: trace.stationID, reverse=reverse)
			sortedSet = sorted(sortedSet, key=lambda trace: trace.baz, reverse=reverse)
			if sortDate:
				sortedSet = sorted(sortedSet, key=lambda trace: trace.origin, reverse=reverse)
		if sort == 'quality':
			sortedSet = sorted(waveformSet, key=lambda trace: trace.quality, reverse=reverse)
			if sortDate:
				sortedSet = sorted(sortedSet, key=lambda trace: trace.origin, reverse=reverse)
	
	except:
		print 'sortDirectory() Unable to sort.'
		sortedSet = []
		pass
		
	return sortedSet
def splitDirectoryByDate(waveformSet):
	'''
	Given a list of Traces, this function will split up the set based on origin time. 
		This returns a dictionary including origin times on the first level, then an array
		of Traces corresponding to each origin time.
	'''
	totalSet = {}
	
	waveformSet = sortDirectory(waveformSet,sortDate=True,sort='complete',reverse=False)	
	
	for aTrace in waveformSet:
		eventTime = aTrace.origin.strftime('%Y-%m-%d %H:%M:%S')
		if eventTime in totalSet:
			totalSet[eventTime].append(aTrace)
		
		else:
			totalSet[eventTime] = []
			totalSet[eventTime].append(aTrace)
	
	return totalSet
def waves2dict(waveformSet,copy=False):
	'''
	Formats a set of waveforms into a dictionary format.
	
	If True, 'copy' will create a deepcopy of each trace.
	'''
	newSet = {}
	
	if not copy:
		for aTrace in waveformSet:
			eventTime = aTrace.origin.strftime('%Y-%m-%d %H:%M:%S')
			network = aTrace.network
			station = aTrace.station
			channel = '{}{}'.format(aTrace.staLocation,aTrace.channel)
			if eventTime in newSet:
				if network in newSet[eventTime]:
					if station in newSet[eventTime][network]:
						if channel in newSet[eventTime][network][station]:
							print 'Duplicate waveform: {}'.format(aTrace.stationID)
						else:
							newSet[eventTime][network][station][channel] = aTrace
					else:
						newSet[eventTime][network][station] = {}
						newSet[eventTime][network][station][channel] = aTrace
				else:
					newSet[eventTime][network] = {}
					newSet[eventTime][network][station] = {}
					newSet[eventTime][network][station][channel] = aTrace
			else:
				newSet[eventTime] = {}
				newSet[eventTime][network] = {}
				newSet[eventTime][network][station] = {}
				newSet[eventTime][network][station][channel] = aTrace
	
	else:
		for aTrace in waveformSet:
			eventTime = aTrace.origin.strftime('%Y-%m-%d %H:%M:%S')
			network = aTrace.network
			station = aTrace.station
			channel = '{}{}'.format(aTrace.staLocation,aTrace.channel)
			if eventTime in newSet:
				if network in newSet[eventTime]:
					if station in newSet[eventTime][network]:
						if channel in newSet[eventTime][network][station]:
							print 'Duplicate waveform: {}'.format(aTrace.stationID)
						else:
							newSet[eventTime][network][station][channel] = aTrace.copy()
					else:
						newSet[eventTime][network][station] = {}
						newSet[eventTime][network][station][channel] = aTrace.copy()
				else:
					newSet[eventTime][network] = {}
					newSet[eventTime][network][station] = {}
					newSet[eventTime][network][station][channel] = aTrace.copy()
			else:
				newSet[eventTime] = {}
				newSet[eventTime][network] = {}
				newSet[eventTime][network][station] = {}
				newSet[eventTime][network][station][channel] = aTrace.copy()
		
	return newSet
def dict2waves(waveDict,copy=False):
	'''
	Pulls waveforms out of a dictionary format.
	
	If True, 'copy' will create a deepcopy of each trace.
	'''
	waveformSet = []
	for aDate in waveDict:
		for aNetwork in waveDict[aDate]:
			for aStation in waveDict[aDate][aNetwork]:
				for aChannel in waveDict[aDate][aNetwork][aStation]:
					if not copy:
						waveformSet.append(waveDict[aDate][aNetwork][aStation][aChannel])
					else:
						waveformSet.append(waveDict[aDate][aNetwork][aStation][aChannel].copy())
	return waveformSet

##~~ Signal Processing ~~##
def xcorr(aTrace,bTrace):
	xcorrArray = signal.correlate(aTrace.data,bTrace.data)
# 	timeArray = 
	
	maxIndex = np.abs(xcorrArray).argmax(axis=0)
def computeFourierTransform(waveform):
	nft = nextpow2(waveform.data.size)
	npts = nft

	anFFT = np.fft.rfft(waveform.data,npts)*waveform.delta
	a     = np.abs(anFFT)

	fnyq = waveform.sampleRate * 0.5
	df   = 1/(nft/waveform.sampleRate)
	freq = df * np.arange(nft/2+1)

	return a,freq,df
def smooth(array, window_size=1):
	'''
	Convolves a boxcar of desired length to smooth the waveform (effectively averaging
		as a function of the window_size. Original timeseries length is preserved.
	
	:param window: Length of unit area boxcar to convolve with timeseries
	'''
	
	window = np.ones(int(window_size)) / float(window_size)
	array = np.convolve(array, window, 'same')
	
	return array

##~~ Plotting ~~##
def plotTraces(waveformSet,per=1,samePlot=False,relative=True,picks=True,
					sortDate=True,sort='gcarc',reverse=False,saveFile=None):
	'''
	sort:	'complete'	sort by network,station,channel
			'origin'
			'network'
			'station'
			'channel'
			'gcarc'		sort by great circle arc
			'az'		sort by azimuth
			'baz'		sort by back azimuth
			'quality'	sort by quality
	'''

	if isinstance(waveformSet,Trace):
		waveformSet.plot()

	else:
		plotCount = None
		
		waveformSet=sortDirectory(waveformSet,sortDate=sortDate,sort=sort,reverse=reverse)
		
		if not samePlot:
			if len(waveformSet) < per:
				per = len(waveformSet)
			elif saveFile:
				plotCount = 0
				fileBase = saveFile
			
			plotSet = []
			limits = [0,0,0,0]
			total = len(waveformSet)
			for aTrace in waveformSet:
			
# 				temp = '%s - GC:%0.2f Az:%0.1f M:%0.1f\nD:%0.1f T: %s' % \
# 					(aTrace.stationID,aTrace.gcarc,aTrace.az,aTrace.mag,aTrace.evDepth,aTrace.origin.strftime('%Y (%j) %H:%M:%S'))
# 				print temp
# 				print ''
			
				total -= 1
									
				if len(plotSet) < per:
					plotSet.append(aTrace)
				
					if (aTrace.npts*aTrace.delta) > limits[1]:
						limits[1] = aTrace.npts*aTrace.delta
					if min(aTrace.data) < limits[2]:
						limits[2] = min(aTrace.data)
					if max(aTrace.data) > limits[3]:
						limits[3] = max(aTrace.data)
				
				if len(plotSet) == per:
										
					plt.figure(figsize=(10,5), dpi=100)

					for ii,trace in enumerate(plotSet):						

						title = '%s - GC:%0.2f Az:%0.1f M:%0.1f\nD:%0.1f T: %s' % \
							(trace.stationID,trace.gcarc,trace.az,trace.mag,
							trace.evDepth,trace.origin.strftime('%Y (%j) %H:%M:%S'))

						if (ii+1) == per:
						
							if isinstance(plotCount,int):
								plotCount += 1
								saveFile = fileBase + '_' + str(plotCount)

							plt.subplot(per, 1, per)
							if relative:
# 								trace.plot(plot='y',xLabel='sec',limits=False,
# 											picks=picks,singlePlot=False,saveFile=saveFile)
								trace.plot(plot='y',xLabel='sec',limits=False,
											picks=picks,singlePlot=False,title=title,
											saveFile=saveFile)
							else:
# 								trace.plot(plot='y',xLabel='sec',limits=limits,
# 											picks=picks,singlePlot=False,saveFile=saveFile)
								trace.plot(plot='y',xLabel='sec',limits=limits,
											picks=picks,singlePlot=False,title=title,
											saveFile=saveFile)
					
						else:
							plt.subplot(per, 1, (ii+1))
							if relative:
# 								trace.plot(plot='n',showXLabel=False,xLabel='sec',
# 											picks=picks,singlePlot=False)
								trace.plot(plot='n',showXLabel=False,xLabel='sec',
											picks=picks,singlePlot=False,title=title)
							else:
# 								trace.plot(plot='n',showXLabel=False,xLabel='sec',
# 									limits=limits,picks=picks,singlePlot=False)
								trace.plot(plot='n',showXLabel=False,xLabel='sec',
									limits=limits,picks=picks,singlePlot=False,title=title)

					if total < per:
						per = total

					plotSet = []
					limits = [0,0,0,0]

		else:
			for aTrace in waveformSet:
				aTrace.plot(plot='n',xLabel='sec',limits=False,picks=False,showTitle=False)
			plt.show()
		
	plt.close('all')
   
def plotSpectrum(U,f,aTitle=None,xLabel=None,yLabel=None,titleFont=16,axesFont=16,
					limitRange=None,color='black',multiColor=False,legend=None,
					lineweight=1.5,linestyle='solid',grid=False,plot='y',saveFile=None):
	'''
	linestyle = solid, dashed, dashdot, dotted
	'''
	axes = plt.subplot(111)
	axLeg = plt.gca()

# 	if len(np.shape(U)) > 1:
# 		if len(np.shape(f)) > 1:
# 			if len(np.shape(f)) == len(np.shape(U)):
	
	legendTrue = False
	legendFig  = False
	
	if isinstance(U, list):
		if isinstance(f, list):
			if len(U) == len(f):
				for i,aU in enumerate(U):
					if multiColor:
						if legend and len(legend)==len(U):
							legendTrue = True
							if len(legend) > 10:
								legendFig = True
							axes.loglog(f[i],aU,linewidth=lineweight,linestyle=linestyle,label=legend[i])
						else:
							axes.loglog(f[i],aU,linewidth=lineweight,linestyle=linestyle)
					else:
						axes.loglog(f[i],aU,color,linewidth=lineweight,linestyle=linestyle)
				
			else:
				print 'Error(pySACio.plotSpectrum): multi-array U and f different lengths'
		
		else:
			for aU in U:
				if multiColor:
					axes.loglog(f, aU,linewidth=lineweight,linestyle=linestyle)
				else:
					axes.loglog(f, aU,color,linewidth=lineweight,linestyle=linestyle)
	
	else:
		axes.loglog(f, U,color,linewidth=lineweight,linestyle=linestyle)
	
	if aTitle:
		plt.title(aTitle, fontname='sans-serif', fontsize=titleFont)
	if xLabel:
		plt.xlabel(xLabel, fontname='sans-serif', fontsize=axesFont)
	if yLabel:
		plt.ylabel(yLabel, fontname='sans-serif', fontsize=axesFont)
	plt.axis('tight')
	
	if grid:
		plt.grid(True, which="both",linestyle=':',linewidth=0.25, color='0.7')
			
	if not limitRange:
		pltLimits = plt.axis()
		plt.xlim(pltLimits[0],pltLimits[1])
		
		yScale = 5
		plt.ylim(pltLimits[2],pltLimits[3]*yScale)

		axes.set_aspect('auto')	

	else:
		plt.xlim(limitRange[0],limitRange[1])
		plt.ylim(limitRange[2],limitRange[3])

		axes.set_aspect('auto')	
		limitRange = np.log10(limitRange)
		aspectRatio = (limitRange[1] - limitRange[0]) / (limitRange[3] - limitRange[2])
		axes.set_aspect(aspectRatio)

	if legendTrue:
		if legendFig:
			legCols = int(np.ceil(len(legend) / 20.0))
			figLegend = plt.figure()
			plt.figlegend(*axLeg.get_legend_handles_labels(), loc = 'upper left', ncol=legCols)

		else:
			axes.legend(loc=3,prop={'size':6})
# 			plt.legend(loc=3,prop={'size':6})
	
	if (plot.lower() == 'y') or saveFile:
		if not saveFile:
			plt.show()

			plt.clf()
			plt.close('all')
		
		else:
			if legendFig:
				figLegend.savefig((saveFile+'_legend.pdf'))
				plt.close(2)	

			plt.savefig((saveFile+'.pdf'))				
			
			plt.clf()
			plt.close('all')

def plotStations(centerLat,centerLon,staLat,staLon,staName=None,title=None,projection='ortho',
					plot='y',saveFile=None):
	'''
	projections:
		ortho: Centered on centerLat,centerLon. Displays the globe as an orthographic
					azimuthal projection
		robin: Centered at centerLon. Displays globe as a Robinson projection
	'''	
	# set up orthographic map projection with
	# use low resolution coastlines.
	if projection=='ortho':	
		map = Basemap(projection='ortho',lat_0=centerLat,lon_0=centerLon,resolution='l')
	elif projection=='merc':
		map = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,\
					llcrnrlon=-180,urcrnrlon=180,lat_ts=centerLat,resolution='c')
	else:
		print '{}: Unkown map projection. Using orthographic.'.format(projection)
		map = Basemap(projection='ortho',lat_0=centerLat,lon_0=centerLon,resolution='l')

	# draw coastlines, country boundaries, fill continents.
	map.drawcoastlines(linewidth=0.25)
	map.drawcountries(linewidth=0.25)

	# draw the edge of the map projection region (the projection limb)
# 	map.fillcontinents(color='coral',lake_color='aqua')
# 	map.drawmapboundary(fill_color='aqua')
	map.drawmapboundary(fill_color='white')

	# draw lat/lon grid lines every 30 degrees.
	if projection=='ortho':	
		map.drawmeridians(np.arange(0,360,30))
		map.drawparallels(np.arange(-90,90,30))

	elif projection=='merc':
		map.drawparallels(np.arange(-90.,91.,30.))
		map.drawmeridians(np.arange(-180.,181.,60.))

	else:	
		map.drawmeridians(np.arange(0,360,30))
		map.drawparallels(np.arange(-90,90,30))

	# plot hypocenter over the map.
	x, y = map(centerLon, centerLat)
	map.scatter(x,y,50,marker='*',color='r')

	# plot stations over the map.
	if len(staLon) == len(staLat):
		x, y = map(staLon, staLat)
		map.scatter(x,y,10,marker='v',color='b')
	else:
		print 'Error (plotStations): Station latitude and longitude arrays are different lengths.'

	# label stations.
	if staName:
		if (len(staName) == len(staLat)):
			for i,aN in enumerate(staName):
				aLa = staLat[i]
				aLo = staLon[i]
				x, y = map(aLo, aLa)
				plt.text(x,y,aN,fontsize=5,ha='left',va='bottom',color='b')
		else:
			print 'Error (plotStations): Station latitude and name arrays are different lengths.'

	# .
	if title:
		plt.title(title)
	else:
		plt.title('Station Map')

	if (plot.lower() == 'y') or saveFile:
		if not saveFile:
			plt.show()
		
		else:
			plt.savefig((saveFile+'.pdf'))	

		plt.clf()
		plt.close('all')

# To be done
'''
class TraceDict():
	Stores traces in a dictionary format. This then provides methods that are the same
	as those listed for the Trace class, but can be applied to the entire trace dictionary.

class Trace():
	def plotPick():
		similar functionality as SAC's PPK

def plotByStation():
	This function plots all three components of a station together

def rotateAndSave(seisStream,eInfo,directoryName,verbose=True):
	# try to rotate the horizontals, if it fails, output the original traces
	r = rotateHorizontalsToRT(seisStream)
	if(r != None):
		output = r
	else:
		output = seisStream
	
	if verbose:
		print output

def plotRecord(waveformSet,sort='gcarc',reverse=False,normalize=False):
	This code is largely hacked from the function recrodsection written 
		by Garrett Euler.
	
	Description:
		RECORDSECTION(DATA) draws all non-xyz records in SEIZMO struct DATA
		spaced out by their 'gcarc' header field values (degree distance
		from event location).  The records are normalized as a group with a
		maximum amplitude range corresponding to a third of the y axis range.
		Each record is drawn as a distinct color from the HSV colormap.
		Spectral records are converted to the time domain prior to plotting.

	sort:	'complete'	sort by network,station,channel
			'origin'
			'network'
			'station'
			'channel'
			'gcarc'		sort by great circle arc
			'az'		sort by azimuth
			'baz'		sort by back azimuth
			'quality'	sort by quality

	waveformSet = splitDirectoryByDate(waveformSet)
	
	for aEvent in waveformSet:
		eventSet = sortDirectory(waveformSet[aEvent],sortDate=True,sort=sort,reverse=reverse)

	#adjust plot
	plt.subplots_adjust(left=0.18)
	fig = plt.figure(figsize=(15,5),facecolor='k')	

	print 'plotRecord() under construction'
'''

#====================================================================
#====================================================================
# path='Waveforms/E*/Dsp/*lh?*.sac'
# path='Waveforms2/E*/Dsp/*.sac'
# wavesPerPage = 6
# path = '/Users/mcleveland/Documents/Projects/pySACio/Waveforms/E*/Dsp/mdj*bhz*.sac'
# mdj = readDirectory(path)
# prefDir = []
# for a in mdj:
# 	a.cut(-1,5,cutFrom='a')
# 	if (a.quality > 1):
# 		prefDir.append(a)		
# # 	a.envelope()
# plotTraces(prefDir,per=wavesPerPage,saveFile='MDJ')	
# # 
# path = '/Users/mcleveland/Documents/Projects/pySACio/Waveforms/E*/Dsp/kkar*bhz*.sac'
# kkar = readDirectory(path)
# 
# prefDir = []
# for a in kkar:
# 	a.cut(-1,5,cutFrom='a')
# 	if (a.quality > 1):
# 		prefDir.append(a)
# # 	a.envelope()
# plotTraces(prefDir,per=wavesPerPage,saveFile='KKAR')	

# test = signal.correlate(waves[0].data,waves[0].data)
# time = np.arange(-waves[0].npts*waves[0].delta,(waves[0].npts-1)*waves[0].delta,waves[0].delta)
