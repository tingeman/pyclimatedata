# Python v 3.7

from typing import Union, Any

try:
    # Import enhanced debugger, if available
    import ipdb as pdb
except:
    import pdb

from ctypes import ArgumentError
from importlib import reload
import pathlib
import zipfile
import io
import re
import fnmatch
import numpy as np
import pandas as pd
import datetime as dt
import copy
import chardet
import rich

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap

pd.set_option('future.no_silent_downcasting', True)

# Install required packages:
# conda install -c conda-forge ipdb rich
# conda install numpy pandas matplotlib ipython openpyxl


# =====================================================================================
# DESCRIPTION
# =====================================================================================

# There are two basic types of data classes:
# ClimateTimeSeries: Holds actual data pulled out of the DMI or other data files
# ClimateDataSet: A collection of ClimateTimeSeries
#
# The following classes are defined and inherit from ClimateTimeSeries:
# AirTemp     (*)
# DegreeDays  (*)
# AccumulatedDegreeDays (*)
# Wind
# Precipitation
# RelativeHumidity
# SnowDepth
# AirPressure
#
# Only the three first marked with (*) do actual work, like calculating daily averages, monthly and 
# annual averages etc. The rest are only basic timeseries. They hold the raw_data, and can merge 
# with other instances of the same type to combine two datasets into a longer timeseries, but they 
# don’t know how to average, calculate statistics etc.
#
# ClimateDataSet derived classes are defined to handle loading of particular filetypes:
# DMIType1: load DMI data in old dataformat (-2013)
# DMIType2: load DMI data in dataformat introduced 2014 (2014-2021)
# DMIType3: load DMI data in new dataformat introduced 2022 (data for all active stations, except snowdepth)
# DMITypeLong: Load the very long timeseries of daily averages, dating back to 1800’s (not fully implemented)
# DMIUnified: Load and combine both old and new DMI format files for the same station
# (… plus some custom classes for specific datasets I have used …)
#
# Four paths to the DMI zip files with datasets need to be updated by the user.
# DMI_PATH_UNIFIED_FORMAT      (Data from all stations in one unified format, introduced 2022)
# DMI_PATH_NEW_FORMAT          (Data from 2014-2021 for all active stations, introduced 2014) 
# DMI_PATH_OLD_FORMAT          (Data prior to 2014 from all stations)
# DMI_PATH_LONG_FORMAT         (Long time series of merged data from multiple stations)

# =====================================================================================
# EXAMPLES OF USAGE 
# =====================================================================================
#
# see __main__ section at end of module
#
# =====================================================================================
# TODOs
# =====================================================================================
#
# - Implement unit checking and conversion using the pint module
#       __init__ methods should take a 'unit' parameter, and automatically convert 
#       from specified to class defined default unit (specified for each class)
# - Implement method in ClimateDataSetBase to align timeseries of all datasets
#       (from first to last date valid for all series)
#       (or expand with NaNs to largest date span in dataset...?) 
# - May need utility method in ClimateTimeSeriesBase to interpolate/gapfill
#       use instance variable to specify gapfill method: interpolate, fill, None (do nothing)
# - 'fill' needs arguments, maybe gapfill_args = [xxx,xxx,xxx], gapfill_kwargs = {xxx:yyy,...}
# - Should we have a gapfill_from(other) method?
#       or should it be up to user to handle this on case by case basis?
# - Should we have a ClimateDataCombined class, which is instantiated when 
#       two time series are merged?
#       It could keep track of which rows comes from which series
#       and delegate certain tasks to the original source instance

# see also __main__ section at end of module

# =====================================================================================
# More EXAMPLES OF USAGE 
# =====================================================================================
#
# THE EXAMPLES BELOW ARE NOT GUARANTEED TO WORK 
#
# import climate_data as cdata
# from matplotlib import pyplot as plt
#
# # load data using the load_climate_data function (takes filename as input)
# cd_old = cdata.load_climate_data('04221.txt', repository=cdata.DMI_PATH_OLD_FORMAT)
#
# # load data using the individual classes (takes filenames as input)
# cd_old2 = cdata.DMIType1('04221.txt')
# cd_new = cdata.DMIType2('4221_2014_2020.csv')
# cd_combined = cd_old.merge_with(cd_new)
#
# # using the DMIUnified class (takes station name as input)
# cd_combined2 = cdata.DMIUnified('4221')
#
# # datasets are stored in “.datasets”:
# cd_combined2.datasets    # list the loaded datasets
#
# # plotting a dataset
# cd_combined.datasets['AT'].daily_avg.plot()
# plt.show(block=False)


# =====================================================================================
# TODO list
# =====================================================================================

# TODO
# Should we have a ClimateDataCombined class, which is instantiated when 
# two time series are merged?
# It could keep track of which rows comes from which series
# and delegate certain tasks to the original source instance


# =====================================================================================
# DEFINITIONS (To be updated by user)
# =====================================================================================

# Paths to DMI datafiles in old format (1958-2013) and new format (2014-present), 
# as well as the long datasets dating back to the late 1800s.
# Paths can be to proper folders or to relevant zip-files. If a zip file is provided
# the code will look for the station file inside the zip.
# If variable holds a list of paths, the code will search each path successively.
DMI_PATH_UNIFIED_FORMAT = [pathlib.Path(r'C:\thin\02_Data\DATA_Monitoring\ClimateData\DMI_climate_data\DMIRep22-08_1953_2021_data1.zip'),
                           pathlib.Path(r'C:\thin\02_Data\DATA_Monitoring\ClimateData\DMI_climate_data\DMIRep22-08_1953_2021_data2.zip')]
DMI_PATH_NEW_FORMAT = pathlib.Path(r'C:\thin\02_Data\DATA_Monitoring\ClimateData\DMI_climate_data\DMIRep21-08_new_dataformat_2014_2020.zip')
DMI_PATH_OLD_FORMAT = pathlib.Path(r'C:\thin\02_Data\DATA_Monitoring\ClimateData\DMI_climate_data\DMIRep21-08_old_dataformat_1958_2013.zip')
DMI_PATH_LONG_FORMAT = pathlib.Path(r'C:\thin\02_Data\DATA_Monitoring\ClimateData\DMI_climate_data\DMIRep21-04.zip')

# The data should be obtained from the most recent DMI report (at time of writing, 21-08 and 21-04).
# The reports can be accessed here: https://www.dmi.dk/publikationer/


# =====================================================================================
# CUSTOM EXCEPTIONS
# =====================================================================================


class NotCompatibleError(Exception):
    pass


# =====================================================================================
# FILE SYSTEM HANDLING
# =====================================================================================

class DataRepository():
    """Defines a common interface to handling data files stored in a directory or a zip-file.
    Implements 'dir', 'glob' and 'get_file' methods which works the same on both directories and zip files.

    2022-12-18: Now also works with a list of paths (e.g. list of zip-files). Will find the requested
                file in any of the listed paths (whether zip or real folder)

    """
    def __init__(self, datapath):
        if isinstance(datapath, list):
            self.multipath = True
            self.path = []
            suffixes = []
            for dp in datapath:
                if dp.exists():
                    self.path.append(dp)
                else:
                    raise IOError('Location does not exist: {0}'.format(dp))    
                suffixes.append(dp.suffix.lower()=='.zip')
            
            if all(suffixes):
                self.zipped = True
            elif not any(suffixes):
                self.zipped = False
            else:
                raise ValueError('Datapath contains a mix of folders and zip-files. This feature is not implemented.')    
        else:
            self.multipath = False
            self.path = pathlib.Path(datapath)
            if not self.path.exists():
                raise IOError('Location does not exist: {0}'.format(self.path))

            if self.path.suffix.lower() == '.zip':
                self.zipped = True
            else:
                self.zipped = False

    def dir(self):
        """Return a list of filenames in the current repository (list of strings)"""
        filelist = []
        if self.multipath:
            pathlist = self.path
        else:
            pathlist = [self.path]
        
        for dp in pathlist:
            if self.zipped:
                filelist.extend(zipfile.ZipFile(dp).namelist())
            else:
                filelist.extend([f.name for f in dp.glob('*.*')])
        return filelist

    def glob(self, pattern):
        """Return a list of filenames matching the filename pattern (list of strings)"""

        filelist = []
        if self.multipath:
            pathlist = self.path
        else:
            pathlist = [self.path]

        if self.zipped:
            re_pattern = re.compile(fnmatch.translate(pattern))
            filelist.extend(filter(re_pattern.match, self.dir()))
        else:
            for dp in pathlist:
                filelist.extend([f.name for f in dp.glob(pattern)])
        return filelist

    def exists(self, pattern):
        """Check if filename or pattern exists in repository"""
        if len(self.glob(pattern))>0:
            return True
        else:
            return False

    def get_filepath(self, filename):
        """Return the path that contains the filename passed."""
        
        if self.multipath:
            pathlist = self.path
        else:
            pathlist = [self.path]

        for dp in pathlist:
            if self.zipped:
                namelist = zipfile.ZipFile(dp).namelist()
                
                if filename in namelist:
                    # File exists in the root of the zipfile, so return path to zipfile
                    return dp
                else:
                    for f in namelist:
                        if f.endswith(filename):
                            # filename was found at the end of a longer path, so return path to zipfile
                            return dp
            else:
                if (dp / filename).exists():
                    return dp
            
        raise IOError('File does not exist in repository ({0})'.format(filename))

    def get_file(self, filename):
        """Return a DataFile instance populated with the data from the specified file."""
        if self.multipath:
            path = self.get_filepath(filename)
        else:
            path = self.path

        return DataFile(filename, path)

    def dirlist(self):
        title = 'Contents of: {0}'.format(self.path)
        rich.print(rich.columns.Columns(sorted(self.dir()), 
                                        equal=True, expand=True,
                                        column_first=True,
                                        title=title))

class DataFile(io.BytesIO):
    """Defines a common interface to reading data files stored in directories and inside zip-files.
    Reads and holds the data from the file in memory. 
    Consider deleting the instance when it is no longer needed, if memory usage is an issue.
    """
    def __init__(self, filename, path=None):
        super().__init__()

        self._enc = None
        self.encoding = None
        self.encoding_confidence = None

        if path is None:
            self._filename = pathlib.Path(filename)
        else:
            self._filename = pathlib.Path(path) / filename
        
        if self._filename.parent.suffix == '.zip':
            self._read_zipped_data()
        else:
            self._read_plain_data()
        self.get_encoding()

    def _read_zipped_data(self):
        """Reads the contents of a file inside a zip-file and place it
        in the StringIO buffer. 
        """
        if not self._filename.parent.exists():
            raise FileNotFoundError('File does not exist: {0}'.format(self._filename.parent))

        zf = zipfile.ZipFile(self._filename.parent)

        namelist = zf.namelist()

        if self._filename.name in namelist:
            # File exists in the root of the zipfile
            with zf.open(self._filename.name, 'r') as file:
                self.write(file.read())
        else:
            # File was not found in the root, check in subfolders
            file_exists = False
            for f in namelist:
                if f.endswith(self._filename.name):
                    # filename was found at the end of a longer path, so read it...
                    with zf.open(f, 'r') as file:
                        self.write(file.read())
                        file_exists = True
                    break   
            if not file_exists:
                raise FileNotFoundError('File does not exist: {0}'.format(self._filename))

    def _read_plain_data(self):
        """Reads the contents of a regular file and place it in the BytesIO buffer."""
        if not self._filename.exists():
            raise FileNotFoundError('File does not exist: {0}'.format(self._filename))

        with open(self._filename, 'rb') as file:
            self.write(file.read())

    def get_encoding(self, count=300):
        """Detect the encoding of the file."""
    
        # If we passed an open file
        self.seek(0)
        dat = self.read(count)
        self.seek(0)
    
        self._enc = chardet.detect(dat)
        self.encoding = self._enc['encoding']
        self.encoding_confidence = self._enc['confidence']


# =====================================================================================
# HELPER FUNCTIONS
# =====================================================================================


def farenheit2celsius(f):
    """Convert degrees farenheit to celcius."""
    return (f - 32) * 5.0/9.0

def celsius2farenheit(c):
    """Convert degrees celcius to farenheit."""
    return c * 9.0/5.0 + 32

def inches2mm(inches):
    """Convert inches to millimeters."""
    return inches * 25.4

def mm2inches(mm):
    """Convert millimeters to inches."""
    return mm / 25.4

def write_xlsx(file, df, sheet, cols=None, drop_cols=None, overwrite=False, **kwargs):
    """Writes the dataframe to an excel file and the specified sheet name.

    file:       filename string, or ExcelWriter instance
    df:         the dataframe to write
    sheet:      name of the sheet in which to write the dataframe
    cols:       list of columns to write
    drop_cols:  list of columns not to write
    overwrite:  if True, any existing file will be overwritten
                if False, an integer will be appended to filename to make it unique

    Obviously, cols and drop_cols should be considered mutually exclusive.
    You should use one or the other.

    Additional keyword arguments will be passed to df.to_excel method call.
    """

    if isinstance(file, str):
        file = pathlib.Path(file)

    if not isinstance(file, pd.ExcelWriter):
        if file.exists():
            print('file: {0} exists!'.format(file))
            if overwrite:
                # remove existing file
                file.unlink()     #   missing_ok=True)
            else:
                # create new filename
                print('creating new filename...')
                n = 0
                file = (file.parent / (file.stem + '_{0}'.format(n) + file.suffix))
                while file.exists():
                    print('file: {0} exists!'.format(file))
                    n = n + 1
                    file = (file.parent / (file.stem + '_{0}'.format(n) + file.suffix))

                print('Creating file: {0}'.format(file))

        file = pd.ExcelWriter(file)

    cols_to_write = df.columns

    if cols is not None:
        cols_to_write = [val for val in cols_to_write if val in cols]

    if drop_cols is not None:
        cols_to_write = [val for val in cols_to_write if val not in drop_cols]

    df.to_excel(file, sheet_name=sheet, columns=cols_to_write, **kwargs)

    return file


def get_file_version(file_or_filename, encoding=None):
    # try to identify dmi data file version
    try:
        # If we passed an open file or a DataFile
        file_or_filename.seek(0)
        first_line = file_or_filename.readline()
        second_line = file_or_filename.readline()
        if encoding is not None:
            first_line = first_line.decode(encoding)
            second_line = second_line.decode(encoding)
        elif hasattr(file_or_filename, 'encoding'):
            first_line = first_line.decode(file_or_filename.encoding)
            second_line = second_line.decode(file_or_filename.encoding)
    except:
        # If we passed a filename/path
        if encoding is None:
            with open(file_or_filename, 'r') as f:
                first_line = f.readline()
                second_line = f.readline()
        else:
            with open(file_or_filename, 'rb') as f:
                first_line = f.readline().decode(encoding)
                second_line = f.readline().decode(encoding)

    #with open(file, 'r') as f:
    #    first_line = f.readline()

    if first_line.startswith('Station;') and (('101' in first_line) or ('601' in first_line)):
        if 'Year' in first_line:
            # This is the new DMI data format introduced in 2022
            version = 'DMI3'
        else:
            # This is the DMI data format starting from 2014
            version = 'DMI2'
    elif first_line.startswith('Station;date(utc);'):
        # NOT SURE WHICH FORMAT THIS REPRESENTS - IS IT EVEN USED?
        version = 'DMIX'
    elif first_line.startswith('stat_no\tyear\t'):
        # This is the old DMI data format for files holding data up until 2013
        version = 'DMI1'
    elif first_line.startswith('stat_no;year;'):
        # This is the DMI data format for files holding combined long data series (e.g. report 19-04)
        version = 'DMILong'
    elif 'pgtm' in first_line.lower():
        # This is the format retrieved from the NESDIS online database at https://www.ncdc.noaa.gov/cdo-web/
        version = 'NESDIS'
    elif 'yrmodahrmn;temp' in first_line.lower():
        # Special format of some old data received from Jacob Yde...
        version = 'US_YDE'
    else:
        raise IOError('Unable to determine file version...!')

    return version


def get_encoding(file_or_filename):
    """Detect the encoding of the file passed."""
    try:
        # If we passed an open file
        file_or_filename.seek(0)
        dat = file_or_filename.read(300)
        file_or_filename.seek(0)
    except AttributeError:
        # if direct read failed, try treating the input as a filename/path
        with open(file_or_filename, 'rb') as f:
            dat = f.read(300)
    enc = chardet.detect(dat)
    return enc


def load_climate_data(stno_or_file_or_filename, repository=None):
    """Instantiator function to make sure the right class is chosen for the file type"""
    filename = None

    if repository is None:
        if pathlib.Path(stno_or_file_or_filename).name == stno_or_file_or_filename:
            # no path information was passed, assume DMI UNIFIED format and repository
            drp = DataRepository(DMI_PATH_UNIFIED_FORMAT)
        else:
            # argument contains path information
            file_path = pathlib.Path(stno_or_file_or_filename.parent)
            if file_path.exists():
                drp = DataRepository(file_path)
            else:
                raise ArgumentError('Specified file path does not exist')
    elif isinstance(repository, DataRepository):
        drp = repository
    else:
        drp = DataRepository(repository)

    if hasattr(stno_or_file_or_filename, '_filename'):
        filename = pathlib.Path(stno_or_file_or_filename._filename)
    elif isinstance(stno_or_file_or_filename, pathlib.Path):
        filename = stno_or_file_or_filename
    elif isinstance(stno_or_file_or_filename, str):
        # Check stno_or_file_or_filename is only alphanumeric
        if stno_or_file_or_filename.isalnum():
            # Assume it is a station number, and add the .txt extension
            filename = pathlib.Path(stno_or_file_or_filename + '.csv')
        else:
            # Consider it a filename, possibly including a path
            filename = pathlib.Path(stno_or_file_or_filename)
    else:
        raise ArgumentError('Unknown file type of filename/path type')

    filename = filename.name
    file = drp.get_file(filename)
    version = get_file_version(file)

    if version is not None:
        print('Parsing file as version: {0}  [encoding: {1} ({2})]'.format(version, file.encoding, file.encoding_confidence))
        file.seek(0)
        cd = CLIMATE_FILE_CLASSES[version](file)
    return cd



# =====================================================================================
# CLIMATE TIMESERIES CLASSES
# =====================================================================================

class ClimateTimeSeriesBase():
    identifier = None
    unit = None
    daily_aggregator = None    # could be one of 'mean', 'sum', 'max' (see Pandas GroupBy computation methods)
    days = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
    months = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    _attrs = ['identifier', 'unit', 'daily_aggregator']

    def __init__(self, *args, **kwargs):

        #if hasattr(self.identifier, '__iter__'):
        if not isinstance(self.identifier, str):
            npar = len(self.identifier)
        elif self.identifier is not None:
            npar = 1
        else:
            npar = 0

        self.raw_data = None
        self.daily_data = None
        name_list = []

        if npar >= 1:
            if len(args) == npar:
                if npar >= 2:
                    args_list = [pd.to_numeric(a) for a in args if a is not None]
                    name_list = [self.identifier[k] for k in range(len(args)) if args[k] is not None]
                else:
                    args_list = [pd.to_numeric(a) for a in args if a is not None]
                    name_list = [self.identifier for k in range(len(args)) if args[k] is not None]
            elif len(args) == 0:
                if len(kwargs)>0:
                    args_list = []
                    name_list = []
                    if npar >= 2:
                        for k in self.identifier:
                            if (k in kwargs) and (kwargs[k] is not None):
                                args_list.append(pd.to_numeric(kwargs[k]))
                                name_list.append(k)
                    else:
                        if kwargs[self.identifier] is not None:
                            args_list = [pd.to_numeric(kwargs[self.identifier])]
                            name_list = [self.identifier]

            if len(name_list) > 0:
                self.raw_data = pd.concat(args_list, axis=1)
                self.raw_data.columns = name_list

    def register_attrs(self, attrs=[]):
        self._attrs = self._attrs
        self._attrs.extend(attrs)
        # Now make sure only unique values exist
        #if len(attrs)>0:
        #    used = set()
        #    self._attrs = [x for x in self._attrs if x not in used and (used.add(x) or True)]

    def to_xarray(self):
        import xarray as xr
        ds = xr.Dataset.from_dataframe(self.raw_data)
        for k in self._attrs:
            try:
                v = getattr(self, k)
            except:
                continue
            if v is not None:
                ds.attrs[k] = v
            else:
                ds.attrs[k] = 'None'
        ds.attrs['__class__'] = '{0}.{1}'.format(self.__class__.__module__, self.__class__.__name__)
        return ds

    @classmethod
    def from_xarray(cls, ds):
        new = cls()
        new.raw_data = ds.to_dataframe()
        for k, v in ds.attrs.items():
            if k == '__class__': continue
            if v == 'None':
                setattr(new, k, None)
            else:
                setattr(new, k, v)
        new.aggregate_daily()        
        return new

    def aggregate_daily(self):
        """Generic daily aggregator method. Maybe used to aggregate raw data into
        daily values using a daily mean, sum or max value, depending on the value
        of self.daily_aggregator.
        
        Method may be overwritten by child classes, to implement special functionality.
        """

        def _aggregate(grouped, parameter, aggregator):
            # calculate aggregated value and count number of entries on each date
            agg_values = getattr(grouped[parameter], aggregator)()
            agg_count = grouped[parameter].count()
            agg_count.name = 'count_'+agg_count.name
            return agg_values, agg_count

        if (self.raw_data is None):
            return

        if self.daily_aggregator is not None:   
            grouped = self.raw_data.groupby(self.raw_data.index.date)
            dataseries = []
            countseries =  []

            if isinstance(self.identifier, str):
                if self.daily_aggregator.lower() != 'none':
                    # calculate aggregated value and count number of entries on each date
                    values, counts = _aggregate(grouped, self.identifier, self.daily_aggregator)
                    dataseries.append(values)
                    countseries.append(counts)
            else:
                for idx,par_id in enumerate(self.identifier):
                    if (self.daily_aggregator[idx] is not None) and (self.daily_aggregator[idx].lower() != 'none'):
                        # calculate aggregated value and count number of entries on each date
                        values, counts = _aggregate(grouped, par_id, self.daily_aggregator[idx])
                        dataseries.append(values)
                        countseries.append(counts)

            if len(dataseries) >= 0:
                df_agg = pd.concat([*dataseries, *countseries], axis=1)
                self.daily_data = df_agg
            else:
                self.daily_data = None
        else:
            self.daily_data = None

        if self.daily_data is not None:
            self.daily_data = self.daily_data.asfreq('D')  # Make sure the timeseries is continuous

            if not isinstance(self.identifier, str):
                idx = [k for k in self.identifier if k in self.daily_data]
            else:
                idx = self.identifier
            self.daily_data = self.daily_data[self.daily_data[idx].first_valid_index():self.daily_data[idx].last_valid_index()]

    def merge_with(self, other, inplace=True):
        if (isinstance(other, type(self))) or (other is None):
            if inplace:
                new_cd = self
            else:
                new_cd = copy.deepcopy(self)
        else:
            raise NotCompatibleError('Unable to merge instances of {0} with {1}'.format(type(self), type(other)))       

        if other is not None:
            new_cd.raw_data = self.raw_data.combine_first(other.raw_data)
            new_cd.raw_data.sort_index(inplace=True)

        new_cd.filename = None
        return new_cd

    def scale_debias_to(self, other):
        """Scale and bias correct this timeseries to other timeseries.
        return new combined timeseries.
        """
        if isinstance(other, type(self)) or (other is None):
            new = copy.deepcopy(self)
        else:
            raise ValueError('Unable to scale instances of {0} to {1}'.format(type(self), type(other)))

        print('{0}: No scaling/bias correction implemented, retaining data without modification.'.format(self.identifier))

        return new

    def get_ymd(self, df):
        """Return dataframe with year, month and weekday columns, which may be later used for 
        calculating annual averages
        """
        date_series = df.index.to_series()
        return pd.DataFrame({'year': date_series.apply(lambda x: x.year),
                             'month': date_series.apply(lambda x: x.month),
                             'weekday': date_series.apply(lambda x: x.weekday())})

    def nominal_counts(self, yearlist):
        """Calculate the nominal counts for each year in the list passed.
        Will calculate:
        - Total number of days in year
        - Number of days in each month of the year
        - Number of days that are mondays, tuesdays etc. of this year
        """
        dates = pd.date_range(start="{0:d}-01-01".format(int(yearlist[0])),
                              end="{0:d}-12-31".format(int(yearlist[-1])))

        df = pd.DataFrame({'year':dates.year, 'month':dates.month, 'weekday': dates.weekday})

        month_stats = df.groupby(['year', 'month']).agg(ndays=('month', "size")).unstack().fillna(0)
        month_stats.columns = [self.months[x[1]] for x in month_stats.columns]

        weekday_stats = df.groupby(['year', 'weekday']).agg(ndays=('weekday', "size")).unstack().fillna(0)
        weekday_stats.columns = [self.days[x[1]] for x in weekday_stats.columns]

        nday_stats = df.groupby(['year']).agg(ndays=('month', "size"))

        stats = pd.concat([nday_stats, month_stats, weekday_stats], axis=1)
        return stats.loc[stats.index.isin(yearlist)]

    def limit(self, lim:list[Union[str,dt.datetime,dt.date, None]]):
        self.raw_data = self._limit(lim, 'raw_data')
        self.daily_data = self._limit(lim, 'daily_data')


    def _limit(self, lim:list[Union[str,dt.datetime,dt.date, None]], dataset:str='raw_data'):
        if len(lim) != 2:
            raise ValueError('lim must be a two element iterable.')
        
        ds = getattr(self, dataset)

        if ds is not None:
            if lim[0] is None:
                lim[0] = ds.first_valid_index()
            if lim[1] is None:
                lim[1] = ds.last_valid_index()

            return ds[(ds.index>=lim[0]) & (ds.index<=lim[1])]
        else:
            return None


class AirTemp(ClimateTimeSeriesBase):
    identifier = ['AT', 'ATmin', 'ATmax']
    unit = ['celsius', 'celsius', 'celsius']    # units as defined by the Pint package
    daily_aggregator = 'mean'   

    def __init__(self, AT=None, ATmin=None, ATmax=None, minmax_data_threshold=6):
        # Takes air temperature time series (pandas Series) with datetime index as input argument.
        #
        # Data aggregation criteria:
        #
        # If number of measurements per day is less than 6, use average of min and max temperature
        #      Do not count the other measurements from those days in aggregation
        #
        # If more than 6 measurements per day, use actual measurements in daily average
        #
        # For annual averages, filter out days with less than 20 measurements, assuming 24 measurements per day as nominal frequency

        # store names of attributes that should be saved to disk when storing instance.
        self.register_attrs(['data_freq', 'daily_data_threshold', 'annual_data_threshold', 
                             'dd_data_threshold', 'minmax_data_threshold'])

        # Nominal data frequency
        self.data_freq = 24  # measurements per day

        # Data threshold for daily averaging
        self.daily_data_threshold = 6  # measurements per day

        # Data threshold for annual averaging
        self.annual_data_threshold = 340  # measurements per year
        # Set this to 0 to include all years that have data

        # Data threshold for annual FDD/TDD calculation
        self.dd_data_threshold = 340  # measurements per year
        
        # threshold for using min/max averaging
        self.minmax_data_threshold = minmax_data_threshold

        super().__init__(AT=AT, ATmin=ATmin, ATmax=ATmax)

        #self.raw_data = pd.concat([at_series, at_min, at_max], axis=1, keys=['AT', 'min', 'max'])
        if self.raw_data is not None:
            self.aggregate_daily()

    def extract_AT_min_max(self):
        """Extract min and max temperatures based on specific
        times of day, on days where the daily data threshold is not met,
        and where min/max values are not aready available."""

        counts = self.raw_data.groupby(self.raw_data.index.date).count()

        if ('ATmin' in counts.columns) & ('ATmax' in counts.columns):
            # We already have some ATmin/ATmax values

            # PSEUDO CODE
            # where AT.count < data threshold
            #       and ATmin.count = 0
            #       and ATmax.count = 0
            #    take 00:00:00 as ATmin (if it exists)
            #    take 12:00:00 as ATmax (if it exists)

            idx = (counts['AT']<self.daily_data_threshold) & (counts['ATmin']==0) & (counts['ATmax']==0).values
            
            ds_idx = self.raw_data.index.floor('d').isin(counts[idx].index)

            rd = self.raw_data
            rd['ATmin'].iloc[ds_idx & (rd.index.hour==0) & (rd.index.minute==0)] = rd[ds_idx & (rd.index.hour==0) & (rd.index.minute==0)]['AT']
            rd['ATmax'].iloc[ds_idx & (rd.index.hour==12) & (rd.index.minute==0)] = rd[ds_idx & (rd.index.hour==12) & (rd.index.minute==0)]['AT']

        else:
            # We have no ATmin/ATmax values, and must create the columns

            # PSEUDO CODE
            # where AT.count < data threshold
            #       and ATmin.count = 0
            #       and ATmax.count = 0
            #    take 00:00:00 as ATmin (if it exists)
            #    take 12:00:00 as ATmax (if it exists)

            idx = (counts['AT']<self.daily_data_threshold).values
            
            ds_idx = self.raw_data.index.floor('d').isin(counts[idx].index)

            rd = self.raw_data
            rd['ATmin'].iloc[ds_idx & (rd.index.hour==0) & (rd.index.minute==0)] = rd[ds_idx & (rd.index.hour==0) & (rd.index.minute==0)]['AT']
            rd['ATmax'].iloc[ds_idx & (rd.index.hour==12) & (rd.index.minute==0)] = rd[ds_idx & (rd.index.hour==12) & (rd.index.minute==0)]['AT']


    def aggregate_daily(self):
        if (self.raw_data is None):
            return

        if self.daily_aggregator is not None:   
            grouped = self.raw_data.groupby(self.raw_data.index.date)

            if ('ATmin' in self.raw_data.columns) and ('ATmax' in self.raw_data.columns):
                if 'AT' in self.raw_data.columns:
                    # We have both AT, ATmin and ATmax, use mean(AT) where possible.

                    # Use either min/max average or actual data value averages, depending on daily data count
                    # some years have max and min measured twice per day... thus take the max.max and min.min
                    mean_values = np.where(grouped['AT'].count() < self.minmax_data_threshold,
                                            (grouped['ATmax'].max() + grouped['ATmin'].min()) / 2,
                                            grouped['AT'].mean())

                    # For days where min/max average was used, set count to -1
                    mean_count = np.where(grouped['AT'].count() < self.minmax_data_threshold,
                                          -np.ones(len(mean_values)),
                                          grouped['AT'].count())
                else:
                    # We only have ATmin and ATmax, no AT
                    # Use always average of ATmin and ATmax
                    
                    # some years have max and min measured twice per day... thus take the max.max and min.min
                    mean_values = (grouped['ATmax'].max() + grouped['ATmin'].min()) / 2
                    mean_count = -np.ones(len(mean_values))
        
                # Set count to 0 on days where min/max calculation is unsuccessful
                mean_count = np.where(np.logical_and(mean_count == -1, np.isnan(mean_values)),
                                      np.zeros(len(mean_count)),
                                      mean_count)
            else:
                # if ATmin and ATmax are not specified, just calculate means and counts
                # from AT time series measurements
                mean_values = grouped.mean()
                mean_count = grouped.count()

            # Create new dataframe with daily averages
            df_mean = pd.DataFrame(mean_values, index=pd.to_datetime(self.raw_data.index.date).unique())
            df_mean.columns = ['AT']
            df_mean['count'] = mean_count

            #self.daily_avg = df_mean
            self.daily_data = df_mean
        else:
            self.daily_data = None

        if self.daily_data is not None:
            self.daily_data = self.daily_data.asfreq('D')  # Make sure the timeseries is continuous
            if not isinstance(self.identifier, str):
                idx = [k for k in self.identifier if k in self.daily_data]
            else:
                idx = self.identifier
            self.daily_data = self.daily_data[self.daily_data[idx].first_valid_index():self.daily_data[idx].last_valid_index()]

        
    def calc_MOAT(self):
        """Calculate monthly average air temperatures and statistics.
        Days with less than daily_data_threshold observations will be excluded from averages.
        """
        temp = self.daily_data.drop('count', axis=1)
        
        # add ymd information
        temp = pd.concat([temp, self.get_ymd(temp)], axis=1)

        # Exclude days with too few measurements
        id_exclude = np.logical_and(self.daily_data['count'] >= 0, self.daily_data['count'] <= self.daily_data_threshold)
        temp.loc[id_exclude, 'AT'] = np.nan
        
        # flag any incomplete days, so they will not be included in the statistics for annual averages
        temp.year = np.where(np.isnan(temp['AT']), temp['AT'], temp.year)
        temp.month = np.where(np.isnan(temp['AT']), temp['AT'], temp.month)
        temp.weekday = np.where(np.isnan(temp['AT']), temp['AT'], temp.weekday)

        # Calculate the MAAT and related monthly and weekday stats
        MAAT = temp.groupby(['year']).agg(MAAT=('AT', "mean"), ndays=('AT', "size"))

        MAAT_month_AT = temp.groupby(['year', 'month']).agg(ndays=('AT', "mean")).unstack().fillna(0)
        MAAT_month_AT.columns = ['T_{0}'.format(self.months[x[1]]) for x in MAAT_month_AT.columns]

        MAAT_month_stats = temp.groupby(['year', 'month']).agg(ndays=('AT', "size")).unstack().fillna(0)
        MAAT_month_stats.columns = [self.months[x[1]] for x in MAAT_month_stats.columns]

        MAAT_weekday_stats = temp.groupby(['year', 'weekday']).agg(ndays=('AT', "size")).unstack().fillna(0)
        MAAT_weekday_stats.columns = [self.days[x[1]] for x in MAAT_weekday_stats.columns]

        MOAT_stats = pd.concat([MAAT, MAAT_month_AT, MAAT_month_stats, MAAT_weekday_stats], axis=1)
        # Reindex to get a full timeseries without gaps
        new_index = np.arange(MOAT_stats.index[0] - 1, MOAT_stats.index[-1] + 2, 1, dtype=float)
        MOAT_stats = MOAT_stats.reindex(index=new_index)

        return MOAT_stats
    
    def calc_MAAT(self):
        """Calculate mean annual air temperature of timeseries, while correcting for any common 
        deficiencies in the measurements. F.ex. in some older datasets, air temperatures were 
        manually observed, and observations were only done on weekdays (thus saturdays and
        sundays are missing from dataset. Since these are evenly distributed across the year,
        reasonably good estimations of MAAT can still be obtained.
        
        The column MAAT holds the raw mean annual air temperature.
        The column MAAT_corr holds the corrected mean annual air temperature.
        
        Correction methodology:
        First the monthly mean air temperature is calculated based on all available data for a 
        given month in a given year.
        Then the monthly average temperature is multiplied by the nominal day count for that
        month to give a monthly degree day value.
        The monthly degree day valuess are then summed over the year, and divided by the 
        nominal day count for that year.
        
        Skip flags:       
        Certain problems with the data would result in erroneus corrected MAAT values.
        E.g. if full months are missing, while it may be OK if e.g. measurements on saturdays and
        sundays are missing.
        
        Problems are indicated with the skip flag column. If skip flag is True, the MAAT value 
        should not be considered (but it is still included in the output).
        
        Weekdays are considered completely missing in a given year, if observations were obtained 
        on less than 5 days of a particular weekday, e.g. if observations were obtained on Mondays 
        only 4 times in a year.
        
        Months are considered completely missing in a given year, if observations were obtained
        on less than 10 days in a particular month.
        
        Weekdays that are completely missing from the dataset is OK, since the missing data will 
        be spread evenly across the year, and therefore not give a large bias. But only if there
        are not too many weekdays missing. We set the threshold to 2 (if more than two weekdays
        are completely missing, skip the year from calculations).
        
        Months that are completely missing will significantly bias the calculations, so we skip
        years with one or more months completely missing.
        """
        maat = self.calc_MOAT()
        nc = self.nominal_counts(maat.index)
        maat['skip_flag'] = False
        maat['missing_count'] = 0
        maat['wdays_missing'] = 0
        maat['months_missing'] = 0

        for day in list(self.days.values()):
            # looping over names of the days of the week
        
            # we consider a weekday 'completely missing' from the dataset if observations were 
            # obtained on this weekday less than 5 times in a year

            # If this particular day of week is 'completely missing', increase the count
            # of the 'missing weekdays' (wdays_missing) for that year.
            maat.loc[maat[day] <= 5, 'wdays_missing'] += 1
            # wdays_missing will range from 0 to 7.

            # For years where the weekday is not 'completely missing', calculate how many
            # times observations were NOT measured on this particular day of the week.
            maat.loc[maat[day] > 5, 'missing_count'] += (nc - maat).loc[maat[day] > 5, day]

        for month in list(self.months.values()):
            # looping over names of the months of the year
            
            # and count how many of the months had less than 10 days with observations measured
            maat.loc[maat[month] <= 10, 'months_missing'] += 1

        # set the skip flag for years where more than two weekdays are missing
        maat.loc[maat['wdays_missing'] > 2, 'skip_flag'] = True

        # set the skip flag for years where one full month (or more) is missing
        maat.loc[maat['months_missing'] >= 1, 'skip_flag'] = True

        # skip where more than 20 days are missing, not counting weekdays that are consistently not measured
        # e.g. saturdays and sundays not measured
        #maat.loc[maat['missing_count'] > 25, 'skip_flag'] = True

        # set the skip flag for any year where MAAT could not be calculated
        maat.loc[maat['MAAT'].isna(), 'skip_flag'] = True

        # calculate the corrected mean annual air temperature
        # by calculating the monthly mean temperature multiplied by the nominal day count for each month
        # summing these over the year, and dividing by the total number of days in the year.
        maat_corr = pd.DataFrame(maat.loc[:, ['T_{0}'.format(x) for x in self.months.values()]].values *
                                 nc.loc[:, [x for x in self.months.values()]].values,
                                 index=maat.index).sum(axis=1) / nc.ndays

        # Rework the output
        maat_corr.name = 'MAATcor'
        maat_orig = maat['MAAT']
        maat_orig.name = 'MAAT'

        # drop the monthly mean temperatures from the dataframe
        maat = maat.drop(['T_{0}'.format(x) for x in self.months.values()], axis=1)
        
        # include the corrected MAAT values as a separate column
        maat = pd.concat([maat_orig, maat_corr, maat[maat.columns[1:]]], axis=1)
        
        # Mask MAATcor where MAAT could not be calculated
        maat.loc[maat['MAAT'].isna(), 'MAATcor'] = np.nan  

        return maat

    def merge_with(self, other, inplace=True):
        """Merge this timeseries with other time series, return new combined timeseries."""
        new_cd = super().merge_with(other, inplace=inplace)
        try:
            new_cd.aggregate_daily()
        except KeyError:
            # One of the necessary parameters were not present ['AT', 'ATmax' or 'ATmin']
            pass
        return new_cd

    def get_scale_bias_to(self, other):
        """Return scale and bias of this timeseries relative to other timeseries.
        returns [scale, bias, r_sq] such that other.AT = scale * self.AT + bias
        """
        ids = self.daily_data.dropna(subset=['AT']).index.intersection(other.daily_data.dropna(subset=['AT']).index)

        x = self.daily_data.loc[ids, 'AT']
        y = other.daily_data.loc[ids, 'AT']
        
        p, cov = np.polyfit(x, y, 1, cov=True)

        func = lambda x: np.polyval(p,x)

        r_sq = 1 - (sum((y - func(x))**2) / ((len(y) - 1) * np.var(y, ddof=1)))

        return p[0], p[1], r_sq

    def scale_debias_to(self, other):
        """Scale and bias correct this timeseries to other timeseries.
        return new combined timeseries.
        """
        if isinstance(other, type(self)):
            new = copy.deepcopy(self)
        else:
            raise ValueError('Unable to scale instances of {0} to {1}'.format(type(self), type(other)))

        # ids = self.daily_data.dropna(subset=['AT']).index.intersection(other.daily_data.dropna(subset=['AT']).index)
        
        # x = self.daily_data.loc[ids, 'AT']
        # y = other.daily_data.loc[ids, 'AT']
        
        # p, cov = np.polyfit(x, y, 1, cov=True)

        scale, bias, r_sq = self.get_scale_bias_to(other)

        def a2b(x):
            """Return a 1D polynomial."""
            return np.polyval([scale, bias], x)

        new.daily_data['AT'] = a2b(new.daily_data['AT'])
        
        return new

		
class DegreeDays(ClimateTimeSeriesBase):
    identifier = 'DD'
    unit = 'celsius*day'    # units as defined by the Pint package
    
    def __init__(self, AT=None):
        # Takes AirTemp instance or air temperature time series (pandas Series)
        # as input argument
        #
        
        # store names of attributes that should be saved to disk when storing instance.
        self.register_attrs(['daily_data_threshold', 'dd_data_threshold'])

        # Data threshold for daily averaging
        self.daily_data_threshold = 6  # measurements per day

        # Data threshold for annual FDD/TDD calculation
        self.dd_data_threshold = 340  # measurements per year
        
        self.degree_days = None

        if not isinstance(AT, AirTemp):
            self.AT = AirTemp(AT)
        else:
            self.AT = AT

        self.calc_degree_days()

    def calc_degree_days(self):
        """Calculates the number of freezing and thawing degree days per year in the
        dataframe passed.

        The algorithm does not account for missing days in the time-series.
        Incomplete timeseries data will result in biased FDD and TDD values.

        """
        # Exclude days with too few measurements
        temp = self.AT.daily_data.drop('count', axis=1)
        
        # add ymd information
        temp = pd.concat([temp, self.get_ymd(temp)], axis=1)

        # Remove data where daily data count is below threshold
        # But keep days that were averaged from max and min temperatures (count = -1)
        temp.loc[(self.AT.daily_data['count'] >= 0) & (self.AT.daily_data['count'] < self.daily_data_threshold), 'AT'] = np.nan

        # flag any incomplete days, so they will not be included in the statistics for annual averages
        temp.year = np.where(np.isnan(temp['AT']), temp['AT'], temp.year)
        temp.month = np.where(np.isnan(temp['AT']), temp['AT'], temp.month)
        temp.weekday = np.where(np.isnan(temp['AT']), temp['AT'], temp.weekday)

        # Calculate freezing degree days
        fdd = temp.loc[temp['AT'] < 0, ['AT', 'year']].groupby('year').agg(fdd=('AT', "sum"), fdd_days=('AT', "size"))

        # Calculate thawing degree days
        tdd = temp.loc[temp['AT'] >= 0, ['AT', 'year']].groupby('year').agg(tdd=('AT', "sum"), tdd_days=('AT', "size"))

        # Calculate total number of days included in tdd and fdd
        dsum = tdd.tdd_days + fdd.fdd_days
        dsum = dsum.to_frame(name='total_days')

        # stack up results
        dd = pd.concat([tdd, fdd, dsum], axis=1)
        dd = dd.sort_index(ascending=True)

        # Reindex to get a full timeseries without gaps
        new_index = np.arange(dd.index.min() - 1, dd.index.max() + 2, 1, dtype=float)
        dd = dd.reindex(index=new_index)

        dd['skip_flag'] = False
        dd.loc[(dd['total_days'] < self.dd_data_threshold) | (dd['total_days'].isna()), 'skip_flag'] = True

        self.degree_days = dd    
        
class AccumulatedDegreeDays(DegreeDays):
    identifier = 'ADD'
    unit = 'celsius*day'    # units as defined by the Pint package

    def __init__(self, *args, **kwargs):
        self.CDD = None
        self.ADDT = None
        super().__init__(*args, **kwargs)
        self.calc_CDD()
        self.calc_ADDT()

    def calc_CDD(self):
        """Calculates the cumulative degree days of a time series.

        The algorithm does not account for missing days in the time-series.
        Incomplete timeseries data will result in biased FDD and TDD values.

        """
        # Exclude days with too few measurements
        temp = self.AT.daily_data.drop('count', axis=1)
        
        # add ymd information
        temp = pd.concat([temp, self.get_ymd(temp)], axis=1)
        
        temp.loc[(self.AT.daily_data['count'] >= 0) & (self.AT.daily_data['count'] <= self.daily_data_threshold), 'AT'] = np.nan
        
        temp['CDD'] = temp.groupby(lambda x: x.year)['AT'].cumsum()
        
        # store as instance variable
        self.CDD = temp        
        
    def calc_ADDT(self):
        """Calculate the Accumulated Degree Days of Thawing (ADDT) timeseries.
        
        ADDT is the cumulative sum of thawing degree days since the onset of
        thawing until the onset of freezing in a given year. It is often used
        to estimate thaw penetration in frozen ground.
        
        Onset of thawing is defined as the date of the spring minimum in the 
        cumulative degree days curve.
        Onset of freezing is defined as the date of the fall maximum in the 
        cumulative degree days curve.
        """
        temp = self.CDD
        
        # calculate the minimum and maximum degree days within the season and the dates they occur
        # The minimum should be in the spring, and the maximum in the summer.
        min_stats = temp.loc[temp['month'] <= 8].groupby(['year']).agg(min=('CDD', np.min), min_date=('CDD', 'idxmin'))
        max_stats = temp.loc[temp['month'] >= 5].groupby(['year']).agg(max=('CDD', np.max), max_date=('CDD', 'idxmax'))
        stats = pd.concat([min_stats, max_stats], axis=1)
        self.ADD_stats = stats.reindex(np.arange(stats.index[0], stats.index[-1]+1))
        
        # drop any years where we don't have data for both min and max
        stats = stats.dropna(axis=0)  
        
        # Filter the dataset to only contain data from years where
        # we have both minimum and maximum defined.
        temp2 = temp[temp.year.isin(stats.index)]

        # Extract data only for the thawing season
        temp3 = temp2[(temp2.index >= stats.min_date.loc[temp2.year]) & (temp2.index <= stats.max_date.loc[temp2.year])].copy()

        # calculate the Accumulated Degree Days of Thawing
        temp3['ADDT'] = temp3.groupby(['year'])['AT'].cumsum()
        
        self.ADDT = temp3 # add the dataframe as a class variable

        # calculate the maximum ADDT value for each thawing season
        maxADDT = temp3.groupby(['year'])['ADDT'].max()
        
        # Calculate the normalized ADDT value by dividing each ADDT value
        # by the max ADDT for that particular year.
        self.ADDT['NADDT'] = self.ADDT['ADDT']/maxADDT.loc[self.ADDT['year'].dropna(axis=0)].values

        # reindex the timeseries from first of january on the first year to end of december the last year
        # adding any missing dates.
        idx = pd.date_range(self.ADDT.index[0].replace(month=1, day=1), self.ADDT.index[-1].replace(month=12, day=31))
        self.ADDT = self.ADDT.reindex(idx)
        
        # use method get_ymd(self.ADDT) to obtain these values
        # # update year, month and weekday values after reindexing
        # self.ADDT.year = self.ADDT.index.year
        # self.ADDT.month = self.ADDT.index.month
        # self.ADDT.weekday = self.ADDT.index.weekday

    def fill_ADDT(self):
        """Extend the ADDT timeseries with zeros from start of the year to
        the onset of thawing, and with the maximum value from the onset of 
        freezing to the end of the year.
        """
        if not hasattr(self, 'ADDT'):
            self.calc_ADDT()
    
        # make a copy
        ADDT = self.ADDT.copy()
        
        # fill with 0 before thawing starts
        idx = (ADDT.index < self.ADD_stats.min_date.loc[ADDT.year])
        ADDT.loc[idx, 'ADDT'] = 0
        ADDT.loc[idx, 'NADDT'] = 0
        
        # fill with yearly max ADDT after thaw season ends
        idx = (ADDT.index >= self.ADD_stats.max_date.loc[ADDT.year])
        mx_datetime =  pd.to_datetime(self.ADD_stats.max_date.loc[ADDT.loc[idx].year])
        mx_value = ADDT.loc[idx, 'ADDT'][ADDT.iloc[idx].index == mx_datetime][mx_datetime]
        ADDT.loc[idx, 'ADDT'] = mx_value.values
        ADDT.loc[idx, 'NADDT'] = 1
        
        return ADDT
        
    def get_ADDT(self, t, fill=True):
        """Get the ADDT value for a given date.
        If fill=True, use the filled timeseries, which yields a value for any date.
        If fill=False, the method will yield a numerical value within the thawing season
        and an NaN value in the freezing season.
        
        t is a datetime.date or datetime.datetime object
        """
        # t must be a date or datetime.datetime object
        if fill:
            ADDT = self.fill_ADDT()
        else:
            if not hasattr(self, 'ADDT'):
                self.calc_ADDT()
            ADDT = self.ADDT
        
        # make sure we have a date time object
        t = dt.datetime(t.year, t.month, t.day)
        
        # extract information from the given date
        tmp = ADDT[ADDT.index==t].T
        
        dct = tmp[t].to_dict()
        
        return tmp[t].to_dict()   #orient='list')
	

class Wind(ClimateTimeSeriesBase):
    identifier = ['WS', 'WD']
    unit = ['m/s', 'degree']    # units as defined by the Pint package
    daily_aggregator = ['mean', 'mean']
    pass


class Precipitation(ClimateTimeSeriesBase):
    identifier = 'PRE'
    unit = 'mm'    # units as defined by the Pint package
    daily_aggregator = 'sum'
    pass


class RelativeHumidity(ClimateTimeSeriesBase):
    identifier = 'RH'
    unit = 'percent'    # 'percent' is not in Pint package, needs custom declaration, see https://stackoverflow.com/a/39154101
    daily_aggregator = 'mean'
    pass


class SnowDepth(ClimateTimeSeriesBase):
    identifier = 'SD'
    unit = 'm'    # units as defined by the Pint package
    daily_aggregator = 'max'
    pass


class AirPressure(ClimateTimeSeriesBase):
    identifier = 'AP'
    unit = 'hPa'    # units as defined by the Pint package
    daily_aggregator = 'mean'
    pass


# =====================================================================================
# CLIMATE DATASET CLASS DEFINITIONS
# =====================================================================================


class ClimateDataSetBase:
    """Base class for climate data, implementing all functionality
    common to all climate data.
    Intended for subclassing, to implement reading and processing
    methods specific to a certain format of input data.
    """
    
    dstype = 'unknown'
    description = 'BaseClass for climate data type'
    
    days = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
    months = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

    _attrs = ['type', 'description']

    def __init__(self, file=None, filepath=None, encoding=None, read_data=True):
        # Data aggregation criteria:
        #
        # If number of measurements per day is less than 6, use average of min and max temperature
        #      Do not count the other measurements from those days in aggregation
        #
        # If more than 6 measurements per day, use actual measurements in daily average
        #
        # For annual averages, filter out days with less than 20 measurements, assuming 24 measurements per day as nominal frequency

        # store names of attributes that should be saved to disk when storing instance.
        self.register_attrs(['data_freq', 'daily_data_threshold', 'annual_data_threshold', 
                             'dd_data_threshold', 'minmax_data_threshold', 'station_name',
                             'converted', 'filename', 'repository'])

        # Nominal data frequency
        self.data_freq = 24  # measurements per day

        # Data threshold for daily averaging
        self.daily_data_threshold = 6  # measurements per day

        # Data threshold for annual averaging
        self.annual_data_threshold = 340  # measurements per year
        # Set this to 0 to include all years that have data

        # Data threshold for annual FDD/TDD calculation
        self.dd_data_threshold = 340  # measurements per year

        # Use min/max average when less than ndat_minmax measurements per day
        self.minmax_data_threshold = 6

        try:
            _ = self.station_name
        except:
            self.station_name = ""

        self.raw_data = None
        self.converted = None
        self.datasets = {}

        if (not isinstance(file, DataFile)) & (file is not None):
            if filepath is not None:
                self.repository = DataRepository(filepath)
                file = self.repository.get_file(file)
            else:
                file = DataFile(file)
        
        if isinstance(file, DataFile):
            self.filename = file._filename
            self.encoding = file.encoding
        
        if (file is not None) and (read_data is True):
            self.read_data(file, self.encoding)

    def register_attrs(self, attrs=[]):
        self._attrs = self._attrs
        self._attrs.extend(attrs)
        # Now make sure only unique values exist
        #if len(attrs)>0:
        #    used = set()
        #    self._attrs = [x for x in self._attrs if x not in used and (used.add(x) or True)]

    def to_xarray(self):
        import xarray as xr
        ds = xr.Dataset()
        for k in self._attrs:
            try:
                v = getattr(self, k)
            except:
                continue
            if v is not None:
                ds.attrs[k] = v
            else:
                ds.attrs[k] = 'None'
        ds.attrs['datasets'] = list(self.datasets.keys())
        ds.attrs['__class__'] = '{0}.{1}'.format(self.__class__.__module__, self.__class__.__name__)
        return ds

    def to_netcdf(self, filename):
        ds = self.to_xarray()
        ds.to_netcdf(filename, mode='w', group='metadata')
        for k, thisds in self.datasets.items():
            ds = thisds.to_xarray()
            ds.to_netcdf(filename, mode='a', group='datasets/{0}'.format(k))

    @classmethod
    def from_netcdf(cls, filename):
        import xarray as xr
        import pydoc
        ds = xr.open_dataset(filename, group='metadata')
        
        # trick from here: https://stackoverflow.com/a/55968374/1760389
        #classname = ds.attrs.pop('__class__')
        #thistype = pydoc.locate(classname)
        __class__ = ds.attrs.pop('__class__')
        toks = __class__.split('.')
        module_name = '.'.join(toks[0:-1])
        class_name = toks[-1]
        module = __import__(module_name)
        class_ = getattr(module, class_name)
        new = class_(read_data=False)

        datasets = ds.attrs.pop('datasets')
        if isinstance(datasets, str):
            datasets = [datasets]

        for k, v in ds.attrs.items():
            if v == 'None':
                setattr(new, k, None)
            else:
                setattr(new, k, v)

        for dsname in datasets:
            tmpds = xr.open_dataset(filename, group='datasets/{0}'.format(dsname))
        
            # trick from here: https://stackoverflow.com/a/55968374/1760389
            #thistype = pydoc.locate(tmpds.attrs['__class__'])
            toks = tmpds.attrs['__class__'].split('.')
            module_name = '.'.join(toks[0:-1])
            class_name = toks[-1]
            module = __import__(module_name)
            class_ = getattr(module, class_name)
            
            new.datasets[dsname] = class_.from_xarray(tmpds)

        return new

    def read_data(self, file, encoding=None):
        """Abstract method for reading data from file. Should be implemented in subclass.
        Must accept input argument 'file', which can be a string path, a pathlib.Path instance or
        a DataFile instance.
        """
        raise NotImplementedError()

    def is_valid(self):
        """Abstract method for validating data from file. Should be implemented in subclass."""
        raise NotImplementedError()
        
    def merge_with(self, other, inplace=True, warn=True):
        """Merge this Climate Data collection with other Climate Data collection,
        by looping over individual datasets and merging them individually."
        
        Returns a new ClimateSeries instance."""

        if inplace:
            new = self
        else:
            new = copy.copy(self)   
        
        # should NEW be recast as ClimateDataBase, or e.g. a new class ClimateDataCombined?
        # ClimateDataCombined could inherit from ClimateDataBase.

        keys = set([*self.datasets.keys(), *other.datasets.keys()])
        for k in keys:
            if (k in self.datasets) and (k in other.datasets) and (self.datasets[k] is not None) and (other.datasets[k] is not None):
                try:
                    new.datasets[k] = self.datasets[k].merge_with(other.datasets[k])
                except NotCompatibleError as err:
                    print(err)
            elif (k in self.datasets) and (self.datasets[k] is not None):
                new.datasets[k] = self.datasets[k]
                if warn:
                    print('Dataset {0} only present in first dataset. Copied, no merge possible.'.format(k))
            elif (k in other.datasets) and (other.datasets[k] is not None):
                new.datasets[k] = other.datasets[k]
                if warn:
                    print('Dataset {0} only present in second dataset. Copied, no merge possible.'.format(k))
            else:
                new.datasets[k] = None
                if warn:
                    print('Dataset {0} is not present in any of the datasets.'.format(k))
        
        return new

    def scale_debias_to(self, other, warn=True):
        """ """
        new = copy.copy(self)   
        
        # should NEW be recast as ClimateDataSetBase, or e.g. a new class ClimateDataSetCombined?
        # ClimateDataCombined could inherit from ClimateDataSetBase.

        keys = set([*self.datasets.keys(), *other.datasets.keys()])
        for k in keys:
            if k in self.datasets and k in other.datasets:
                try:
                    new.datasets[k] = self.datasets[k].scale_debias_to(other.datasets[k])
                except AttributeError as err:
                    # so we didn't have a scale_debias_to method...
                    new.datasets[k] = None
                except NotImplementedError as err:
                    # scale_debias_to raised NotImplementedError...
                    new.datasets[k] = None
                except NotCompatibleError as err:
                    print(err)
                    new.datasets[k] = None
            elif k in self.datasets:
                new.datasets[k] = None
                if warn:
                    print('Dataset {0} only present in first dataset. No scale/debias possible.'.format(k))
            elif k in other.datasets:
                new.datasets[k] = None
                if warn:
                    print('Dataset {0} only present in second dataset. No scale/debias possible.'.format(k))
        
        return new

    def copy_missing_from(self, other, keys=None, inplace=False):
        """Copy missing datasets from other.
        Takes keyword argument 'keys' as a list of dataset identifiers to copy.
        If 'keys' is empty (default) any missing dataset will be copied.
        
        Returns a new ClimateSeries instance."""

        if inplace:
            new = self
        else:
            new = copy.copy(self)   
        
        if keys is None:
            keys = set(*other.datasets.keys())

        for k in keys:
            if (k not in self.datasets) or (self.datasets[k] is not None):
                if (k in other.datasets):
                    new.datasets[k] = other.datasets[k]
        
        return new

    def aggregate_daily(self):
        """Aggregate raw data into daily values, using the aggregation method
        specified for each timeseries class.
        Aggregation will be applied to each timeseries in the dataset
        """
        for k in self.datasets.keys():
            if (k in self.datasets) and (self.datasets[k] is not None):
                self.datasets[k].aggregate_daily()
        
    def limit(self, lim:list[Union[str,dt.datetime,dt.date, None]], inplace:bool=False):
        """Limit all timeseries in dataset to the specified start and end time.
        Input:
            List of start and end times, may be string, or datetime or date types.
        Return:
            New ClimateDataSet with the time series limited to the specified range
        """
        if inplace:
            new = self
        else:
            new = copy.deepcopy(self) 

        if lim is not None:
            for k in self.datasets.keys():
                if (k in self.datasets) and (self.datasets[k] is not None):
                    new.datasets[k].limit(lim)
        
        return new


class DMIType1(ClimateDataSetBase):
    dstype = 'DMI1'
    description = 'Old DMI data type used for data provided until 2013'

    def __init__(self, file, filepath=None, encoding=None, **kwargs):
        if (not isinstance(file, DataFile)) & (file is not None):
            # File is a string or Path instance
            file = pathlib.Path(file)
            if len(file.parts) == 1:
                # file contains no path
                if filepath is None:
                    # if path is not specified, use default
                    filepath = DMI_PATH_OLD_FORMAT
            else:
                # file contains a path, use it...
                filepath = file.parent
                file = file.name
        
        super().__init__(file, filepath=filepath, encoding=encoding, **kwargs)

    def read_data(self, file, encoding=None):

        # Read time series and process date columns
        raw_data = pd.read_csv(file, sep='\t', dtype=float, na_values=[' ', '\t'],
                               comment='#', encoding=encoding)

        # 2019-1208:  Floating point version of date conversion broken - changed to integer version. Not sure if the datetime functionality changed?
        # dates = pd.to_datetime(raw_data['year']*1000000+raw_data['month']*10000+raw_data['day']*100+raw_data['hour'], format='%Y%m%d%H.%f')
        dates = pd.to_datetime(
            raw_data['year'] * 1000000 + raw_data['month'] * 10000 + raw_data['day'] * 100 + raw_data['hour'],
            format='%Y%m%d%H')

        # Index the dataframe based on datetimes
        # dates = pd.to_datetime(dates)
        raw_data.index = pd.DatetimeIndex(dates)

        # To add time-zone information use:
        # raw_data.tz_localize('utc')                    # to get UTC+00:00
        # or
        # raw_data.tz_localize(tz.FixedOffset(-2*60))    # to get UTC-02:00
        # both will convert to time-zone aware datetimes, leaving the time stamp as it is, just adding the tz information.

        # DateTimes are here stored timezone-unaware, but are UTC times

        converted = pd.DataFrame()

        # adjust temperatures by factor of 10
        converted['ttt'] = raw_data['ttt'] / 10                  # Dry bulb air temperature [degC]
        converted['txtxtx'] = raw_data['txtxtx'] / 10            # Absolute maximum temperature [degC]
        converted['tntntn'] = raw_data['tntntn'] / 10            # Absolute minimum temperature [degC]

        converted['pppp'] = raw_data['pppp'] / 10                # Air pressure at mean sea level [hPa]
        converted['rh'] = raw_data['rh']                         # Relative humidity [%]

        converted['dd'] = raw_data['dd']                         # Mean wind direction, no conversion needed  [degrees]
        converted['ff'] = raw_data['ff'] / 10                    # Mean wind speed [m/s]
        
        converted['rrr6'] = raw_data['rrr6'] / 10                 # Accumulated precipitation [mm]
        converted['sss'] = raw_data['sss']                       # Snow depth [cm]

        self.raw_data = raw_data
        self.converted = converted

        # CURRENTLY NOT USING THE MAX MIN INFORMATION
        #self.datasets['AT'] = AirTemp(AT=converted['ttt'], ATmax=converted['txtxtx'], ATmin=converted['tntntn'])
        
        # Now assign datasets where data is present
        if not converted['ttt'].isna().all():
            self.datasets['AT'] = AirTemp(AT=converted['ttt'],
                                          ATmin=converted['tntntn'],
                                          ATmax=converted['txtxtx'])
        
        if not converted['rh'].isna().all():
            self.datasets['RH'] = RelativeHumidity(RH=converted['rh'].dropna())

        if not converted['pppp'].isna().all():
            self.datasets['AP'] = AirPressure(AP=converted['pppp'].dropna())
        
        if not (converted['ff'].isna().all() & converted['dd'].isna().all()):
            self.datasets['WIND'] = Wind(WS=converted['ff'],
                                         WD=converted['dd'])
        
        if not converted['rrr6'].isna().all():
            self.datasets['PRE'] = Precipitation(PRE=converted['rrr6'].dropna())
        
        if not converted['sss'].isna().all():
            self.datasets['SD'] = SnowDepth(SD=converted['sss'].dropna())

    def is_valid(self):
        if self.raw_data.ttt.count() == 0 and \
                (self.raw_data.txtxtx.count() == 0 or self.raw_data.tntntn.count() == 0):
            return False
        else:
            return True

    def extract_AT_min_max(self):
        """Extract min and max temperatures based on specific
        times of day, on days where the daily data threshold is not met,
        and where min/max values are not aready available."""

        counts = self.datasets['AT'].raw_data.groupby(self.datasets['AT'].raw_data.index.date).count()

        if ('ATmin' in counts.columns) & ('ATmax' in counts.columns):
            # We already have some ATmin/ATmax values

            # PSEUDO CODE
            # where AT.count < data threshold
            #       and ATmin.count = 0
            #       and ATmax.count = 0
            #    take 00:00:00 as ATmin (if it exists)
            #    take 12:00:00 as ATmax (if it exists)

            idx = (counts['AT']<self.datasets['AT'].daily_data_threshold) & (counts['ATmin']==0) & (counts['ATmax']==0).values
            
            ds_idx = self.datasets['AT'].raw_data.index.floor('d').isin(counts[idx].index)

            rd = self.datasets['AT'].raw_data
            rd['ATmin'].iloc[ds_idx & (rd.index.hour==0) & (rd.index.minute==0)] = rd[ds_idx & (rd.index.hour==0) & (rd.index.minute==0)]['AT']
            rd['ATmax'].iloc[ds_idx & (rd.index.hour==12) & (rd.index.minute==0)] = rd[ds_idx & (rd.index.hour==12) & (rd.index.minute==0)]['AT']

        else:
            # We have no ATmin/ATmax values, and must create the columns
            pass

    def __calc_daily_averages(self):

        # METHOD IS KEPT HERE FOR DOCUMENTATION PURPOSES.
        # some old data sets have only max and min temperatures measured
        # these are currently not handled.
        # But this is the way to handle them when needed.
        #
        # We need to implement a way to use this information in the AirTemp
        # class.

        grouped = self.raw_data.groupby(self.raw_data.index.date)

        # Use either min/max average or actual data value averages, depending on daily data count
        AT_mean_values = np.where(grouped['ttt'].count() <= self.minmax_data_threshold,
                                  (grouped['txtxtx'].mean() + grouped['tntntn'].mean()) / 2,
                                  grouped['ttt'].mean())

        # For days where min/max average was used, set count to -1
        AT_mean_count = np.where(grouped['ttt'].count() <= self.minmax_data_threshold,
                                 -np.ones(len(AT_mean_values)),
                                 grouped['ttt'].count())

        # Set count to 0 on days where min/max calculation is unsuccessful
        AT_mean_count = np.where(np.logical_and(AT_mean_count == -1, np.isnan(AT_mean_values)),
                                 np.zeros(len(AT_mean_count)),
                                 AT_mean_count)

        # Create new dataframe with daily averages
        AT_mean = pd.DataFrame(AT_mean_values, index=np.unique(self.raw_data.index.date))
        AT_mean.columns = ['AT']
        AT_mean['count'] = AT_mean_count

        self.AT_mean = AT_mean


class DMIType2(ClimateDataSetBase):
    # This is the new DMI data format starting from 2014
    dstype = 'DMI2'
    description = 'New DMI data type used for data provided after 2014'
    decimal_point = ','

    def __init__(self, file=None, filepath=None, encoding=None, **kwargs):
        # store names of attributes that should be saved to disk when storing instance.
        self.register_attrs(['decimal_point'])

        if (not isinstance(file, DataFile)) & (file is not None):
            # File is a string or Path instance
            file = pathlib.Path(file)
            if len(file.parts) == 1:
                # file contains no path
                if filepath is None:
                    # if path is not specified, use default
                    filepath = DMI_PATH_NEW_FORMAT
            else:
                # file contains a path, use it...
                filepath = file.parent
                file = file.name
        
        super().__init__(file, filepath=filepath, encoding=encoding, **kwargs)

    def read_data(self, file, encoding=None):
        raw_data = pd.read_csv(file, sep=';', dtype=float, na_values=[' ', '\t'], 
                               decimal=self.decimal_point, comment='#', encoding=encoding)

        if len(raw_data.columns[1]) == 2:
            cols = raw_data.columns.values
            cols[1] = 'year'
            cols[2] = 'month'
            cols[3] = 'day'
            cols[4] = 'hour(utc)'
            raw_data.columns = cols

        # 2019-1208:  Floating point version of date conversion broken - changed to integer version. Not sure if the datetime functionality changed?
        # dates = pd.to_datetime(raw_data['year']*1000000+raw_data['month']*10000+raw_data['day']*100+raw_data['hour(utc)'], format='%Y%m%d%H.%f')
        dates = pd.to_datetime(raw_data['year'] * 1000000 + raw_data['month'] * 10000 + raw_data['day'] * 100 + raw_data['hour(utc)'], format='%Y%m%d%H')

        # Index the dataframe based on datetimes
        raw_data.index = pd.DatetimeIndex(dates)

        # To add time-zone information use:
        # raw_data.tz_localize('utc')                    # to get UTC+00:00
        # or
        # raw_data.tz_localize(tz.FixedOffset(-2*60))    # to get UTC-02:00
        # both will convert to time-zone aware datetimes, leaving the time stamp as it is, just adding the tz information.

        self.raw_data = raw_data
        
        # Post processing
        precip = self._get_accumulated_precip()
        winddir = self._get_wind_directions()

        # Now assign datasets where data is present
        if not self.raw_data['101'].isna().all():
            self.datasets['AT'] = AirTemp(AT=raw_data['101'])
        
        if not self.raw_data['201'].isna().all():
            self.datasets['RH'] = RelativeHumidity(RH=raw_data['201'])

        if not self.raw_data['401'].isna().all():
            self.datasets['AP'] = AirPressure(AP=raw_data['401'])
        
        if not (self.raw_data['301'].isna().all() & winddir.isna().all()):
            self.datasets['WIND'] = Wind(WS=raw_data['301'],
                                         WD=winddir)
        
        if not precip.isna().all():
            self.datasets['PRE'] = Precipitation(PRE=precip)

        # New data format does not contain snow depths
        

    def _get_accumulated_precip(self):
        """Return a timeseries of raw accumulated precipitations.
        Stations use different reporting strategies, and they may 
        change over time.
        
        1) hourly accumulated precipitation  (601)
        2) 12-hour accumulated precipitation (603)
        3) 24-hour accumulated precipitation (609)
        
        This function will return a combined timeseries, with the
        highest frequency of measurements available any given date.
        """

        precip = self.raw_data['601'].copy()  # take all data from 365 column

        for k in ['603', '609']:
            # loop over additional columns to patch dataseries with
            # lower frequency measurements where missing

            if not k in self.raw_data:
                continue

            # find dates where dataset is all NaNs
            ndat_per_day = precip.notnull().astype(int).groupby(precip.index.date).sum()
            dates = ndat_per_day[ndat_per_day==0]

            # find indices of these dates
            dates_idx = precip.index.normalize().isin(dates)

            if len(dates_idx) > 0:
                # patch with data from other column, where missing.
                precip[dates_idx] = self.raw_data[k][dates_idx]

        precip.name = 'PRE'  # set series name
        return precip

    def _get_wind_directions(self):
        """Return a timeseries of measured wind directions.
        Stations use different reporting strategies, and they may 
        change over time.
        
        1) Mean wind direction last 10 minutes  (365)
        2) Mean wind direction last hour  (371)
                
        This function will return a combined timeseries, with
        10-minute averages where available, and otherwise
        hourly averages.
        """

        if not '365' in self.raw_data:
            return None

        wdir = self.raw_data['365'].copy()  # take all data from 365 column

        for k in ['371']:
            # loop over additional columns to patch dataseries with
            # lower frequency measurements where missing

            # find dates where dataset is all NaNs
            ndat_per_day = wdir.notnull().astype(int).groupby(wdir.index.date).sum()
            dates = ndat_per_day[ndat_per_day==0]

            # find indices of these dates
            dates_idx = wdir.index.normalize().isin(dates)

            if len(dates_idx) > 0:
                # patch with data from other column, where missing.
                wdir[dates_idx] = self.raw_data[k][dates_idx]

        wdir.name = 'WD'  # set series name
        return wdir


    def is_valid(self):
        if self.raw_data['101'].count() == 0:
            return False
        else:
            return True


class DMIType3(DMIType2):
    # This is the new DMI data format starting from 2022
    # It includes all data (except snowdepth) from all stations, any data that could previously
    # be obtained through combination of DMIType1 and DMIType2 datasets.
    # It should be identical to DMIType2 except the decimal point changed from comma to point.
    dstype = 'DMI3'
    description = 'New DMI data type introduced 2022, containing all historic data'
    decimal_point = '.'

    def __init__(self, file=None, filepath=None, encoding=None, **kwargs):
        # store names of attributes that should be saved to disk when storing instance.
        self.register_attrs(['decimal_point'])

        if (not isinstance(file, DataFile)) & (file is not None):
            # File is a string or Path instance
            file = pathlib.Path(file)
            self.station_name = file.stem
            if len(file.parts) == 1:
                # file contains no path
                if filepath is None:
                    # if path is not specified, use default
                    filepath = DMI_PATH_UNIFIED_FORMAT
            else:
                # file contains a path, use it...
                filepath = file.parent
                file = file.name
        elif file is not None:
            self.station_name = file._filename.stem
        else: 
            self.station_name = ''

        super().__init__(file, filepath=filepath, encoding=encoding, **kwargs)

    def read_data(self, file, encoding=None):
        raw_data = pd.read_csv(file, sep=';', dtype=float, na_values=[' ', '\t'], 
                               decimal=self.decimal_point, comment='#', encoding=encoding)

        raw_data.columns = [cname.lower() for cname in raw_data.columns.values]

        if len(raw_data.columns[1]) == 2:
            cols = raw_data.columns.values
            cols[1] = 'year'
            cols[2] = 'month'
            cols[3] = 'day'
            cols[4] = 'hour(utc)'
            raw_data.columns = cols

        # 2019-1208:  Floating point version of date conversion broken - changed to integer version. Not sure if the datetime functionality changed?
        # dates = pd.to_datetime(raw_data['year']*1000000+raw_data['month']*10000+raw_data['day']*100+raw_data['hour(utc)'], format='%Y%m%d%H.%f')
        if 'hour(utc)' in raw_data.columns: 
            dates = pd.to_datetime(raw_data['year'] * 1000000 + raw_data['month'] * 10000 + raw_data['day'] * 100 + raw_data['hour(utc)'], format='%Y%m%d%H')
        else:
            dates = pd.to_datetime(raw_data['year'] * 1000000 + raw_data['month'] * 10000 + raw_data['day'] * 100, format='%Y%m%d%H')

        # Index the dataframe based on datetimes
        raw_data.index = pd.DatetimeIndex(dates)

        # To add time-zone information use:
        # raw_data.tz_localize('utc')                    # to get UTC+00:00
        # or
        # raw_data.tz_localize(tz.FixedOffset(-2*60))    # to get UTC-02:00
        # both will convert to time-zone aware datetimes, leaving the time stamp as it is, just adding the tz information.

        self.raw_data = raw_data
        
        # Post processing
        precip = self._get_accumulated_precip()
        winddir = self._get_wind_directions()

        # Now assign datasets where data is present
        if ('101' in self.raw_data) and (not self.raw_data['101'].isna().all()):
            self.datasets['AT'] = AirTemp(AT=raw_data['101'],
                                          ATmin=raw_data['123'],
                                          ATmax=raw_data['113'])
        
        if ('201' in self.raw_data) and (not self.raw_data['201'].isna().all()):
            self.datasets['RH'] = RelativeHumidity(RH=raw_data['201'])

        if ('401' in self.raw_data) and (not self.raw_data['401'].isna().all()):
            self.datasets['AP'] = AirPressure(AP=raw_data['401'])
        
        if ('301' in self.raw_data) and (not (self.raw_data['301'].isna().all() & winddir.isna().all())):
            self.datasets['WIND'] = Wind(WS=raw_data['301'],
                                         WD=winddir)
        
        if not precip.isna().all():
            self.datasets['PRE'] = Precipitation(PRE=precip)

        # New data format does not contain snow depths

class DMITypeX(DMIType2):
    dstype = 'DMIX'
    description = 'Not sure what filetype this is, but quite similar to DMIType2'

    def __init__(self, file, **kwargs):
        super().__init__(file, **kwargs)

    def read_data(self, file, encoding=None):

        raw_data = pd.read_csv(file, sep=';', dtype=float, na_values=[' ','\t'], parse_dates=[1], decimal=',',
                               comment = '#', encoding=encoding)
        raw_data = raw_data.set_index('date(utc)')

        # To add time-zone information use:
        # raw_data.tz_localize('utc')                    # to get UTC+00:00
        # or
        # raw_data.tz_localize(tz.FixedOffset(-2*60))    # to get UTC-02:00
        # both will convert to time-zone aware datetimes, leaving the time stamp as it is, just adding the tz information.

        raw_data['weekday'] = raw_data.index.weekday  # added to be used for calculating statistics

        self.raw_data = raw_data

        self.datasets['AT'] = AirTemp(AT=raw_data['101'])
        #self.datasets['RH'] = RelativeHumidity(RH=raw_data['XXX'])
        #self.datasets['WIND'] = Wind(WS=raw_data['XXX'], WD=raw_data('XXX'))
        #self.datasets['PRE'] = Precipitation(PRE=raw_data['XXX'])
        #self.datasets['SD'] = SnowDepth(SD=raw_data['XXX'])


class DMITypeLongDaily(ClimateDataSetBase):
    dstype = 'DMILongDaily'
    description = 'Data format combined long daily climate dataseries from DMI'

    def __init__(self, station_name, data_path=None):
        super().__init__()

        if data_path is None:
            data_repository = DataRepository(DMI_PATH_LONG_FORMAT)
        else:
            data_repository = DataRepository(data_path)

        self.station_name = station_name

        # check if station exists
        filepattern = 'gr_daily_{0}_*.csv'.format(station_name)
        flist = data_repository.glob(filepattern)
        if len(flist)==0:
            raise ValueError('Station name "{0}" was not found in repository'.format(station_name))

        # Each station consists of up to four files (but not necessarily all of them),
        # _112  Highest temperature
        # _122  Lowest temperature
        # _401  Atmospheric pressure
        # _601  Accuumulated precipitation

        # read them, and instantiate each series as ClimateSeriesBase.
        name_112 = 'gr_daily_{0}_112.csv'.format(station_name)
        if data_repository.exists(name_112):
            file = data_repository.get_file(name_112)
            self.read_112_data(file, encoding=file.encoding)

        name_122 = 'gr_daily_{0}_122.csv'.format(station_name)
        if data_repository.exists(name_122):
            file = data_repository.get_file(name_122)
            self.read_122_data(file, encoding=file.encoding)

        name_601 = 'gr_daily_{0}_601.csv'.format(station_name)
        if data_repository.exists(name_601):
            file = data_repository.get_file(name_601)
            self.read_601_data(file, encoding=file.encoding)
        
        # name_401 = 'gr_daily_{0}_401.csv'.format(station_name)
        # if data_repository.exists(name_401):
        #     file = data_repository.get_file(name_401)
        #     self.read_401_data(file)


    def read_raw_data(self, file, encoding=None):
        # Read time series and process date columns
        raw_data = pd.read_csv(file, sep=';', na_values=['null', '-999,9'], decimal=',', encoding=encoding, comment='#')

        dates = pd.to_datetime(raw_data['year'] * 1000000 + raw_data['month'] * 10000 + raw_data['day'] * 100 + raw_data['hour'], format='%Y%m%d%H')

        # Index the dataframe based on datetimes
        raw_data.index = pd.DatetimeIndex(dates)
        raw_data = raw_data[raw_data.index.notnull()]
        return raw_data

    def read_112_data(self, file, encoding=None):
        # Read time series and process date columns
        raw_data = self.read_raw_data(file, encoding)
        
        ts = AirTemp(ATmax=raw_data['elem_val'])
        if 'AT' in self.datasets:
            self.datasets['AT'] = self.datasets['AT'].merge_with(ts)
        else:
            self.datasets['AT'] = ts

    def read_122_data(self, file, encoding=None):
        # Read time series and process date columns
        raw_data = self.read_raw_data(file, encoding)

        ts = AirTemp(ATmin=raw_data['elem_val'])
        if 'AT' in self.datasets:
            self.datasets['AT'] = self.datasets['AT'].merge_with(ts)
        else:
            self.datasets['AT'] = ts

    def read_601_data(self, file, encoding=None):
        # Read time series and process date columns
        raw_data = self.read_raw_data(file, encoding)
        
        ts = Precipitation(PRE=raw_data['elem_val'])
        if 'PRE' in self.datasets:
            self.datasets['PRE'] = self.datasets['PRE'].merge_with(ts)
        else:
            self.datasets['PRE'] = ts


class DMITypeLongMonthly(ClimateDataSetBase):
    dstype = 'DMITypeLongMonthly'
    description = 'Data format combined long air temperature dataseries from DMI'

    def __init__(self, station_name='99999', **kwargs):
        # raise NotImplementedError()

        # Each station consists of three files,
        # _112  Highest temperature
        # _122  Lowest temperature
        # _601  Accuumulated precipitation

        # read them, and instantiate each series as ClimateSeriesBase.

        # station names can be like 99999, 34221, 04221 or 4221
        # parse old and new station names
        if len(station_name) == 5 and station_name[0] == '0':
            self.station_name = station_name[1:]
        else:
            self.station_name = station_name

        super().__init__('gr_monthly_all_1784_2020.csv', filepath=DMI_PATH_LONG_FORMAT, encoding=None, **kwargs)


    def read_data(self, file, encoding=None):
        # Read time series and process date columns

        month_translation = {'jan': 1,
                             'feb': 2,
                             'mar': 3,
                             'apr': 4,
                             'may': 5,
                             'jun': 6,
                             'jul': 7,
                             'aug': 8,
                             'sep': 9,
                             'oct': 10,
                             'nov': 11,
                             'dec': 12}

        months = list(month_translation.keys())
        month_nos = list(month_translation.values())

        elem_no_translation = {101: 'AT',
                               111: 'ATmaxAvg', 
                               112: 'ATmax', 
                               121: 'ATminAvg', 
                               122: 'ATmin', 
                               401: 'AP', 
                               601: 'PRE',
                               602: 'PRE_max_24h', 
                               701: 'days with snow', 
                               801: 'cloud cover'}
                               
        # From the dmi report:
        # 101 Average air temperature                    average     °C
        # 111 Average of daily maximum air temperature   average     °C
        # 112 Highest air temperature                    max         °C
        # 121 Average of daily minimum air temperature   average     °C
        # 122 Lowest air temperature                     min         °C
        # 401 Atmospheric pressure (msl)                 obs/average hPa
        # 601 Accumulated precipitation                  sum         mm
        # 602 Highest 24-hour precipitation              max         mm
        # 701 Number of days with snow cover (> 50 % covered) sum    days
        # 801 Average cloud cover                        average     %

        raw_data = pd.read_csv(file, sep=';', na_values=['null', '-999,9'], decimal=',', encoding=encoding, comment='#')

        df = pd.melt(raw_data, id_vars=['stat_no', 'elem_no', 'year'], value_vars=months)
        
        df = df.replace({'variable': month_translation})
        df.rename(columns={'variable': 'month'}, inplace=True)

        df = df.replace({'elem_no': elem_no_translation})
        df.rename(columns={'elem_no': 'parameter'}, inplace=True)

        dates = pd.to_datetime(df['year'] * 1000000 + df['month'] * 10000 + 1 * 100, format='%Y%m%d%H')

        # Index the dataframe based on datetimes
        df.index = pd.DatetimeIndex(dates)
        df['weekday'] = df.index.weekday  # added to be used for calculating statistics
        df = df.sort_index(ascending=True)
        # adjust column names
        #  df.rename(columns={'value': 'AT'}, inplace=True)

        self.raw_data = df.sort_index(ascending=True)

        df_station = df[df['stat_no']==int(self.station_name)]
        
        if not df_station[df_station['parameter']=='AT']['value'].isna().all():
            self.datasets['AT'] = AirTemp(AT=df_station[df_station['parameter']=='AT']['value'])

        if not df_station[df_station['parameter']=='PRE']['value'].isna().all():
            self.datasets['PRE'] = Precipitation(PRE=df_station[df_station['parameter']=='PRE']['value'])

        if not df_station[df_station['parameter']=='AP']['value'].isna().all():
            self.datasets['AP'] = AirPressure(AP=df_station[df_station['parameter']=='AT']['value'])




class NESDISType(ClimateDataSetBase):
    dstype = 'NESDIS'
    description = 'Data type as retreived from NESDIS online database at https://www.ncdc.noaa.gov/cdo-web/'

    def __init__(self, file, **kwargs):
        super().__init__(file, **kwargs)

    def read_data(self, file, encoding=None):
        # Read time series and process date columns
        raw_data = pd.read_csv(file, sep=';',
                         na_values=['', ' ', '\t', 'NaN', 'T', '99999', '9999.9', '999.99'],
                         encoding=encoding)

        dates = pd.to_datetime(raw_data['year'] * 1000000 + raw_data['month'] * 10000 + raw_data['day'] * 100, format='%Y%m%d%H')

        # Index the dataframe based on datetimes
        raw_data.index = pd.DatetimeIndex(dates)

        raw_data['weekday'] = raw_data.index.weekday  # added to be used for calculating statistics

        # To add time-zone information use:
        # raw_data.tz_localize('utc')                    # to get UTC+00:00
        # or
        # raw_data.tz_localize(tz.FixedOffset(-2*60))    # to get UTC-02:00
        # both will convert to time-zone aware datetimes, leaving the time stamp as it is, just adding the tz information.

        raw_data['weekday'] = raw_data.index.weekday  # added to be used for calculating statistics

        # convert Farenheit to Celcius
        for field in ['Tmax', 'Tmin', 'Tobs', 'Tmean']:
            raw_data[field] = raw_data[field].apply(farenheit2celsius)

        # convert inches to mm
        for field in ['Prcp']:
            raw_data[field] = raw_data[field].apply(inches2mm)

        self.raw_data = raw_data

        # Now assign datasets where data is present
        self.datasets['AT'] = AirTemp(ATmin=raw_data['Tmin'],
                                      ATmax=raw_data['Tmax'])
       
        self.datasets['PRE'] = Precipitation(PRE=raw_data['Prcp'])

        #self.datasets['AT'] = AirTemp(at_series=converted['ttt'])
        #self.datasets['RH'] = RelativeHumidity(rh=converted['rh'])
        #self.datasets['WIND'] = Wind(windspeed=converted['ff'], winddir=converted('ddd'))
        #self.datasets['PRE'] = Precipitation(precip=converted['pppp'])
        #self.datasets['SD'] = SnowDepth(precip=converted['sss'])

    # def is_valid(self):
    #     if self.raw_data['TAVG'].count() == 0:
    #         return False
    #     else:
    #         return True

    # def calc_daily_averages(self):

    #     # METHOD IS KEPT HERE FOR DOCUMENTATION PURPOSES.
    #     # some old data sets have only max and min temperatures measured
    #     # these are currently not handled.
    #     # But this is the way to handle them when needed.
    #     #
    #     # We need to implement a way to use this information in the AirTemp
    #     # class.

    #     AT_mean_values = np.where(self.raw_data['TAVG'].isna(),
    #                               (self.raw_data['TMAX'] + self.raw_data['TMIN']) / 2,
    #                               self.raw_data['TAVG'])
    #     AT_mean_count = 24   # TODO: Should this be set to 2 instead of 24?

    #     # Create new dataframe with daily averages
    #     AT_mean = pd.DataFrame(AT_mean_values, index=np.unique(self.raw_data.index.date))
    #     AT_mean.columns = ['AT']
    #     AT_mean['count'] = AT_mean_count

    #     self.AT_mean = AT_mean


class CustomType1(ClimateDataSetBase):
    dstype = 'US_YDE'
    description = 'Custom data format for some data received from Jacob Yde'

    def __init__(self, file, **kwargs):
        super().__init__(file, **kwargs)

    def read_data(self, file, encoding=None):
        # Read time series and process date columns
        raw_data = pd.read_csv(file, sep=';', dtype={'YRMODAHRMN': str, 'TEMP': float},
                               na_values=['', ' ', '\t', 'NaN', 'T', '99999', '9999.9', '999.99', '****'], encoding=encoding)

        # Convert dates, assume 12 o'clock noon
        dates = pd.to_datetime(raw_data['YRMODAHRMN'], format='%Y%m%d%H%M')

        # Index the dataframe based on datetimes
        # dates = pd.to_datetime(dates)
        raw_data.index = pd.DatetimeIndex(dates)

        raw_data['weekday'] = raw_data.index.weekday  # added to be used for calculating statistics

        # adjust column names to be like DMI column names
        raw_data.rename(columns={'TEMP': 'ttt'}, inplace=True)
        raw_data['ttt'] = raw_data['ttt'].apply(farenheit2celsius)


        self.raw_data = raw_data

        self.datasets['AT'] = AirTemp(AT=raw_data['ttt'])
        #self.datasets['RH'] = RelativeHumidity(rh=converted['rh'])
        #self.datasets['WIND'] = Wind(windspeed=converted['ff'], winddir=converted('ddd'))
        #self.datasets['PRE'] = Precipitation(precip=converted['pppp'])
        #self.datasets['SD'] = SnowDepth(precip=converted['sss'])



class DMIUnified(ClimateDataSetBase):
    dstype = 'DMIUnified'
    description = 'Will read both old and new files for a given station, provided they are available'

    def __init__(self, station_name, old_data_path=None, new_data_path=None, **kwargs):
        super().__init__(**kwargs)

        cd_old = None
        cd_new = None

        if old_data_path is None:
            old_data_repository = DataRepository(DMI_PATH_OLD_FORMAT)
        else:
            old_data_repository = DataRepository(old_data_path)

        if new_data_path is None:
            new_data_repository = DataRepository(DMI_PATH_NEW_FORMAT)
        else:
            new_data_repository = DataRepository(new_data_path)

        # station names can be like 34221, 04221 or 4221
        # parse old and new station names
        if len(station_name) == 5 and station_name[0] == '0':
            self.station_name = station_name[1:]
        else:
            self.station_name = station_name

        # Read the old data if it exists
        try:
            self.old_file = old_data_repository.get_file(self.station_name.rjust(5,'0')+'.txt')
        except FileNotFoundError:
            self.old_file = None

        # Create old format time series, if it exists
        if self.old_file is not None:
            cd_old = DMIType1(self.old_file, encoding=self.old_file.encoding)
            self.merge_with(cd_old, warn=False)
        
        # find the new type datafile:
        file_list = new_data_repository.glob(self.station_name+'*.csv')
        if len(file_list)>0:
            self.new_file = new_data_repository.get_file(sorted(file_list)[-1])
        else:
            self.new_file = None

        # Create new format time series, if it exists
        if self.new_file is not None:
            cd_new = DMIType2(self.new_file, encoding=self.new_file.encoding)
            self.merge_with(cd_new, warn=False)

    def extract_AT_min_max(self):
        """Extract min and max temperatures based on specific
        times of day, on days where the daily data threshold is not met,
        and where min/max values are not aready available."""

        counts = self.datasets['AT'].raw_data.groupby(self.datasets['AT'].raw_data.index.date).count()

        if ('ATmin' in counts.columns) & ('ATmax' in counts.columns):
            # We already have some ATmin/ATmax values

            # PSEUDO CODE
            # where AT.count < data threshold
            #       and ATmin.count = 0
            #       and ATmax.count = 0
            #    take 00:00:00 as ATmin (if it exists)
            #    take 12:00:00 as ATmax (if it exists)

            idx = (counts['AT']<self.datasets['AT'].daily_data_threshold) & (counts['ATmin']==0) & (counts['ATmax']==0).values
            
            ds_idx = self.datasets['AT'].raw_data.index.floor('d').isin(counts[idx].index)

            rd = self.datasets['AT'].raw_data
            rd['ATmin'].iloc[ds_idx & (rd.index.hour==0) & (rd.index.minute==0)] = rd[ds_idx & (rd.index.hour==0) & (rd.index.minute==0)]['AT']
            rd['ATmax'].iloc[ds_idx & (rd.index.hour==12) & (rd.index.minute==0)] = rd[ds_idx & (rd.index.hour==12) & (rd.index.minute==0)]['AT']


        else:
            # We have no ATmin/ATmax values, and must create the columns
            pass


CLIMATE_FILE_CLASSES = {'DMI1': DMIType1,
                        'DMI2': DMIType2,
                        'DMI3': DMIType3,
                        'DMIX': DMITypeX,
                        'DMILongDaily': DMITypeLongDaily,
                        'DMILongMonthly': DMITypeLongMonthly,
                        'NESDIS': NESDISType,
                        'US_YDE': CustomType1,}


# =====================================================================================
# PLOTTING FUNCTIONS
# =====================================================================================


def plot_maat(df_maat, title='MAAT plot', ax=None, color='k', skip=False, plot_corrected=False):
    """Plot mean annual air temperature alone"""
    if ax is None:
        fig = plt.figure()
        ax = plt.subplot(111)
    else:
        fig = ax.figure

    if skip:
        maat = df_maat.mask(df_maat['skip_flag'])
    else:
        maat = df_maat

    if plot_corrected:
        maat['MAATcor'].plot(ax=ax, drawstyle="steps-post", color=color)
    else:
        maat['MAAT'].plot(ax=ax, drawstyle="steps-post", color=color)

    ax.set_xlabel('')
    ax.set_ylabel(r'MAAT [C]')
    ax.set_xlabel('Year')

    ax.grid(True)

    plt.gcf().suptitle(title)
    return ax


def plot_maat_dd(df_maat, df_dd, title='MAAT+DegreeDay plot', ax=None, cmaat='k', ctdd='r', cfdd='b', skip=False, plot_corrected=False):
    """ Plot mean annual air temperature and the degree days of freezing 
    and thawing in separate axes.
    """
    if ax is None:
        fig = plt.figure()
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)
    else:
        fig = ax.figure

    if skip:
        maat = df_maat.mask(df_maat['skip_flag'])
        dd = df_dd.mask(df_dd['skip_flag'])
    else:
        maat = df_maat
        dd = df_dd

    if plot_corrected:
        maat['MAATcor'].plot(ax=ax1, drawstyle="steps-post", color=cmaat, label='MAAT')
    else:
        maat['MAAT'].plot(ax=ax1, drawstyle="steps-post", color=cmaat, label='MAAT')

    ax1.set_xlabel('')
    ax1.set_ylabel(r'MAAT [C]')

    dd['tdd'].plot(ax=ax2, drawstyle="steps-post", color=ctdd, label='TDD')
    dd['fdd'].plot(ax=ax2, drawstyle="steps-post", color=cfdd, label='FDD')

    ax2.set_xlabel('Year')
    ax2.set_ylabel(r'TDD [C*days]')
    ax2.set_ylabel(r'FDD & TDD [C*days]')

    # make these tick labels invisible
    plt.setp(ax1.get_xticklabels(), visible=False)

    ax1.grid(True)
    ax2.grid(True)

    #lines_1, labels_1 = ax1.get_legend_handles_labels()
    #lines_2, labels_2 = ax2.get_legend_handles_labels()
    #ax1.legend(lines_1+lines_2,labels_1+labels_2, fontsize='small')

    ax1.legend(fontsize='small')
    ax2.legend(fontsize='small')

    plt.gcf().suptitle(title)
    return ax

def cross_plot_temp(sta1, sta2, title='Cross-plot', ax=None):
    """Create a cross plot of data from the two stations passed."""

    #raise NotImplementedError()

    if ax is None:
        fig = plt.figure()
        ax = plt.subplot(111)
    
    def a2b(p):
        """Return a 1D polynomial."""
        return lambda x: np.polyval(p, x) 
    
    def b2a(p):
        """Return a 1D polynomial."""
        return lambda x: np.polyval([1/p[0], -p[1]/p[0]], x) 
    
    ids = sta1.datasets['AT'].daily_data.dropna(subset=['AT']).index.intersection(sta2.datasets['AT'].daily_data.dropna(subset=['AT']).index)
    
    x = sta1.datasets['AT'].daily_data.loc[ids, 'AT']
    y = sta2.datasets['AT'].daily_data.loc[ids, 'AT']
    
    ax.scatter(x, y)
    ax.plot([-40, 15], [-40, 15], '--k')
    #ax.plot([0, 1], [0, 1], '--k', transform=ax1.transAxes)
    ax.axis('square')
    
    ax.set_xlabel(sta1.station_name+' Temperature [$^\\circ$C]')
    ax.set_ylabel(sta2.station_name+' Temperature [$^\\circ$C]')
    
    p, cov = np.polyfit(x, y, 1, cov=True)
    y_model = a2b(p)                                   # model using the fit parameters; NOTE: parameters here are coefficients
    
    r_sq = 1 - (sum((y - y_model(x))**2) / ((len(y) - 1) * np.var(y, ddof=1)))
    #r_sq_b = np.corrcoef(x,y)[0, 1]**2
    
    ax.plot(x, y_model(x), "-", color="0.1", linewidth=1.5, alpha=0.5, label="Fit")
    
    #plot_trendline(sta1.AT_mean.loc[ids, 'AT'], sta2.AT_mean.loc[ids, 'AT'], ax=ax1)
    
    if np.sign(p[1]) == -1:
        eqn = '$y = {0:.3f}\\cdot x-{1:.3f}$\n$R^2 = {2:.3f}$\n$N = {3:.0f}$'.format(p[0], np.abs(p[1]), r_sq, len(x))
    else:
        eqn = '$y = {0:.3f}\\cdot x+{1:.3f}$\n$R^2 = {2:.3f}$\n$N = {3:.0f}$'.format(p[0], np.abs(p[1]), r_sq, len(x))

    text_kwargs = dict(ha='left', va='top', fontsize=10, color='k')
    plt.text(0.05, 0.98, eqn,  transform=ax.transAxes, **text_kwargs)
    
    ax.set_title(title)
    #plt.gcf().suptitle(title)
    
    return ax
    

def plot_warming_stripes(cd, FIRST=None, LAST=None):
    """Plot warming stripes illustration of mean annual air temperature
    for the climate data passed.

    cd is a ClimateDataSetBase decendant (e.g. DMIType1, DMIUnified or other class instance).
    """

    #raise NotImplementedError()
    
    TICK_FS = 14 # Tick label font size
    
    maat = cd.datasets['AT'].calc_MAAT()  # extract only MAAT data
    #maat = maat[~maat['skip_flag']]
    reference = maat['MAAT'].mean()  # Use average of whole timeseries as center of colorscale
    
    if LAST is None:
        LAST = maat.index[-1]
    
    if FIRST is None:
        FIRST = maat.index[0]
    
    # set the colorscale range to +/- the maximum deviation from reference
    minT = maat.MAAT.min()
    maxT = maat.MAAT.max()
    
    LIM1 = np.nanpercentile(maat.MAAT-reference, 1)
    LIM2 = np.nanpercentile(maat.MAAT-reference, 99)
    
    LIM = np.max([LIM1,LIM2])
    
    # the colors in this colormap come from http://colorbrewer2.org
    # the 8 more saturated colors from the 9 blues / 9 reds
    
    cmap = ListedColormap([
        '#08306b', '#08519c', '#2171b5', '#4292c6',
        '#6baed6', '#9ecae1', '#c6dbef', '#deebf7',
        '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a',
        '#ef3b2c', '#cb181d', '#a50f15', '#67000d',
    ])
    cmap.set_bad('#ffffff')
    
    
    fig = plt.figure(figsize=(10*(LAST-FIRST)/(LAST-1794), 1.5))
    ax = plt.axes()
    
    fig.canvas.manager.set_window_title(cd.station_name)
    
    # create a collection with a rectangle for each year
    # and set the appropriate color (no edge/line)
    # if bad data, draw white rectangle
    patches = []
    for y in range(int(FIRST), int(LAST) + 1):
        if y in maat.index:
            if pd.isna(maat.MAAT[y]):
                patches.append(Rectangle((y, 0), 1, 1, color='#ffffff', fill=False, ec='#ffffff', lw=None))
            else:
                color = cmap((maat.MAAT[y]-(reference-LIM))/(2*LIM))
                patches.append(Rectangle((y, 0), 1, 1, color=color, fill=True, ec=color, lw=None))
        else:
            patches.append(Rectangle((y, 0), 1, 1, color='#ffffff', fill=False, ec='#ffffff', lw=None))
            
    col = PatchCollection(patches, match_original=True)
    
    
    # set data, colormap and color limits
    col.set_cmap(cmap)
    col.set_clim(reference - LIM, reference + LIM)
    ax.add_collection(col)
    
    # Set axes properties
    ax.set_ylim(0, 1)
    ax.set_xlim(FIRST, LAST + 1)
    ax.xaxis.tick_top()
    plt.xticks(fontsize=TICK_FS) 
    plt.gca().axes.yaxis.set_visible(False)  # don't show y-axis
    
    # add colorbar to plot
    cbar = fig.colorbar(col, orientation='horizontal', fraction=.1)
    
    # rearange axes to show everything
    plt.tight_layout()
    
    #plt.savefig('warming_stripes_{0}.png'.format(site), transparent=True, dpi=300)
    return ax



# =====================================================================================
# EXAMPLE OF USAGE
# =====================================================================================


if __name__ == '__main__':
    # Define stations and read data files
    stations = {}
    #stations['QNQ04205'] = DMIUnified('4205')
    stations['ILU04221'] = DMIUnified('4221')
    stations['ILU04216'] = DMIUnified('4216')
    #stations['SIS04234'] = DMIUnified('4234')
    #stations['SIS04230'] = DMIUnified('4230')
    #stations['KAN04231'] = DMIUnified('4231')
    #stations['ITT04339'] = DMIUnified('4339')
    #stations['ITT04340(UUN)'] = DMIUnified('4340')
    
    # ==========================================================
    # example of processing and plotting individual timeseries
    # ==========================================================
    
    for key in stations.keys():
        # loop through each station

        print(' ')
        print('Processing site: {0}'.format(key))

        # Calculate mean annual air temperature
        maat = stations[key].datasets['AT'].calc_MAAT()
        
        # Calculate freezing and thawing degree days
        dd = DegreeDays(AT=stations[key].datasets['AT']).degree_days

        # Write air temperatures to excel file
        print('Writing "Daily mean Tair" sheet...')
        
        out_file = '{0}_weather_proccessed.xlsx'.format(key)
        file = write_xlsx(out_file,                  # filename to write to    
                          stations[key].datasets['AT'].daily_data,     # dataframe to write
                          'Daily mean Tair',         # name of the sheet in the excel file
                          drop_cols=['stat_no', 'year', 'month', 'day', 'weekday', 'hour', 'minute'],   # names of columns NOT to write
                          overwrite=True)            # overwrites file if it exists (if False, an integer will be appended to the filename to make it unique)

        # Write mean annual air temperatures to excel file
        print('Writing "Mean annual Tair" sheet...')
        file = write_xlsx(file, maat, 'Mean annual Tair w stats', overwrite=True)

        # Write freezing and thawing indices to excel file
        print('Writing "FDD and TDD" sheet...')
        file = write_xlsx(file, dd, 'FDD and TDD', overwrite=True)
        
        # finally close the file
        file.close()
        print('done!')

        # plot mean annual air temperature and degree days
        plot_maat_dd(maat, dd, title='{0} MAAT and DegreeDays'.format(key),
                     skip=True, plot_corrected=False)

        # save the figure to a png file
        plt.savefig('{0}_MAAT_dd.png'.format(key), dpi=300, facecolor='w', edgecolor='w',
                    orientation='portrait', format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1)

  
    # ==========================================================
    # example of combining two time series
    # ==========================================================
    
    # ILU04221 and ILU04216 are data from two weather stations in
    # the Ilulissat area which had overlapping data collection
    # periods.
    # We may approximate a continuous timeseries from the two by
    # combining the two after scaling and bias correcting the
    # on to the other.
    
    print('')
    print('Constructing combined time series for Ilulissat')
    
    # Produce a cross plot of the overlapping part of two timeseries
    ax = cross_plot_temp(stations['ILU04216'],stations['ILU04221'])
    ax.set_title('Crossplot before correction')
    
    # Perform scaling and bias correction of older timeseries to 
    # newer timeseries, and return a new station instance.
    new = stations['ILU04216'].scale_debias_to(stations['ILU04221'])
    
    # Merge the scaled and bias corrected timeseries with the data 
    # from the SIS04234 station
    new = new.merge_with(stations['ILU04221'])
    new.old_file = None
    new.new_file = None
    new.station_name = 'ILU04216+ILU04221'

    # plot a new cross plot of the corrected station data
    ax = cross_plot_temp(new,stations['ILU04221'])
    ax.set_title('Crossplot after scaling and bias correction')
    
    # Now that we have the full time series, calculate mean 
    # annual air temperature and degree days statistics.
    maat = new.datasets['AT'].calc_MAAT()
    dd = DegreeDays(AT=new.datasets['AT']).degree_days
    
    # Write air temperatures to excel file
    print('Writing "Daily mean Tair" sheet...')
    
    out_file = '{0}_weather_proccessed.xlsx'.format('ILU04216+ILU04221')
    file = write_xlsx(out_file,                  # filename to write to    
                        new.datasets['AT'].daily_data,     # dataframe to write
                        'Daily mean Tair',         # name of the sheet in the excel file
                        drop_cols=['stat_no', 'year', 'month', 'day', 'weekday', 'hour', 'minute'],   # names of columns NOT to write
                        overwrite=True)            # overwrites file if it exists (if False, an integer will be appended to the filename to make it unique)

    # Write mean annual air temperatures to excel file
    print('Writing "Mean annual Tair" sheet...')
    file = write_xlsx(file, maat, 'Mean annual Tair w stats', overwrite=True)

    # Write freezing and thawing indices to excel file
    print('Writing "FDD and TDD" sheet...')
    file = write_xlsx(file, dd, 'FDD and TDD', overwrite=True)
    
    # finally close the file
    file.close()
    print('done!')

    # ... and plot it
    plot_maat_dd(maat, dd, title='ILU04216+ILU04221 MAAT and DegreeDays', skip=True)
    
    # ... and save it to a png file.
    plt.savefig('ILU04216+04221_MAAT_dd.png'.format(key), dpi=300, facecolor='w', edgecolor='w',
                orientation='portrait', format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1)
    
    # plot warming stripes figure for the combined time series
    plot_warming_stripes(new)

    # ... and save it to a png file.
    plt.savefig('ILU_warming_stripes.png'.format(key), dpi=300, facecolor='w', edgecolor='w',
                orientation='portrait', format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1)
                
    plt.show(block=False)                