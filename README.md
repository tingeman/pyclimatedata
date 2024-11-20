# pyclimatedata

Package for reading and processing weather data, particularly from DMI, but can be modified to read files from any source.

## Installation & Usage

To install the package locally, follow either of these methods:

#### Method 1: Download the ZIP file from GitHub

1. Go to the GitHub repository page: https://github.com/tingeman/pyclimatedata
2. Click on "Code" and select "Download ZIP".
3. Extract the downloaded ZIP file.
4. Navigate into the extracted directory:
```sh
   cd /path/to/pyclimatedata
```

5. Install the package using pip:
```sh
   pip install .
```

This will install all necessary dependencies and set up the package for local use.


#### Method 2: Clone the repository from GitHub

1. Clone the repository from GitHub:
```sh
   git clone https://github.com/tingeman/pyclimatedata.git
```

2. Navigate into the cloned directory:
```sh
   cd /path/to/pyclimatedata
```

3. Install the package using pip:
```sh
   pip install .
```

This will install all necessary dependencies and set up the package for local use.



### Weather station data availability from DMI for Greenland

DMI publishes reports on a yearly basis, with data from available weather stations in Greenland.

Most recently, the 2024 report provides all available data (except snow depths) from all available stations in Greenland.
- report number *24-08* "Weather Observations From Greenland, 1958-2023"

In pprevious years (up until 2021) there were two yearly reports Published:
- report number *21-08* reports the full time series of hourly measurements (when available)
- report number *21-04* reports only daily averages, but data dates back to the late 1800s.

The reports can be downloaded from: https://www.dmi.dk/publikationer/
There are also links to downloads of zip files containing the actual data.

### Usage

Import module and other dependencies:
```python
import climate_data as cdata
import pathlib
```
Set relevant data paths to where you store the zip-files with DMI data.

The latest (2024) DMI data report includes data spread across two different zip files. We therefore provide a list of paths to search, when loading files from this dataset.

```python
DMI_PATH_UNIFIED_FORMAT = [pathlib.Path(r'C:\path\to\DMIRep24-08_1958_2023_data1.zip'),
                           pathlib.Path(r'C:\path\to\DMIRep24-08_1958_2023_data2.zip')]
```

Import data from Sisimiut airport weatherstation (423400):

```python
dat = cdata.load_climate_data('428000')
```

The module will load and process the data from the repository specified for `DMI_PATH_UNIFIED_FORMAT`
as default. A `repository` keyword argument may be passed to load data from elsewhere. See code or 
included Notebook for documentation/example usage.

### Extending to other data sources

Other data sources than DMI weather station data may be used.
Implement a subclass of `ClimateDataSetBase`, `DMIType1`, `DMIType2`,  or `DMIType3`, and override the `read_data` method, to set up data import. Other methods and properties will be inherited from the parent class.

## Uninstall

To uninstall package, use:

```sh
    pip uninstall pyclimatedata
```






