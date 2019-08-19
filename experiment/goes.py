"""GOES module

A module for working with GOES data.

Module Author:: S.Chakraborty, 17th May 2017

Functions
--------------------------------------------------------
read_goes       download GOES data
goes_plot       plot GOES data
classify_flare  convert GOES data to string classifier
flare_value     convert string classifier to lower bound
find_flares     find flares in a certain class
--------------------------------------------------------

"""

import os
import re
import urllib2
import pycurl
import datetime as dt
from dateutil import relativedelta
from davitpy import rcParams
import pandas as pd
import netCDF4
import numpy as np

URLS = ['https://satdat.ngdc.noaa.gov/sem/goes/data/', 'https://satdat-vip.ngdc.noaa.gov/sem/goes/data/']


def __monthdelta(sdate, edate):
    r = relativedelta.relativedelta(edate, sdate)
    if r.years > 0:
	r.months = r.years * 12 
	r.years = 0
    return r

def __get_urls(sdate,delta,sat_nr,base_url):
    urls = []
    mformat = '%02d'
    for m in range(delta.months+1):
	DT = sdate + relativedelta.relativedelta(months=m)
	urls.append(base_url + str(DT.year) + "/" + mformat % DT.month + "/" + "goes" + sat_nr + "/" + "netcdf/")
    return urls

def __clear_tmp_dir(f_paths):
    for f in f_paths:
	try:
	    os.remove(f)
	except:
	    import traceback
            #traceback.print_exc()
	    pass
    return

def __extract_data(f_paths, data_dict, sat_nr, sTime, eTime, is_high_res):
    # Load data into memory. #######################################################
    df_xray     = None
    df_orbit    = None
    for file_path in f_paths:
        nc = netCDF4.Dataset(file_path)

        #Put metadata into dictionary.
        fn  = os.path.basename(file_path)
        data_dict['metadata'][fn] = {}
        md_keys = ['NOAA_scaling_factors','archiving_agency','creation_date','end_date',
                   'institution','instrument','originating_agency','satellite_id','start_date','title']
        for md_key in md_keys:
            try:
                data_dict['metadata'][fn][md_key] = getattr(nc,md_key)
            except:
                pass

        #Store Orbit Data
        tt = nc.variables['time_tag_orbit']
        jd = np.array(netCDF4.num2date(tt[:],tt.units))

        orbit_vars = ['west_longitude','inclination']
        data    = {}

        for var in orbit_vars:
            data[var] = nc.variables[var][:]

        df_tmp = pd.DataFrame(data,index=jd)
	if df_orbit is None:
            df_orbit = df_tmp
        else:
            df_orbit = df_orbit.append(df_tmp)

        #Store X-Ray Data
        tt = nc.variables['time_tag']
        jd = np.array(netCDF4.num2date(tt[:],tt.units))

	if is_high_res: tag_vars = ['A_QUAL_FLAG','A_COUNT','A_FLUX','B_QUAL_FLAG','B_COUNT','B_FLUX']
        else: tag_vars = ['A_QUAL_FLAG','A_NUM_PTS','A_AVG','B_QUAL_FLAG','B_NUM_PTS','B_AVG']
        
        
        if int(sat_nr) <= 12: tag_vars = ['xl','xs']
        data = {}
        for var in tag_vars:
            data[var] = nc.variables[var][:]

        df_tmp = pd.DataFrame(data,index=jd)
        if df_xray is None:
            df_xray = df_tmp
        else:
            df_xray = df_xray.append(df_tmp)

        #Store info about units
        for var in (tag_vars + orbit_vars):
            data_dict['metadata']['variables'][var] = {}
            var_info_keys = ['description','dtype','long_label','missing_value','nominal_max','nominal_min','plot_label','short_label','units']
            for var_info_key in var_info_keys:
                try:
                    data_dict['metadata']['variables'][var][var_info_key] = getattr(nc.variables[var],var_info_key)
                except:
		    pass
	pass
    df_xray = df_xray[np.logical_and(df_xray.index >= sTime,df_xray.index < eTime)]
    data_dict['xray']   = df_xray
    df_orbit = df_orbit[np.logical_and(df_orbit.index >= sTime,df_orbit.index < eTime)]
    data_dict['orbit']  = df_orbit
    return data_dict

def __check_date(sTime, eTime, f, is_high_res):
    f = f.split('_')
    f = f[-1]
    flg = True
    if is_high_res:
	T = dt.datetime(int(f[0:4]),int(f[4:6]),int(f[6:]))
	if((sTime <= T) and (eTime >= T)): flg = True
	else: flg = False
    else: flg = True
    return flg

def read_goes(sdate,edate=None,sat_nr="15",is_high_res=False):
    """Download GOES X-Ray Flux data from the NOAA FTP Site and return a
    dictionary containing the metadata and a dataframe.

    Parameters
    ----------
    sTime : datetime.datetime
        Starting datetime for data.
    eTime : Optional[datetime.datetime]
        Ending datetime for data.  If None, eTime will be set to sTime
        + 0 day.
    sat_nr : Optional[str]
        GOES Satellite number.  Defaults to 15.
    is_high_res : Optional[bool]
	Returns high reolution data only available since 2001
    Returns
    -------
    Tuple containing two dictionary
	- Dictionary containing the meassage related to the operations
	- Dictionary containing metadata, pandas dataframe with GOES data.

    Notes
    -----
    Data is downloaded from
    https://satdat.ngdc.noaa.gov/sem/goes/data/
				or
    https://satdat-vip.ngdc.noaa.gov/sem/goes/data/

    Currently, 1-m averaged x-ray spectrum in two bands
    (0.5-4.0 A and 1.0-8.0 A).

    NOAA NetCDF files are cached in rcParams['DAVIT_TMPDIR']
    
    Example
    -------
        goes_data = read_goes(datetime.datetime(2014,6,21))
      
    written by S.Chakraborty, 7th March 2017

    """
    output = {}

    if edate is None: edate = sdate + dt.timedelta(days=1)

    if urllib2.urlopen(URLS[0]).getcode() == 200: 
	url = URLS[0]
    elif urllib2.urlopen(URLS[1]).getcode() == 200:
	url = URLS[1]
    else:
	raise Exception("Problem to connect to main server.", "NOAA Site is down!!")

    if is_high_res: url += "new_full/"
    else: url += "new_avg/"
    
    fold_urls = __get_urls(sdate,__monthdelta(sdate, edate),sat_nr,url)
    
    FILE_URLS = []
    FILES = []
    output['code'] = 0
    output['message'] = ''
    p = 'g' + sat_nr + '_xrs_(.*?).nc'
    for fold in fold_urls:
    	try:
            print fold
    	    response = urllib2.urlopen(fold).read()
    	    files = re.findall(p,response)
    	    file_urls = []
    	    for i in range(len(files)):
    		if __check_date(sdate, edate, files[i], is_high_res):
    		    files[i] = "g" + sat_nr + "_xrs_" + files[i] +".nc"
    		    file_urls.append(fold+files[i])
    	    FILES.extend(files)
    	    FILE_URLS.extend(file_urls)
    	except:
    	    import traceback
    	    traceback.print_exc()
    	    output['code'] = 1
    	    output['message'] = output['message'] + '\n Link - ' + fold + ' - is not working!' 
	pass
    indx = 0
    output['files'] = []

    try:
	tmp_dir = rcParams['DAVIT_TMPDIR']
	os.makedirs(tmp_dir)
    except:
	pass

    f_paths = []
    for f in FILE_URLS:
	try:
	    response = urllib2.urlopen(f)
	    if response.getcode() == 200: 
		nc_data = response.read()
		f_path = os.path.join(tmp_dir + FILES[indx])
		f_paths.append(f_path)
		with open(f_path,"w") as code:
		    code.write(nc_data)
		    pass
		output['files'].append(f_path)
	    else:
		output['code'] = 1
	        output['message'] = output['message'] + '\n Link - ' + f + ' - is not working!'
	except:
	    output['code'] = 1
	    output['message'] = output['message'] + '\n Link - ' + f + ' - is not working!'
	indx += 1
	pass
    data_dict = {}
    data_dict['sat_nr'] = sat_nr
    data_dict['metadata'] = {}
    data_dict['metadata']['variables'] = {}
    if len(f_paths) > 0:
	data_dict = __extract_data(f_paths, data_dict, sat_nr, sdate, edate, is_high_res)
    else:
	#output['message'] = 'No data available!'
	pass
    __clear_tmp_dir(f_paths)
    return (output,data_dict)


def flare_value(flare_class):
    """Convert a string solar flare class [1] into the lower bound in W/m**2 of the
    1-8 Angstrom X-Ray Band for the GOES Spacecraft.

    An 'X10' flare = 0.001 W/m**2.

    This function currently only works on scalars.

    Parameters
    ----------
    flare_class : string
        class of solar flare (e.g. 'X10')

    Returns
    -------
    value : float
        numerical value of the GOES 1-8 Angstrom band X-Ray Flux in W/m**2.

    References
    ----------
    [1] See http://www.spaceweatherlive.com/en/help/the-classification-of-solar-flares

    Example
    -------
        value = flare_value('X10')

    Written by S.Chakraborty, 7th March 2017

    """
    flare_dict  = {'A':-8, 'B':-7, 'C':-6, 'M':-5, 'X':-4}
    letter      = flare_class[0]
    power       = flare_dict[letter.upper()]
    coef        = float(flare_class[1:])
    value       = coef * 10.**power
    return value


def __split_sci(value):
    """Split scientific notation into (coefficient,power).
    This is a private function that currently only works on scalars.

    Parameters
    ----------
    value :
        numerical value

    Returns
    -------
    coefficient : float

    Written by S.Chakraborty, 7th March 2017

    """
    s   = '{0:e}'.format(value)
    s   = s.split('e')
    return (float(s[0]),float(s[1]))


def classify_flare(value):
    """Convert GOES X-Ray flux into a string flare classification.
    You should use the 1-8 Angstrom band for classification [1]
    (B_AVG in the NOAA data files).

    A 0.001 W/m**2 measurement in the 1-8 Angstrom band is classified as an X10 flare..

    This function currently only works on scalars.

    Parameters
    ----------
    value :
        numerical value of the GOES 1-8 Angstrom band X-Ray Flux in W/m^2.

    Returns
    -------
    flare_class : string
        class of solar flare

    References
    ----------
    [1] http://www.spaceweatherlive.com/en/help/the-classification-of-solar-flares

    Example
    -------
        flare_class = classify_flare(0.001)

    Written by S.Chakraborty, 7th March 2017

    """
    coef, power = __split_sci(value)
    if power < -7:
        letter  = 'A'
        coef    = value / 1e-8
    elif power >= -7 and power < -6:
        letter  = 'B'
    elif power >= -6 and power < -5:
        letter  = 'C'
    elif power >= -5 and power < -4:
        letter  = 'M'
    elif power >= -4:
        letter  = 'X'
        coef    = value / 1.e-4

    flare_class = '{0}{1:.1f}'.format(letter,coef)
    return flare_class


def goes_plot(goes_data,sTime=None,eTime=None,ymin=1e-9,ymax=1e-2,legendSize=10,legendLoc=None,ax=None,is_high_res=False):
    """Plot GOES X-Ray Data.

    Parameters
    ----------
    goes_data : dict
        data dictionary returned by read_goes()
    sTime : Optional[datetime.datetime]
        object for start of plotting.
    eTime : Optional[datetime.datetime]
        object for end of plotting.
    ymin : Optional[float]
        Y-Axis minimum limit
    ymax : Optional[float]
        Y-Axis maximum limit
    legendSize : Optional[int]
        Character size of the legend
    legendLoc : Optional[ ]
    ax : Optional[ ]
    is_high_res: Optional[bool]
	If data is a high resolution data

    Returns
    -------
    fig : matplotlib.figure
        matplotlib figure object that was plotted to

    Notes
    -----
    If a matplotlib figure currently exists, it will be modified
    by this routine.  If not, a new one will be created.

    Written by S.Chakraborty, 7th March 2017

    """
    import datetime
    import matplotlib

    if ax is None:
        from matplotlib import pyplot as plt
        fig     = plt.figure(figsize=(10,6))
        ax      = fig.add_subplot(111)
    else:
        fig     = ax.get_figure()

    if sTime is None: sTime = goes_data['xray'].index.min()
    if eTime is None: eTime = goes_data['xray'].index.max()

    sat_nr = goes_data['sat_nr']

    if is_high_res: var_tags = ['A_AVG','B_AVG']
    else: var_tags = ['A_FLUX','B_FLUX']
    if sat_nr <= 12: var_tags = ['xs','xl']
    for var_tag in var_tags:
        ax.plot(goes_data['xray'].index,goes_data['xray'][var_tag],label=goes_data['metadata']['variables'][var_tag]['long_label'])


    #Format the x-axis
    if eTime - sTime > datetime.timedelta(days=1):
        ax.xaxis.set_major_formatter(
                matplotlib.dates.DateFormatter('%H%M\n%d %b %Y')
                )
    else:
        ax.xaxis.set_major_formatter(
                matplotlib.dates.DateFormatter('%H%M')
                )

    sTime_label = sTime.strftime('%Y %b %d')
    eTime_label = eTime.strftime('%Y %b %d')
    if sTime_label == eTime_label:
        time_label = sTime_label
    else:
        time_label = sTime_label + ' - ' + eTime_label

    ax.set_xlabel('\n'.join([time_label,'Time [UT]']))
    ax.set_xlim(sTime,eTime)

    #Label Flare classes
    trans = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    classes = ['A', 'B', 'C', 'M', 'X']
    decades = [  8,   7,   6,   5,   4]

    for cls,dec in zip(classes,decades):
        ax.text(1.01,2.5*10**(-dec),cls,transform=trans)

    #Format the y-axis
    ax.set_ylabel(r'W m$^{-2}$')
    ax.set_yscale('log')
    ax.set_ylim(1e-9,1e-2)

    ax.grid()
    ax.legend(prop={'size':legendSize},numpoints=1,loc=legendLoc)

    file_keys = goes_data['metadata'].keys()
    file_keys.remove('variables')
    file_keys.sort()
    md      = goes_data['metadata'][file_keys[-1]]
    title   = ' '.join([md['institution'],md['satellite_id'],'-',md['instrument']])
    ax.set_title(title)
    return fig

def find_flares(goes_data,window_minutes=60,min_class='X1',sTime=None,eTime=None,is_high_res=False):
    """Find flares of a minimum class in a GOES data dict created by read_goes().
    This works with 1-minute averaged GOES data.

    Classifications are based on the 1-8 Angstrom X-Ray Band for the GOES Spacecraft.[1]

    Parameters
    ----------
    goes_data : dict
        GOES data dict created by read_goes()
    window_minutes : Optional[int]
        Window size to look for peaks in minutes.
        I.E., if window_minutes=60, then no more than 1 flare will be found
        inside of a 60 minute window.
    min_class : Optional[str]
        Only flares >= to this class will be reported. Use a
        format such as 'M2.3', 'X1', etc.
    sTime : Optional[datetime.datetime]
        Only report flares at or after this time.  If None, the earliest
        available time in goes_data will be used.
    eTime : Optional[datetime.datetime]
        Only report flares before this time.  If None, the last
        available time in goes_data will be used.

    Returns
    -------
    flares : Pandas dataframe listing:
        * time of flares
        * GOES 1-8 Angstrom band x-ray flux
        * Classification of flare

    References
    ----------
    [1] See http://www.spaceweatherlive.com/en/help/the-classification-of-solar-flares

    Example
    -------
        sTime       = datetime.datetime(2014,1,1)
        eTime       = datetime.datetime(2014,6,30)
        sat_nr      = 15 # GOES15
        goes_data   = read_goes(sTime,eTime,sat_nr)
        flares = find_flares(goes_data,window_minutes=60,min_class='X1')

    Written by S.Chakraborty, 7th March 2017

    """
    import datetime
    import pandas as pd
    import numpy as np

    df  = goes_data['xray']
    sat_nr = goes_data['sat_nr']

    if sTime is None: sTime = df.index.min()
    if eTime is None: eTime = df.index.max()

    # Figure out when big solar flares are.
    time_delta      = datetime.timedelta(minutes=window_minutes)
    time_delta_half = datetime.timedelta( minutes=(window_minutes/2.) )

    window_center = [sTime + time_delta_half ]

    while window_center[-1] < eTime:
	window_center.append(window_center[-1] + time_delta)

    if not is_high_res: lam_l = 'B_AVG'; lam_s = 'A_AVG';
    else: lam_l = 'B_FLUX'; lam_s = 'A_FLUX';
    if sat_nr <= 12: lam_s = 'xs'; lam_l = 'xl'

    b_avg = df[lam_l]
    b_avg = b_avg.groupby(b_avg.index).first()

    keys = []
    for win in window_center:
        sWin = win - time_delta_half
        eWin = win + time_delta_half

        try:
            arg_max = b_avg[sWin:eWin].argmax()
            if arg_max is np.nan:
                continue
            keys.append(arg_max)
        except:
            pass

    df_win      = pd.DataFrame({lam_l:b_avg[keys]})
    df_win.index = pd.to_datetime(df_win.index)

    flares      = df_win[df_win[lam_l] >= flare_value(min_class)]

    flares      = flares.groupby(flares.index).first()
    # Remove flares that are really window edges instead of local maxima.
    drop_list = []
    for inx_0,key_0 in enumerate(flares.index):
        if inx_0 == len(flares.index)-1: break
        
        inx_1   = inx_0 + 1
        key_1   = flares.index[inx_1]

        arg_min = np.argmin([flares[lam_l][key_0],flares[lam_l][key_1]])
        key_min = [key_0,key_1][arg_min]

        vals_between = b_avg[key_0:key_1]

        if flares[lam_l][key_min] <= vals_between.min():
            drop_list.append(key_min)

    if drop_list != []:
        flares  = flares.drop(drop_list)
    flares['class'] = map(classify_flare,flares[lam_l])

    return flares
