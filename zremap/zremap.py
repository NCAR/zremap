#! /usr/bin/env python
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import logging

import xarray as xr
import numpy as np
from netCDF4 import default_fillvals

from . import remap_z
from . import remap_z_dbl

logging.basicConfig(level=logging.INFO)
xr_open_dataset_args = {'decode_times': False, 'decode_coords': False}


def _remap_z_type(klev_out, KMT, z_edges, VAR_IN, NEW_Z, new_z_edge):
    '''Remap vertical coordinate.'''

    #-- make sure that type is the same
    NEW_Z = NEW_Z.astype(VAR_IN.dtype)
    new_z_edge = new_z_edge.astype(VAR_IN.dtype)
    z_edges = z_edges.astype(VAR_IN.dtype)

    if VAR_IN.dtype == 'float32':
        logging.debug('calling remap_z: single precision')
        msv = default_fillvals['f4']
        THICKNESS, VAR_OUT = remap_z.remap_z(klev_out=klev_out,
                                             kmt=KMT.T,
                                             z_edge=z_edges,
                                             var_in=VAR_IN.T,
                                             new_z=NEW_Z.T,
                                             new_z_edge=new_z_edge,
                                             msv=default_fillvals['f4'])
    elif VAR_IN.dtype == 'float64':
        logging.debug('calling remap_z: double precision')
        msv = default_fillvals['f8']
        THICKNESS, VAR_OUT = remap_z_dbl.remap_z_dbl(klev_out=klev_out,
                                                     kmt=KMT.T,
                                                     z_edge=z_edges,
                                                     var_in=VAR_IN.T,
                                                     new_z=NEW_Z.T,
                                                     new_z_edge=new_z_edge,
                                                     msv=msv)
    else:
        raise ValueError('remap_z: unable to determine type')

    THICKNESS[THICKNESS == msv] = np.nan
    VAR_OUT[VAR_OUT == msv] = np.nan

    return THICKNESS.T, VAR_OUT.T


def _dim_index_to_slice(index_list):
    '''
    .. function:: _dim_index_to_slice(index_list)

    Convert string formatted as: dimname,start[,stop[,stride]]
    to index (for the case where only 'start' is provided)
    or indexing object (slice).

    :param index_list: index list as passed in from
                       -d dimname,start,stop,stride

    :returns: dict -- {dimname: indexing object}
    '''

    if len(index_list) == 1:
        return index_list[0]
    elif len(index_list) == 2:
        return slice(index_list[0], index_list[1])
    elif len(index_list) == 3:
        return slice(index_list[0], index_list[1], index_list[2])
    else:
        raise ValueError('illformed dimension subset')


def sigma_coord_edges(sigma_start, sigma_stop, sigma_delta):
    '''Generate a sigma coordinate vector.'''
    return np.arange(sigma_start, sigma_stop+sigma_delta, sigma_delta)

#------------------------------------------------------------------------
#-- FUNCTION
#------------------------------------------------------------------------
def _compute_kmt(ds, varname):
    '''Infer index into "bottom level" from missing values.'''

    nk = ds[varname].shape[1]

    #-- init KMT array
    KMT = np.zeros(ds[varname].shape[-2:]).astype(int)

    #-- where surface is missing, KMT = 0, else full depth
    KMT = np.where(np.isnan(ds[varname].values[0,0,:,:]), 0, nk)

    #-- loop over k
    #   where level k is missing: KMT = k,
    #   i.e. the level above in 1-based indexing
    for k in range(1,nk):
        KMT = np.where(np.isnan(ds[varname].values[0,k,:,:]) & (KMT > k),
                       k, KMT)

    return KMT

def z_to_sigma(ds, SIGMA, sigma_edges, zname, dzname, kmtname):
    '''Remap a dataset to sigma coordinates.'''

    #-- define output dataset
    dso = ds.copy()

    if isinstance(SIGMA, xr.DataArray):
        SIGMA = SIGMA.values.astype(SIGMA.encoding['dtype'])

    #-- the new coordinate
    sigma = xr.DataArray(np.average(np.vstack((sigma_edges[0:-1],
                                               sigma_edges[1:])), axis=0),
                         dims=('sigma'),
                         attrs={'long_name': 'sigma_theta',
                                'units': 'kg m$^{-3}$'})

    klev_out = len(sigma_edges)-1

    #-- list vars to remap, drop those with "z" dimension
    #   find dimesion lengths
    tlev = len(ds.time)
    varname_list = []
    copy_coords = {}
    for v in ds.variables:
        if zname in ds[v].dims:
            if ds[v].ndim == 4:
                varname_list.append(v)
                copy_coords.update({k: c for k, c in ds[v].coords.items()
                                    if k != zname and
                                    k not in copy_coords.keys()})
            dso = dso.drop(v)

    copy_coords['sigma'] = sigma

    if not varname_list:
        raise ValueError('no 4D variables found')
    else:
        jmt = ds[varname_list[0]].shape[-2]
        imt = ds[varname_list[0]].shape[-1]
        jdim = ds[varname_list[0]].dims[-2]
        idim = ds[varname_list[0]].dims[-1]


    #-- get old coordinate
    logging.debug('reading z')
    logging.debug(ds[zname])
    z = get_values(ds, zname)
    klev = len(z)

    logging.debug('reading dz')
    logging.debug(ds[dzname])
    dz = get_values(ds, dzname)
    z_edges = np.concatenate(([0.], np.cumsum(dz)))

    #-- read 1-based index of bottom level
    if kmtname:
        logging.debug('reading kmt')
        logging.debug(ds[kmtname])
        KMT = get_values(ds, kmtname)
    else:
        logging.debug('constructing kmt')
        KMT = _compute_kmt(ds,varname_list[0])
        dso['KMT_c'] = xr.DataArray(KMT,
                                    dims=(jdim, idim),
                                    attrs={'long_name':'Index of bottom cell'})

    if SIGMA.shape != (tlev, klev, jmt, imt):
        raise ValueError('SIGMA has shape: %s; expected shape %s'%(
                str(SIGMA.shape), str((tlev, klev, jmt, imt))))


    #-- remap z to get a thickness and depth field
    Z = np.broadcast_to(z[np.newaxis, :, np.newaxis, np.newaxis],
                        (tlev, klev, jmt, imt))

    VAR_OUT = np.empty((imt, jmt, klev_out, tlev)).astype(Z.dtype)
    THICKNESS = VAR_OUT.copy()

    logging.debug('remapping z')
    THICKNESS,VAR_OUT = _remap_z_type(klev_out, KMT, z_edges, Z, SIGMA,
                                      sigma_edges)

    dso['Z'] = xr.DataArray(VAR_OUT,
                            dims=('time', 'sigma', jdim, idim),
                            attrs={'long_name': 'Depth',
                                   'units': ds[zname].attrs['units']})

    dso['THICKNESS'] = xr.DataArray(THICKNESS,
                                    dims=('time', 'sigma', jdim, idim),
                                    attrs={'long_name': 'Thickness',
                                           'units': ds[zname].attrs['units']})

    #-- remap variables
    for v in varname_list:
        logging.debug('remapping %s', v)
        logging.debug(ds[v])
        VAR = get_values(ds, v)
        VAR_OUT = np.empty((imt, jmt, klev_out, tlev)).astype(VAR.dtype)

        _, VAR_OUT = _remap_z_type(klev_out, KMT, z_edges, VAR, SIGMA,
                                   sigma_edges)

        dso[v] = xr.DataArray(VAR_OUT,
                              dims=('time', 'sigma', jdim, idim),
                              attrs=ds[v].attrs,
                              encoding=ds[v].encoding)

    print(copy_coords)
    return dso.assign_coords(**copy_coords)

def _batch(file_in_data, zname, dzname, file_in_sigma, sigma_varname,
           convert_from_pd, sigma_start=24.475, sigma_stop=26.975,
           sigma_delta=0.05, file_out='', kmtname='', isel_kwargs={}):

    #-- read sigma file
    SIGMA = _read_sigma(file_in_sigma, sigma_varname, convert_from_pd,
                        isel_kwargs)

    #-- compute sigma coordinate
    sigma_edges = sigma_coord_edges(sigma_start, sigma_stop, sigma_delta)

    #-- open dataset
    ds = xr.open_dataset(file_in_data, **xr_open_dataset_args)
    if isel_kwargs:
        ds = ds.isel(**isel_kwargs)

    dso = z_to_sigma(ds, SIGMA, sigma_edges, zname, dzname, kmtname)

    #-- write output (or not)
    if file_out:
        dso.to_netcdf(file_out, unlimited_dims=['time'])

    return dso



#------------------------------------------------------------------------
#-- FUNCTION
#------------------------------------------------------------------------
def _read_sigma(file_in, sigma_varname,
               convert_from_pd=False,
               isel_kwargs={}):

    ds = xr.open_dataset(file_in,**xr_open_dataset_args)
    if isel_kwargs:
        ds = ds.isel(**isel_kwargs)

    logging.debug('reading %s'%sigma_varname)
    logging.debug(ds[sigma_varname])
    sigma = get_values(ds,sigma_varname)

    if convert_from_pd:
        sigma_order = np.floor(np.log10(np.nanmean(sigma)))

        if sigma_order == 0.:
            logging.debug('%s assumed to be cgs units'%(sigma_varname))
            sigma = (sigma - 1.)*1000.

        elif sigma_order == 3.:
            logging.debug('%s assumed to be mks units'%(sigma_varname))
            sigma = (sigma - 1000.)

        else:
            raise ValueError('ERROR: the units of %s could not be determined'%sigma_varname)

    return sigma


def get_values(ds, v):
    '''Return numpy array of the right type.'''
    return ds[v].values.astype(ds[v].encoding['dtype'])

if __name__ == '__main__':
    import argparse
    import json

    #-- set defaults
    control_defaults = {
        'file_in_data': None,
        'file_out': None,
        'file_in_sigma': None,
        'zname': None,
        'dzname': None,
        'kmtname': '',
        'sigma_varname': None,
        'convert_from_pd': False,
        'isel_kwargs':  {},
        'sigma_start': False,
        'sigma_stop': False,
        'sigma_delta': False}

    help_str = []
    for k,v in control_defaults.items():
        if v is None:
            help_str.append('%s : REQUIRED'%k)
        elif not v:
            help_str.append('%s : \'\''%k)
        else:
            help_str.append('%s : %s'%(k, v))

    p = argparse.ArgumentParser(description='Regrid to sigma coords')
    p.add_argument('json_control',
                   default=control_defaults,
                   help='{'+', '.join(help_str)+'}')
    p.add_argument('-f', dest='json_as_file',
                   action='store_true',default=False,
                   help='Interpret input as a file name')

    args = p.parse_args()
    if not args.json_as_file:
        control_in = json.loads(args.json_control)
    else:
        with open(args.json_control, 'r') as fp:
            control_in = json.load(fp)

    control = control_defaults
    control.update(control_in)

    #-- consider required arguments:
    missing_req = False
    req = ['file_in_data', 'file_out', 'zname', 'dzname', 'kmtname',
           'sigma_varname']
    for k in req:
        if control[k] is None:
            logging.debug('ERROR: missing %s', k)
            missing_req = True
    if missing_req:
        raise ValueError('stopping')

    #-- if no sigma file, assume same as input file
    if control['file_in_sigma'] is None:
        control['file_in_sigma'] = control['file_in_data']

    #-- convert isel_kwargs from list to slice format
    if control['isel_kwargs']:
        for k, v in control['isel_kwargs'].items():
            control['isel_kwargs'][k] = _dim_index_to_slice(v)

    logging.debug('running remap_z')
    for k, v in control.items():
        logging.debug('%s = %s', k, str(v))

    #-- compute
    _batch(**control)
