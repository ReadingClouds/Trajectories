"""
Interpolate selected variables from input gridded dataset to trajectories.
"""
import time
import xarray as xr

from .interpolation import interpolate_3d_field

def data_to_traj(source_dataset: xr.Dataset,
                 ds_traj: xr.Dataset,
                 varlist: list,
                 output_path: str,
                 interp_order:int=5,
                 output_precision:str="float64",
                 write_sleeptime:int=3  
                )->dict():
    """
    Interpolate 3D variables to trajectory points.

    Parameters
    ----------
    source_dataset : xr.Dataset
        Input variables on 3D grid at times matching trajectory times.
    ds_traj : xr.Dataset
        Trajectory positions, 'x', 'y' and 'z'.
    varlist : list(str)
        List of strings with variable names required from source_dataset.
    output_path : str
        Path to save output NetCDF file.
    interp_order : int, optional
        Order of polynomial interpolation. The default is 5.
    output_precision : str, optional
        Data type for output. The default is "float64".
    write_sleeptime : int, optional
        Pause after write. The default is 3.

    Returns
    -------
    dict
        'file': output_path.
        'ds'  : output xarray Dataset.

    """
    
    atts = source_dataset.attrs
   
    ds_out = xr.Dataset()
    for inc in atts:
        if isinstance(atts[inc], (dict, bool, type(None))):
            atts[inc] = str(atts[inc])
            
    ds_out.attrs = atts

    ds_out.to_netcdf(output_path, mode='w')
    
    for var_name in varlist:
        print(f'Mapping {var_name} onto trajectories.')
        
        da = source_dataset[var_name]                                                
        varout = []
        for traj_time in ds_traj.time:
                        
            dat = da.sel(time=traj_time)
            
            ds_positions = ds_traj[['x','y','z']].sel(time=traj_time)
            
            interp_data =interpolate_3d_field(dat, 
                                              ds_positions, 
                                              interp_order=interp_order, 
                                              cyclic_boundaries='xy')
            
            varout.append(interp_data)
            
        ds_out[var_name] = xr.concat(varout, dim="time")
        
        encoding = {var_name: 
                    {"dtype": output_precision }}
       
        print(f'Saving {var_name}.')
        d = ds_out[var_name].to_netcdf(output_path,
                                       unlimited_dims="time",
                                       mode='a', 
                                       encoding = encoding)

        # This wait seems to be needed to give i/o time to flush caches.
        time.sleep(write_sleeptime)
    
    return {'file': output_path, 'ds':ds_out}
    

    

