"""Regressions from the Codex review of PR #1 at 285660e (task 0008).

Each test reproduces one finding; all must pass once the finding is fixed.
"""
import calendar
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

SOURCE = Path(__file__).resolve().parents[1]
from metafilter.core import calculate_daily_metrics, apply_metafilter, load_metafilter_parameters
from metafilter.open_meteo import fetch_open_meteo_archive
from scripts.download_era5 import download_period
from metafilter.analog import AnalogModel
from metafilter.cli import process_era5_main

AREA = {'west':18.09, 'south':59.29, 'east':18.11, 'north':59.31}

def write_dataset(path, times, variables, lat=59.3, lon=18.1):
    data = {k: (('time','latitude','longitude'), np.broadcast_to(np.asarray(v).reshape(-1,1,1), (len(times),1,1)).copy()) for k,v in variables.items()}
    xr.Dataset(data, coords={'time':times, 'latitude':[lat], 'longitude':[lon]}).to_netcdf(path, engine='scipy')
    return path

@pytest.mark.parametrize('profile', ['sentinel1_default', 'sentinel2_extended'])
def test_cds_download_period_fetches_profile_requirements(tmp_path, monkeypatch, profile):
    calls = []
    mapping = {'2m_temperature':('t2m',293.15), 'total_precipitation':('tp',0.0), 'surface_solar_radiation_downwards':('ssrd',1e6), 'skin_temperature':('skt',293.15), 'soil_temperature_level_1':('stl1',293.15), 'volumetric_soil_water_layer_1':('swvl1',0.25), 'snow_depth':('sd',0.0), 'snowfall':('sf',0.0), '2m_dewpoint_temperature':('d2m',280), 'total_cloud_cover':('tcc',0.1), 'low_cloud_cover':('lcc',0.05)}
    def retrieve(dataset, request, output):
        calls.append(request)
        year, month = int(request['year']), int(request['month'])
        times = pd.DatetimeIndex([
            f'{year}-{month:02}-{day}T{hour}'
            for day in request['day'] for hour in request['time']
            if int(day) <= calendar.monthrange(year, month)[1]
        ])
        write_dataset(output, times, dict(mapping[k] for k in request['variable']))
    monkeypatch.setitem(sys.modules, 'cdsapi', SimpleNamespace(Client=lambda:SimpleNamespace(retrieve=retrieve)))
    with patch('scripts.download_era5.OUTPUT_DIR',str(tmp_path)):
        config = SOURCE/'filters'/f'{profile}.json'
        paths = download_period(2024,8,config,area=AREA)
        metrics = calculate_daily_metrics(paths['land'],area=AREA,cloud_file_path=paths['cloud'] or None)
        # A normal download -> aggregate -> shipped-profile chain must run.
        apply_metafilter(metrics,load_metafilter_parameters(config))

def test_empty_cloud_list_from_openmeteo_is_accepted(tmp_path):
    times = pd.date_range('2024-08-01',periods=48,freq='h')
    path = write_dataset(tmp_path/'land.nc',times,{'t2m':[293.15],'tp':[0.0],'tcc':[0.1]})
    result = calculate_daily_metrics([path],area=AREA,cloud_file_path=[])
    assert len(result)==2

def test_cloud_file_honours_small_aoi_grid_snap(tmp_path):
    times = pd.date_range('2024-08-01',periods=48,freq='h')
    land = write_dataset(tmp_path/'land.nc',times,{'t2m':[293.15],'tp':[0.0]})
    cloud = write_dataset(tmp_path/'cloud.nc',times,{'tcc':[0.1],'lcc':[0.05]},lat=59.25,lon=18.0)
    result = calculate_daily_metrics(land,area=AREA,cloud_file_path=cloud)
    assert len(result)==2
    assert result['tcc_mean_overpass'].notna().all()

def test_era5_land_cumulative_ssrd_is_not_summed_repeatedly(tmp_path):
    times = pd.date_range('2024-08-01',periods=73,freq='h')
    # CDS ERA5-Land accumulation resets at 01 UTC; 00 UTC is yesterday's 24h total.
    # Constant flux contributes 1 MJ each hour => 24 MJ per complete day.
    accumulation = np.where(times.hour==0,24,times.hour).astype(float)*1e6
    path = write_dataset(tmp_path/'cds-land.nc',times,{'t2m':[293.15],'ssrd':accumulation})
    result = calculate_daily_metrics(path,area=AREA)
    assert result.loc[result.date=='2024-08-02','ssrd_mj_m2'].item()==24.0

def test_era5_backend_pins_a_reanalysis_model():
    response = Mock()
    response.json.return_value={'hourly':{'time':[]}}
    with patch('requests.get', return_value=response) as get:
        fetch_open_meteo_archive(2024,8,area=AREA)
    assert get.call_args.kwargs['params'].get('models') in {'era5','era5_land','era5_seamless'}

def test_process_cli_accepts_custom_aoi(tmp_path):
    path = write_dataset(tmp_path/'paris.nc',pd.date_range('2024-08-01',periods=25,freq='h'),{'t2m':[293.15], 'tp':[0.]},lat=48.75,lon=2.25)
    process_era5_main([str(path),'--filter',str(SOURCE/'filters/metafilter.json'),'--bbox','2','48','3','49'])

def test_analog_rejects_missing_candidate_date():
    model = AnalogModel(features=['a'],metric='euclidean').fit({2020:pd.DataFrame({'date':['2020-01-01'],'a':[0.]}),2021:pd.DataFrame({'date':[None,'2021-01-01'],'a':[0.,1.]})})
    matches=model.query(reference_year=2020,reference_date='2020-01-01')
    assert all(m.date != 'nan' for m in matches)

def test_failed_refit_preserves_old_result_or_invalidates():
    old={year:pd.DataFrame({'date':[f'{year}-01-01',f'{year}-01-02'],'a':[0.,1.],'b':[0.,1.]}) for year in [2020,2021]}
    model=AnalogModel(features=['a','b'],metric='euclidean').fit(old)
    before=model.query(reference_year=2020,reference_date='2020-01-01')[0]
    new={2020:pd.DataFrame({'date':['2020-01-01','2020-01-02'],'a':[0.,100.],'b':[0.,100.]}),2021:pd.DataFrame({'date':['2021-01-01','2021-01-02'],'a':[100.,200.],'b':[100.,200.]}),2022:pd.DataFrame()}
    with pytest.raises(ValueError):
        model.fit(new)
    try:
        after=model.query(reference_year=2020,reference_date='2020-01-01')[0]
    except RuntimeError:
        return
    assert after.distance==before.distance
