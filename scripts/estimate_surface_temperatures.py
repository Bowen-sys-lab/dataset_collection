"""Estimate exterior surface temperatures before authoring a thermal scene.

Why this exists: a prescribed temperature per surface class is only defensible if it can be
re-derived.  This script solves the steady-state surface energy balance

    h (T_s - T_air) + eps sigma T_s^4 = alpha * G + eps * Q_LW_down + Q_latent

for each class, where:

* ``alpha`` (solar absorptivity) and ``eps`` (LWIR emissivity) come from the project material
  library, not from this file;
* ``T_air``, ``wind_speed``, ``cloud_fraction`` and the vapour pressure come from **the weather
  file the scene actually uses**;
* ``Q_LW_down`` is the plugin's own broadband sky model (``irsim.thermal.longwave``, Brunt),
  so the sky the surface radiates to is the same sky the renderer uses;
* ``h`` is the plugin's own convection model (``irsim.thermal.convection``), so wind coupling is
  identical to the simulation.

This is the sol-air temperature method (ISO 13790 / ASHRAE Fundamentals) solved for the surface
temperature, with the simulator's atmosphere and convection rather than a hand-picked constant.

Usage:
    python scripts/estimate_surface_temperatures.py --scenario night
    python scripts/estimate_surface_temperatures.py --scenario day --out validation/x.json
"""
from __future__ import annotations

import argparse
import sys
import json
from datetime import datetime
from pathlib import Path

SIGMA = 5.670374419e-8
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
VENDOR = ROOT / 'third_party' / 'isaac-thermal-camera-simulation'

#: class -> (material, sky/ground view factor, incident solar W/m2 at the reference G, latent W/m2)
#: G is the irradiance ON THAT SURFACE, not the horizontal global value.
CLASSES = {
    'wall':          ('brick_wall',                 0.5, 500.0,   0.0),
    'road':          ('asphalt_dry',                1.0, 900.0,   0.0),
    # Pavement under a tree canopy never receives direct sun all day: same balance,
    # zero solar load.  This is what makes the shade read as a cold patch at dusk.
    'road_shaded':   ('asphalt_dry',                1.0,   0.0,   0.0),
    'ground_shaded': ('concrete',                   1.0,   0.0,   0.0),
    'ground':        ('concrete',                   1.0, 900.0,   0.0),
    'vegetation':    ('vegetation_leaf',            0.5, 500.0, 200.0),
    'window':        ('glass_windshield',           0.5, 500.0,   0.0),
    'billboard':     ('painted_composite',          0.5, 500.0,   0.0),
    'vehicle':       ('car_paint_black',            0.5, 500.0,   0.0),
    # Parked under the street trees / building shadow: no direct sun on the body.
    'vehicle_shaded': ('car_paint_black',           0.5,   0.0,   0.0),
    'lamp':          ('aircraft_aluminium_painted', 0.5, 300.0,   0.0),
    'prop_plastic':  ('abs_plastic_white',          0.5, 500.0,   0.0),
    'prop_metal':    ('aircraft_aluminium_painted', 0.5, 300.0,   0.0),
}

#: The weather file each scenario runs on, and the instant the snapshot describes.
SCENARIOS = {
    'day': {
        'weather_file': VENDOR / 'data' / 'weather' / 'clear_midlat_summer_48h.csv',
        'at_utc': '2024-06-21T10:00:00Z',   # 12:00 local (UTC+2), sun near its highest
        'irradiance_horizontal_w_m2': None,  # taken from the file's dni/dhi
        'solar_elevation_deg': 60.0,
        'note': 'irsim synthetic clear mid-latitude summer day',
    },
    'night': {
        'weather_file': ROOT / 'factory' / 'model02_thermal_v4' / 'night_weather.csv',
        'at_utc': '2026-10-18T12:00:00Z',
        'irradiance_horizontal_w_m2': 0.0,
        'solar_elevation_deg': 0.0,
        'note': 'model02 synthetic dry operating night',
    },
    # Dusk is not a steady state: the sun has set, but massive surfaces are still releasing the
    # heat they stored during the day.  It is derived from the two other scenarios with a lumped
    # cooling model (see ``estimate('dusk')``) rather than solved directly.
    'dusk': {
        'weather_file': VENDOR / 'data' / 'weather' / 'clear_midlat_summer_48h.csv',
        'at_utc': '2024-06-21T17:25:00Z',   # 19:25 local, sun at 15 deg (WNW)
        'irradiance_horizontal_w_m2': 0.0,
        'solar_elevation_deg': 15.0,
        'site': {'latitude_deg': 48.1, 'longitude_deg': 11.6},
        'integration_start_utc': '2024-06-21T04:00:00Z',
        'integration_step_s': 30.0,
        'note': 'golden hour: sun 15 deg, surfaces integrated through the day',
    },
}


def _solve(alpha, eps, irradiance, latent, view_factor, sample, library):
    """Fixed-point solve of the balance; h and Q_LW come from the plugin's own models."""
    from irsim.thermal.convection import convection_coefficient
    from irsim.thermal.longwave import longwave_down_from_sample

    t_air = float(sample.t_air_k)
    q_lw = float(longwave_down_from_sample(sample, view_factor, t_surround_k=t_air))
    t_s = t_air
    for _ in range(200):
        h = float(convection_coefficient(t_s - t_air, float(sample.wind_speed_m_s)))
        absorbed = alpha * irradiance + eps * q_lw - latent
        t_new = (absorbed - eps * SIGMA * t_s ** 4) / h + t_air
        if abs(t_new - t_s) < 1e-5:
            t_s = t_new
            break
        t_s = t_new
    h_final = float(convection_coefficient(t_s - t_air, float(sample.wind_speed_m_s)))
    return t_s, q_lw, h_final


def _estimate_dusk() -> dict:
    """Golden hour: integrate each surface through the day up to a 15 degree sun.

    A steady balance cannot describe late afternoon, because the answer depends on the whole day
    the surface has just lived through.  Each class is integrated from 04:00 UTC with a lumped
    energy balance,

        C dT/dt = alpha G(t) + eps Q_LW(t) - eps sigma T^4 - h(T)(T - T_air(t)) - Q_latent

    using the scene's own weather file (irradiance from its DNI/DHI and the solar position), the
    plugin's sky and convection models, and the material's heat capacity.  Heavy surfaces arrive
    warm, thin ones have already tracked the air -- which is what the image should show.
    """
    import math
    import sys
    from datetime import datetime, timedelta, timezone

    if str(VENDOR / 'src') not in sys.path:
        sys.path.insert(0, str(VENDOR / 'src'))
    from irsim.thermal.convection import convection_coefficient
    from irsim.thermal.longwave import longwave_down_from_sample
    from irsim.thermal.solar import sun_position_utc
    from irsim.thermal.weather_io import load_weather_csv

    from scripts.project_materials import load_project_material_library

    scenario = SCENARIOS['dusk']
    start = datetime.fromisoformat(scenario['integration_start_utc'].replace('Z', '+00:00'))
    target = datetime.fromisoformat(scenario['at_utc'].replace('Z', '+00:00'))
    site = scenario['site']
    series = load_weather_csv(scenario['weather_file'])
    library = load_project_material_library(ROOT / 'configs' / 'materials')
    step_s = float(scenario['integration_step_s'])
    steps = max(1, int((target - start).total_seconds() // step_s))

    rows = {}
    for name, (material, view_factor, g_scale, latent) in CLASSES.items():
        spec = library[material].spec
        eps = library[material].band_properties('lwir').emissivity
        alpha = spec.thermal.solar_absorptivity
        capacity = spec.thermal.heat_capacity_j_m2_k
        sample0 = series.at_datetime(start)
        temperature = float(sample0.t_air_k)
        def steady_temperature(irradiance_now, q_lw_now, h_now, air_now, latent_now):
            low, high = 200.0, 400.0
            for _ in range(60):
                mid = 0.5 * (low + high)
                balance = (alpha * irradiance_now + eps * q_lw_now - latent_now
                           - eps * SIGMA * mid ** 4 - h_now * (mid - air_now))
                if balance > 0.0:
                    low = mid
                else:
                    high = mid
            return 0.5 * (low + high)

        when = start
        dt = step_s
        while when < target:
            sample = series.at_datetime(when)
            air = float(sample.t_air_k)
            elevation = float(sun_position_utc(site['latitude_deg'], site['longitude_deg'],
                                              when).elevation_deg)
            horizontal = (float(sample.dni_w_m2) * max(0.0, math.sin(math.radians(elevation)))
                          + float(sample.dhi_w_m2))
            # g_scale is the reference irradiance for this class (900 horizontal, 500 vertical),
            # not a fraction: normalise against that reference.
            irradiance = horizontal * (g_scale / 900.0)
            q_lw = float(longwave_down_from_sample(sample, view_factor, t_surround_k=air))
            h = float(convection_coefficient(temperature - air, float(sample.wind_speed_m_s)))
            # Transpiration cannot exceed the energy the leaf actually absorbs, otherwise the
            # leaf sinks far below air temperature.
            latent_now = (min(latent, 0.6 * alpha * irradiance + 30.0)
                          if irradiance > 0.0 else 0.0)
            equilibrium = steady_temperature(irradiance, q_lw, h, air, latent_now)
            # First-order lag towards the instantaneous equilibrium: stable at any step size.
            tau = capacity / (h + 4.0 * eps * SIGMA * temperature ** 3)
            temperature += (equilibrium - temperature) * (-math.expm1(-dt / tau))
            when += timedelta(seconds=dt)
        sample = series.at_datetime(target)
        rows[name] = {
            'material': material,
            'solar_absorptivity': round(alpha, 3),
            'lwir_emissivity': round(eps, 3),
            'heat_capacity_j_m2_k': round(capacity, 1),
            'surface_temperature_c': round(temperature - 273.15, 1),
            'above_air_k': round(temperature - float(sample.t_air_k), 1),
        }
    sample = series.at_datetime(target)
    return {
        'scenario': 'dusk',
        'weather_file': str(scenario['weather_file']),
        'at_utc': scenario['at_utc'],
        'weather_sample': {
            'air_temperature_c': round(float(sample.t_air_k) - 273.15, 2),
            'relative_humidity': round(float(sample.rh_fraction), 3),
            'vapour_pressure_hpa': round(float(sample.vapour_pressure_hpa), 2),
            'wind_speed_m_s': round(float(sample.wind_speed_m_s), 2),
            'cloud_fraction': round(float(sample.cloud_fraction), 2),
            'dni_w_m2': round(float(sample.dni_w_m2), 1),
            'dhi_w_m2': round(float(sample.dhi_w_m2), 1),
        },
        'solar_elevation_deg': scenario['solar_elevation_deg'],
        'method': ('transient lumped balance integrated from 04:00 UTC to the target time with the '
                   'scene weather file, plugin sky/convection and material heat capacity'),
        'classes': rows,
    }


def estimate(scenario_name: str) -> dict:
    if scenario_name == 'dusk':
        return _estimate_dusk()
    import sys

    if str(VENDOR / 'src') not in sys.path:
        sys.path.insert(0, str(VENDOR / 'src'))
    from irsim.thermal.weather_io import load_weather_csv

    from scripts.project_materials import load_project_material_library

    scenario = SCENARIOS[scenario_name]
    series = load_weather_csv(scenario['weather_file'])
    sample = series.at_datetime(datetime.fromisoformat(scenario['at_utc'].replace('Z', '+00:00')))
    library = load_project_material_library(ROOT / 'configs' / 'materials')

    if scenario['irradiance_horizontal_w_m2'] is None:
        # Global horizontal = direct normal * cos(zenith) + diffuse horizontal.
        import math
        cos_zenith = math.cos(math.radians(90.0 - scenario['solar_elevation_deg']))
        g_horizontal = float(getattr(sample, 'dni_w_m2', 0.0)) * cos_zenith + \
            float(getattr(sample, 'dhi_w_m2', 0.0))
    else:
        g_horizontal = float(scenario['irradiance_horizontal_w_m2'])

    rows = {}
    for name, (material, view_factor, g_surface_ref, latent) in CLASSES.items():
        spec = library[material].spec
        alpha = spec.thermal.solar_absorptivity
        eps = library[material].band_properties('lwir').emissivity
        # The surface irradiance scales with the scenario's horizontal global value.
        g_surface = g_surface_ref * g_horizontal / 900.0
        latent_effective = latent if g_surface > 0.0 else 0.0
        t_s, q_lw, h = _solve(alpha, eps, g_surface, latent_effective, view_factor,
                              sample, library)
        rows[name] = {
            'material': material,
            'solar_absorptivity': round(alpha, 3),
            'lwir_emissivity': round(eps, 3),
            'view_factor': view_factor,
            'incident_solar_w_m2': round(g_surface, 1),
            'longwave_down_w_m2': round(q_lw, 1),
            'convection_w_m2k': round(h, 2),
            'latent_w_m2': latent_effective,
            'surface_temperature_c': round(t_s - 273.15, 1),
            'above_air_k': round(t_s - sample.t_air_k, 1),
        }
    return {
        'scenario': scenario_name,
        'weather_file': str(scenario['weather_file']),
        'at_utc': scenario['at_utc'],
        'weather_sample': {
            'air_temperature_c': round(float(sample.t_air_k) - 273.15, 2),
            'relative_humidity': round(float(sample.rh_fraction), 3),
            'vapour_pressure_hpa': round(float(sample.vapour_pressure_hpa), 2),
            'wind_speed_m_s': round(float(sample.wind_speed_m_s), 2),
            'cloud_fraction': round(float(sample.cloud_fraction), 2),
            'dni_w_m2': round(float(getattr(sample, 'dni_w_m2', 0.0)), 1),
            'dhi_w_m2': round(float(getattr(sample, 'dhi_w_m2', 0.0)), 1),
        },
        'global_horizontal_w_m2': round(g_horizontal, 1),
        'method': ('steady-state surface balance h(Ts-Tair) + eps*sigma*Ts^4 = alpha*G + '
                   'eps*Q_LW + Q_latent, with Q_LW from irsim.thermal.longwave (Brunt, cloud '
                   'corrected) and h from irsim.thermal.convection; equivalent to sol-air '
                   'temperature solved for Ts'),
        'classes': rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', default='day', choices=sorted(SCENARIOS))
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()
    report = estimate(args.scenario)
    out = args.out or ROOT / 'validation' / f'surface-temperatures-{args.scenario}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    w = report['weather_sample']
    print(f"scenario {report['scenario']}: T_air {w['air_temperature_c']} C, RH {w['relative_humidity']}, "
          f"wind {w['wind_speed_m_s']} m/s, cloud {w['cloud_fraction']}, "
          f"G_h {report.get('global_horizontal_w_m2', 0.0)} W/m2")
    dusk = report['scenario'] == 'dusk'
    print(f"{'class':14} {'material':30} {'Ts C':>7} {'dT/K':>6} "
          f"{'tau/min' if dusk else 'h':>8} {'note':>18}")
    for name, row in report['classes'].items():
        if dusk:
            extra = f"C={row['heat_capacity_j_m2_k']:>10.0f} J/m2K"
        else:
            extra = f"{row['convection_w_m2k']:>8.2f} LWdn {row['longwave_down_w_m2']:>6.1f}"
        print(f"{name:14} {row['material']:30} {row['surface_temperature_c']:>7.1f} "
              f"{row['above_air_k']:>+6.1f} {extra}")
    print(f"\nwritten: {out}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
