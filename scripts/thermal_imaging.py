"""Describe the actual ISP and expose its input for diagnostics, without changing data."""
import hashlib
import json

import numpy as np


def display_input_dn(config, dn16):
    from irsim.detector.quantise import quantise
    flat = config.flat_field
    if config.chain is not None and config.chain.display_nuc is not None:
        flat = config.chain.display_nuc
    return np.array(dn16, copy=True) if flat is None else quantise(
        flat.apply(dn16), config.sensor.sensor.fpa.bit_depth)


def display_from_outputs(config, outputs, display_profile=None):
    """Use unquantised signal for project flat-field display correction.

    The vendor pipeline also has a flat-field path, but it receives quantised DN16. Applying the
    same coefficients to ``signal_dn`` here avoids turning sub-LSB correction error into visible
    rings. Physical output planes remain untouched; only ``display8`` is replaced.
    """
    from dataclasses import replace
    from irsim.detector.quantise import quantise
    from irsim.isp.display import run_display_branch
    if display_profile is not None and display_profile.mode == 'fixed_apparent_temperature':
        from scripts.thermal_display import fixed_temperature_display
        if outputs.apparent_t is None:
            raise ValueError('fixed apparent-temperature display requires apparent_t output')
        display = fixed_temperature_display(
            outputs.apparent_t,
            display_profile.temperature_min_c,
            display_profile.temperature_max_c,
            polarity=display_profile.polarity or 'white_hot',
            palette=display_profile.palette or 'gray',
        )
        return replace(
            outputs,
            display8=display,
            isp_hash=display_profile.metadata()['profile_sha256'],
        )
    flat = config.flat_field
    if display_profile is None and (flat is None or outputs.display8 is None):
        return outputs
    if display_profile is not None and config.chain is not None and config.chain.display_nuc is not None:
        flat = config.chain.display_nuc
    corrected = quantise(
        outputs.signal_dn if flat is None else flat.apply(outputs.signal_dn),
        config.sensor.sensor.fpa.bit_depth,
    )
    isp = config.sensor.sensor.isp
    if display_profile is not None:
        from scripts.thermal_display import profile_isp
        isp = profile_isp(isp, display_profile)
    display = run_display_branch(corrected, isp, config.sensor.sensor.fpa.bit_depth)
    pixels = display.display8
    if outputs.apparent_t is not None:
        pixels[~np.isfinite(outputs.apparent_t)] = (255, 0, 255, 255)
    return replace(outputs, display8=pixels, isp_hash=display.isp_hash)


def imaging_metadata(config, state, display_profile=None):
    from irsim.pipeline.core import RADIOMETRIC_RANGE_K
    sensor = config.sensor.sensor
    flat = config.flat_field
    settings = sensor.isp.model_dump(mode='json')
    effective_settings = settings
    fixed_temperature = display_profile is not None and display_profile.mode == 'fixed_apparent_temperature'
    if display_profile is not None:
        from scripts.thermal_display import profile_isp
        effective_settings = (display_profile.metadata() if fixed_temperature else
                              profile_isp(sensor.isp, display_profile).model_dump(mode='json'))
    coeff_hash = None if flat is None else hashlib.sha256(
        flat.gain.tobytes() + flat.offset.tobytes() + repr(flat.pedestal).encode('ascii')).hexdigest()
    profile_metadata = None if display_profile is None else display_profile.metadata()
    return {
        'flat_field_enabled': flat is not None,
        'flat_field_scope': 'display branch only, before AGC; raw DN/radiance/apparent_t unchanged',
        'flat_field_reference': None if flat is None else {
            'method': 'vendor two synthetic noiseless blackbodies, not real instrument calibration',
            'low_temperature_k': RADIOMETRIC_RANGE_K[0],
            'high_temperature_k': sensor.fpa.gain_ceiling_k or RADIOMETRIC_RANGE_K[1],
            'housing_temperature_k': config.t_housing_cal_k,
            'coefficients_sha256': coeff_hash,
            'gain_range': [float(flat.gain.min()), float(flat.gain.max())],
        },
        'vignetting_cos4': sensor.optics.vignetting_cos4,
        'horizontal_fov_degrees': float(np.degrees(2*np.arctan(
            sensor.fpa.width * sensor.fpa.pitch_um * 1e-3 / (2*sensor.optics.focal_length_mm)))),
        'housing_temperature_k': float(state.housing_temp_k),
        'configured_housing_mode': sensor.optics.housing_temp_mode,
        'sensor_chain_attached': config.chain is not None,
        'configured_nuc_mode': sensor.nuc.mode,
        'shutter_nuc_active': bool(config.chain is not None and config.chain.ffc.shutters),
        'display_correction_source': 'none_fixed_apparent_temperature' if fixed_temperature else
            ('shutter_refreshed' if config.chain is not None and
            config.chain.display_nuc is not None else ('factory_flat_field' if flat is not None else 'none')),
        'sensor_seed': config.sensor_seed,
        'vendor_isp_parameters': settings,
        'display_parameters': effective_settings,
        'display_parameters_sha256': hashlib.sha256(json.dumps(effective_settings, sort_keys=True).encode()).hexdigest(),
        'display_profile': profile_metadata,
        'display_input_quantity': 'apparent_t (K)' if fixed_temperature else 'flat-field-corrected detector signal DN',
        'project_display_flat_field_applied': bool(not fixed_temperature and flat is not None),
        'display_meaning': ('Fixed apparent-temperature visualization; separate from vendor sensor ISP'
                            if fixed_temperature else 'Vendor detector-signal ISP with project display correction'),
        'temperature_meaning': 'apparent temperature (K), not true surface temperature',
    }
