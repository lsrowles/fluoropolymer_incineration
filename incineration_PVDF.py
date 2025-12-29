#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 2:32:06 2025

@author: Tushar
"""

import numpy as np
import pandas as pd
import lhs

# percentiles for uncertainty summaries
output_perc_mid = 50
output_perc_low = 5
output_perc_high = 95

# load assumptions
general_assumptions = pd.read_excel('assumptions.xlsx', sheet_name='General', index_col='Parameter')
design_assumptions  = pd.read_excel('assumptions.xlsx', sheet_name='design',  index_col='Parameter')
LCA_assumptions     = pd.read_excel('assumptions.xlsx', sheet_name='LCA',     index_col='Parameter')

# Monte Carlo runs
n_samples = int(general_assumptions.loc['n_samples', 'expected'])

# correlation scaffolding
correlation_distributions = np.full((n_samples, n_samples), np.nan)
correlation_parameters = np.full((n_samples, 1), np.nan).tolist()

# general inputs
discount_rate, correlation_distributions, correlation_parameters = lhs.lhs_distribution(general_assumptions.loc['discount_rate'], correlation_distributions, correlation_parameters, n_samples)
analysis_period, correlation_distributions, correlation_parameters = lhs.lhs_distribution(general_assumptions.loc['analysis_period'], correlation_distributions, correlation_parameters, n_samples)
tax_rate, correlation_distributions, correlation_parameters = lhs.lhs_distribution(general_assumptions.loc['tax_rate'], correlation_distributions, correlation_parameters, n_samples)

# LCA unit factors used in PVDF block
transport_unit_LCA      = np.reshape(LCA_assumptions.loc['transport', :].iloc[2:].to_numpy(dtype=float), (1, -1))
stainless_steel_unit_LCA= np.reshape(LCA_assumptions.loc['stainless_steel', :].iloc[2:].to_numpy(dtype=float), (1, -1))
electricity_unit_LCA    = np.reshape(LCA_assumptions.loc['electricity', :].iloc[2:].to_numpy(dtype=float), (1, -1))
concrete_unit_LCA       = np.reshape(LCA_assumptions.loc['concrete', :].iloc[2:].to_numpy(dtype=float), (1, -1))
sand_LCA                = np.reshape(LCA_assumptions.loc['sand', :].iloc[2:].to_numpy(dtype=float), (1, -1))
lime_LCA                = np.reshape(LCA_assumptions.loc['lime', :].iloc[2:].to_numpy(dtype=float), (1, -1))
incineration_LCA        = np.reshape(LCA_assumptions.loc['incineration', :].iloc[2:].to_numpy(dtype=float), (1, -1))
PM_to_air_LCA           = np.reshape(LCA_assumptions.loc['PM_to_air', :].iloc[2:].to_numpy(dtype=float), (1, -1))
NOx_to_air_LCA          = np.reshape(LCA_assumptions.loc['NOx_to_air', :].iloc[2:].to_numpy(dtype=float), (1, -1))
HF_to_air_LCA           = np.reshape(LCA_assumptions.loc['HF_to_air', :].iloc[2:].to_numpy(dtype=float), (1, -1))
CO2_fossil_to_air_LCA   = np.reshape(LCA_assumptions.loc['CO2_fossil_to_air', :].iloc[2:].to_numpy(dtype=float), (1, -1))
fuel_LCA = np.reshape(LCA_assumptions.loc['methane_to_air',:].iloc[2:].to_numpy(dtype=float), (1,-1))

# design inputs needed for PVDF block
diesel_transport, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['diesel_transport'], correlation_distributions, correlation_parameters, n_samples)
driver_wages_US, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['driver_wages_US'], correlation_distributions, correlation_parameters, n_samples)
average_speed, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['average_speed'], correlation_distributions, correlation_parameters, n_samples)
diesel_cost_US, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['diesel_cost_US'], correlation_distributions, correlation_parameters, n_samples)
truck_weight, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['truck_weight'], correlation_distributions, correlation_parameters, n_samples)

total_mass_input_waste, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['total_mass_input_waste'], correlation_distributions, correlation_parameters, n_samples)
mass_fraction_PVDF, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['mass_fraction_PVDF'], correlation_distributions, correlation_parameters, n_samples)

rho_pvdf, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['rho_pvdf'], correlation_distributions, correlation_parameters, n_samples)
rho_metal, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['rho_metal'], correlation_distributions, correlation_parameters, n_samples)

moisture_frac, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['moisture_frac'], correlation_distributions, correlation_parameters, n_samples)
burn_rate, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['burn_rate'], correlation_distributions, correlation_parameters, n_samples)
operation_time, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['operation_time'], correlation_distributions, correlation_parameters, n_samples)
fuel_consumption, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['fuel_consumption'], correlation_distributions, correlation_parameters, n_samples)
fuel_price, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['fuel_price'], correlation_distributions, correlation_parameters, n_samples)

incinerator_weight, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['incinerator_weight'], correlation_distributions, correlation_parameters, n_samples)
steel_fraction, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['steel_fraction'], correlation_distributions, correlation_parameters, n_samples)
concrete_fraction, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['concrete_fraction'], correlation_distributions, correlation_parameters, n_samples)
sand_fluidized_bed_incinerator, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['sand_fluidized_bed_incinerator'], correlation_distributions, correlation_parameters, n_samples)
lime_incin, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['lime_incin'], correlation_distributions, correlation_parameters, n_samples)

PM_to_air_incin, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['PM_to_air_incin'], correlation_distributions, correlation_parameters, n_samples)
NOx_to_air_incin, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['NOx_to_air_incin'], correlation_distributions, correlation_parameters, n_samples)
CO2_to_air_incin_PVDF, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['CO2_to_air_incin_PVDF'], correlation_distributions, correlation_parameters, n_samples)
HF_to_air_incin_PVDF, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['HF_to_air_incin_PVDF'], correlation_distributions, correlation_parameters, n_samples)

pollution_control_factor, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['pollution_control_factor'], correlation_distributions, correlation_parameters, n_samples)

ash_incin, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['ash_incin'], correlation_distributions, correlation_parameters, n_samples)


T_ambient, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['T_ambient'], correlation_distributions, correlation_parameters, n_samples)
T_base, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['T_base'], correlation_distributions, correlation_parameters, n_samples)

electricity_use_incin, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['electricity_use_incin'], correlation_distributions, correlation_parameters, n_samples)
electricity_cost, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['electricity_cost'], correlation_distributions, correlation_parameters, n_samples)

landfill_distance_US, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['landfill_distance_US'], correlation_distributions, correlation_parameters, n_samples)
normalize_factor, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['normalize_factor'], correlation_distributions, correlation_parameters, n_samples)

# capture sampled inputs for sensitivity
def _is_sample_vec(x):
    return isinstance(x, np.ndarray) and x.ndim in (1, 2) and x.shape[0] == n_samples and (x.ndim == 1 or x.shape[1] == 1)

inputs_for_sensitivity = {
    name: np.ravel(val)
    for name, val in locals().items()
    if _is_sample_vec(val)
}

# remove non PVDF fields so they cannot appear in sensitivity
drop_keys = [
    'mass_fraction_PTFE', 'mass_fraction_PCTFE',
    'rho_ptfe', 'rho_pctfe',
    'CO2_fossil_to_air_incin',  # not used directly in PVDF block
    'C2F6_to_air_incin', 'CF4_to_air_incin',
    'CO_to_air_incin',
    'HF_to_air_incin',          # PVDF uses HF_to_air_incin_PVDF
    'Cl_to_air_incin', 'TFA_to_air_incin'
]
for k in drop_keys:
    inputs_for_sensitivity.pop(k, None)

input_name_order = list(inputs_for_sensitivity.keys())

# daily masses in tonne per day
PVDF_mass = total_mass_input_waste * mass_fraction_PVDF
mass_fraction_metal = 1 - mass_fraction_PVDF
metal_mass = total_mass_input_waste * mass_fraction_metal

# total wet mass and dry fraction
solids_mass_total_disposal = total_mass_input_waste
solids_conc_disposal = 1.0 - moisture_frac
solids_mass_disposal = solids_mass_total_disposal * solids_conc_disposal

# component volumes then total volume flow
pvdf_vol_m3_d = (PVDF_mass * 1000.0) / rho_pvdf
metal_vol_m3_d = (metal_mass * 1000.0) / rho_metal
solids_volume_disposal_PVDF = pvdf_vol_m3_d + metal_vol_m3_d

driver_wages = driver_wages_US
diesel_cost = diesel_cost_US
landfill_distance = landfill_distance_US

solids_volume_disposal_incineration_PVDF = solids_volume_disposal_PVDF
solids_mass_disposal_incineration = solids_mass_disposal
solids_mass_total_disposal_incineration = solids_mass_total_disposal

# incineration costs for PVDF
incineration_construction_cost_PVDF = (3e6 * (solids_volume_disposal_incineration_PVDF * normalize_factor) ** 0.271) / normalize_factor
incin_maintenance_PVDF = (6523.9 * (solids_volume_disposal_incineration_PVDF * normalize_factor) + 16047) / normalize_factor
incin_material_PVDF = (15360 + (solids_volume_disposal_incineration_PVDF * normalize_factor) ** 0.2703) / normalize_factor

# Target Temperature (e.g., 500 or 1050)
T_target = 500 

# 1. Calculate Temperature Scaling Factors
# Fuel scales with Delta T (heat required)
fuel_scaling_factor = (T_target - T_ambient) / (T_base - T_ambient)

# Electricity scales with Absolute Temp (Kelvin) for fan/motor load 
elec_scaling_factor = (T_target + 273.15) / (T_base + 273.15)
number_of_incinerators = np.ceil((total_mass_input_waste * 1000) / (burn_rate * operation_time))

fuel_demand_l_hr = fuel_consumption * fuel_scaling_factor
fuel_demand = fuel_demand_l_hr * operation_time * 365 * number_of_incinerators * 0.264 # gallon/yr #propane
fuel_cost_annual = fuel_demand * fuel_price

incin_energy_cost_PVDF = solids_mass_disposal_incineration * electricity_use_incin * electricity_cost * 365
incin_capital_cost_annualized_PVDF = incineration_construction_cost_PVDF * ((discount_rate * (1 + discount_rate) ** analysis_period) / ((1 + discount_rate) ** analysis_period - 1))

number_of_trucks_solids_incineration = (solids_mass_total_disposal_incineration / 100) / truck_weight
transport_solids_incineration_annual_cost = (
    (((landfill_distance / average_speed) * driver_wages * number_of_trucks_solids_incineration)
      + (landfill_distance * diesel_transport * (solids_mass_total_disposal_incineration / 100) * diesel_cost)) * 2 * 365
)

# normalize cost per tonne treated
capital_cost_PVDF = incin_capital_cost_annualized_PVDF / (total_mass_input_waste * 365)
OM_cost_PVDF = (incin_maintenance_PVDF + incin_material_PVDF + transport_solids_incineration_annual_cost + fuel_cost_annual) / (total_mass_input_waste * 365)
energy_cost_PVDF = incin_energy_cost_PVDF / (total_mass_input_waste * 365)
total_cost_PVDF = capital_cost_PVDF + OM_cost_PVDF + energy_cost_PVDF

# LCA construction
steel_incin_construction    = (incinerator_weight * steel_fraction)    / (incinerator_weight / 1000)
concrete_incin_construction = (incinerator_weight * concrete_fraction) / (incinerator_weight / 1000)
incineration_construction_LCA = (
      solids_mass_disposal_incineration * steel_incin_construction * stainless_steel_unit_LCA
    + solids_mass_disposal_incineration * concrete_incin_construction * concrete_unit_LCA
)
incineration_construction_LCA_annual = incineration_construction_LCA / analysis_period

incineration_annual_LCA = (
    (solids_mass_disposal_incineration * sand_fluidized_bed_incinerator * sand_LCA)
  + (solids_mass_disposal_incineration * lime_incin * lime_LCA)
  + (solids_mass_disposal_incineration * ash_incin * incineration_LCA)
  + (fuel_demand * 3.785 * 0.493 * fuel_LCA / 365)
) * 365

# shared pieces for impacts normalization
transport_solids_incineration_annual_LCA = (landfill_distance * transport_unit_LCA) * 2 * number_of_trucks_solids_incineration * 365
total_impacts_capital = incineration_construction_LCA_annual / (total_mass_input_waste * 365)
total_impacts_OM      = (incineration_annual_LCA + transport_solids_incineration_annual_LCA) / (total_mass_input_waste * 365)
total_impacts_energy  = ((solids_mass_disposal_incineration * electricity_use_incin * electricity_unit_LCA) * 365) / (total_mass_input_waste * 365)

# PVDF direct emissions to air
direct_emissions_from_waste_PVDF = (
    solids_mass_disposal_incineration * PM_to_air_incin  * PM_to_air_LCA
  + solids_mass_disposal_incineration * NOx_to_air_incin * NOx_to_air_LCA
  + solids_mass_disposal_incineration * mass_fraction_PVDF * pollution_control_factor * CO2_to_air_incin_PVDF * CO2_fossil_to_air_LCA
  + solids_mass_disposal_incineration * mass_fraction_PVDF * pollution_control_factor * HF_to_air_incin_PVDF   * HF_to_air_LCA
) * 365

total_impacts_direct_emissions_from_waste_PVDF = direct_emissions_from_waste_PVDF / (total_mass_input_waste * 365)
total_impacts_PVDF = total_impacts_capital + total_impacts_OM + total_impacts_energy + total_impacts_direct_emissions_from_waste_PVDF

# results for PVDF
impact_cols = ['AP','GWP','EFW','EP','HTC','HTNC','ODP','PMFP','MIR']
gwp_PVDF = np.ravel(np.asarray(total_impacts_PVDF)[:, 1])

df_results_PVDF = pd.DataFrame({
    'cost_incineration_PVDF': np.ravel(total_cost_PVDF),
    'impacts_GWP_incineration_PVDF': gwp_PVDF
})

df_impacts_PVDF = pd.DataFrame(total_impacts_PVDF, columns=impact_cols)
df_direct_emissions_impacts_PVDF = pd.DataFrame(total_impacts_direct_emissions_from_waste_PVDF, columns=impact_cols)

df_total_impacts_capital = pd.DataFrame(total_impacts_capital, columns=impact_cols)
df_total_impacts_OM      = pd.DataFrame(total_impacts_OM,      columns=impact_cols)
df_total_impacts_energy  = pd.DataFrame(total_impacts_energy,  columns=impact_cols)

df_cost = pd.DataFrame({
    'capital_cost_PVDF': np.ravel(capital_cost_PVDF),
    'OM_cost_PVDF': np.ravel(OM_cost_PVDF),
    'energy_cost_PVDF': np.ravel(energy_cost_PVDF),
    'total_cost_PVDF': np.ravel(total_cost_PVDF),
})

# sensitivity inputs dataframe
dfinputs = pd.DataFrame(inputs_for_sensitivity)[input_name_order]

# PVDF sensitivity using only PVDF impacts
df_impacts = df_impacts_PVDF.copy()
sensitivity_cost_PVDF = dfinputs.corrwith(df_cost['total_cost_PVDF'], method='spearman')
sensitivity_PVDF = pd.DataFrame({
    'cost': sensitivity_cost_PVDF,
    'AP':   dfinputs.corrwith(df_impacts.iloc[:, 0], method='spearman'),
    'GWP':  dfinputs.corrwith(df_impacts.iloc[:, 1], method='spearman'),
    'EFW':  dfinputs.corrwith(df_impacts.iloc[:, 2], method='spearman'),
    'EP':   dfinputs.corrwith(df_impacts.iloc[:, 3], method='spearman'),
    'HTC':  dfinputs.corrwith(df_impacts.iloc[:, 4], method='spearman'),
    'HTNC': dfinputs.corrwith(df_impacts.iloc[:, 5], method='spearman'),
    'ODP':  dfinputs.corrwith(df_impacts.iloc[:, 6], method='spearman'),
    'PMFP': dfinputs.corrwith(df_impacts.iloc[:, 7], method='spearman'),
    'MIR':  dfinputs.corrwith(df_impacts.iloc[:, 8], method='spearman'),
})

# write outputs
with pd.ExcelWriter("incin_PVDF.xlsx") as writer:
    df_results_PVDF.to_excel(writer, sheet_name="results", index=False)

    df_total_impacts_capital.to_excel(writer, sheet_name="capital_impacts", index=False)
    df_total_impacts_OM.to_excel(writer,      sheet_name="OM_impacts",      index=False)
    df_total_impacts_energy.to_excel(writer,  sheet_name="energy_impacts",  index=False)

    df_direct_emissions_impacts_PVDF.to_excel(writer, sheet_name="direct_emissions_impacts_PVDF", index=False)
    df_impacts_PVDF.to_excel(writer, sheet_name="impacts_PVDF", index=False)

    df_cost.to_excel(writer, sheet_name="cost", index=False)
    sensitivity_PVDF.to_excel(writer, sheet_name="sensitivity_PVDF")
    dfinputs.to_excel(writer, sheet_name="inputs")
