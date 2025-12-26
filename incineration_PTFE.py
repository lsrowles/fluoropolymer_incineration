#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 2:322:06 2025

@author: Tushar
"""

import numpy as np
import pandas as pd
import copy
import lhs
import math
from spearman_matrix import spearman_matrix
import matplotlib.pyplot as plt

# specity percentiles to use when extracting mid, low, and high results from uncertainty analysis
output_perc_mid = 50
output_perc_low = 5
output_perc_high = 95


# general parameters - import spreadsheet tabs as dataframes
general_assumptions = pd.read_excel('assumptions.xlsx', sheet_name = 'General', index_col = 'Parameter')
design_assumptions = pd.read_excel('assumptions.xlsx', sheet_name = 'design', index_col='Parameter')
LCA_assumptions = pd.read_excel('assumptions.xlsx', sheet_name = 'LCA', index_col='Parameter')

# number of Monte Carlo runs
n_samples = int(general_assumptions.loc['n_samples','expected'])

# create empty datasets to eventually store data for sensitivity analysis (Spearman's coefficients)
correlation_distributions = np.full((n_samples, n_samples), np.nan)
correlation_parameters = np.full((n_samples, 1), np.nan)
correlation_parameters = correlation_parameters.tolist()

# general inputs
discount_rate, correlation_distributions, correlation_parameters = lhs.lhs_distribution(general_assumptions.loc['discount_rate'], correlation_distributions, correlation_parameters, n_samples)
analysis_period, correlation_distributions, correlation_parameters  = lhs.lhs_distribution(general_assumptions.loc['analysis_period'],correlation_distributions, correlation_parameters, n_samples)
tax_rate, correlation_distributions, correlation_parameters = lhs.lhs_distribution(general_assumptions.loc['tax_rate'], correlation_distributions, correlation_parameters, n_samples)


# LCA inputs (includes factors for all TRACI impact categories)
transport_unit_LCA = np.reshape(LCA_assumptions.loc['transport',:].iloc[2:].to_numpy(dtype=float), (1,-1))
stainless_steel_unit_LCA = np.reshape(LCA_assumptions.loc['stainless_steel',:].iloc[2:].to_numpy(dtype=float), (1,-1))
electricity_unit_LCA = np.reshape(LCA_assumptions.loc['electricity',:].iloc[2:].to_numpy(dtype=float), (1,-1))
concrete_unit_LCA = np.reshape(LCA_assumptions.loc['concrete',:].iloc[2:].to_numpy(dtype=float), (1,-1))
diesel_transport_LCA = np.reshape(LCA_assumptions.loc['diesel_transport',:].iloc[2:].to_numpy(dtype=float), (1,-1))
sand_LCA = np.reshape(LCA_assumptions.loc['sand',:].iloc[2:].to_numpy(dtype=float), (1,-1))
lime_LCA = np.reshape(LCA_assumptions.loc['lime',:].iloc[2:].to_numpy(dtype=float), (1,-1))
incineration_LCA = np.reshape(LCA_assumptions.loc['incineration',:].iloc[2:].to_numpy(dtype=float), (1,-1))
CO_to_air_LCA = np.reshape(LCA_assumptions.loc['CO_to_air',:].iloc[2:].to_numpy(dtype=float), (1,-1))
PM_to_air_LCA = np.reshape(LCA_assumptions.loc['PM_to_air',:].iloc[2:].to_numpy(dtype=float), (1,-1))
NOx_to_air_LCA = np.reshape(LCA_assumptions.loc['NOx_to_air',:].iloc[2:].to_numpy(dtype=float), (1,-1))

HF_to_air_LCA = np.reshape(LCA_assumptions.loc['HF_to_air',:].iloc[2:].to_numpy(dtype=float), (1,-1))
mercury_to_air_LCA = np.reshape(LCA_assumptions.loc['mercury_to_air',:].iloc[2:].to_numpy(dtype=float), (1,-1))
CO2_fossil_to_air_LCA = np.reshape(LCA_assumptions.loc['CO2_fossil_to_air',:].iloc[2:].to_numpy(dtype=float), (1,-1))
C2F6_to_air_LCA = np.reshape(LCA_assumptions.loc['C2F6_to_air',:].iloc[2:].to_numpy(dtype=float), (1,-1))
CF4_to_air_LCA = np.reshape(LCA_assumptions.loc['CF4_to_air',:].iloc[2:].to_numpy(dtype=float), (1,-1))
fuel_LCA = np.reshape(LCA_assumptions.loc['propane_to_air',:].iloc[2:].to_numpy(dtype=float), (1,-1))

#transport
tansport_emissions_factor_truck, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['tansport_emissions_factor_truck'], correlation_distributions, correlation_parameters, n_samples)

diesel_transport, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['diesel_transport'], correlation_distributions, correlation_parameters, n_samples)
driver_wages_US, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['driver_wages_US'], correlation_distributions, correlation_parameters, n_samples)
average_speed, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['average_speed'], correlation_distributions, correlation_parameters, n_samples)
diesel_cost_US, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['diesel_cost_US'], correlation_distributions, correlation_parameters, n_samples)
truck_weight, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['truck_weight'], correlation_distributions, correlation_parameters, n_samples)

total_mass_input_waste, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['total_mass_input_waste'], correlation_distributions, correlation_parameters, n_samples)
mass_fraction_PTFE, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['mass_fraction_PTFE'], correlation_distributions, correlation_parameters, n_samples)
rho_ptfe, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['rho_ptfe'], correlation_distributions, correlation_parameters, n_samples)
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
CO_to_air_incin, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['CO_to_air_incin'], correlation_distributions, correlation_parameters, n_samples)
PM_to_air_incin, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['PM_to_air_incin'], correlation_distributions, correlation_parameters, n_samples)
NOx_to_air_incin, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['NOx_to_air_incin'], correlation_distributions, correlation_parameters, n_samples)
CO2_fossil_to_air_incin, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['CO2_fossil_to_air_incin'], correlation_distributions, correlation_parameters, n_samples)
C2F6_to_air_incin,correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['C2F6_to_air_incin'], correlation_distributions, correlation_parameters, n_samples)
CF4_to_air_incin,correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['CF4_to_air_incin'], correlation_distributions, correlation_parameters, n_samples)
pollution_control_factor,correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['pollution_control_factor'], correlation_distributions, correlation_parameters, n_samples)


ash_incin, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['ash_incin'], correlation_distributions, correlation_parameters, n_samples)

electricity_use_incin, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['electricity_use_incin'], correlation_distributions, correlation_parameters, n_samples)
electricity_cost, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['electricity_cost'], correlation_distributions, correlation_parameters, n_samples)

#distance to disposal 
landfill_distance_US, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['landfill_distance_US'], correlation_distributions, correlation_parameters, n_samples)

# distribution of solids volume disposal based on MLE and A2O
#solids_volume_disposal_MLE_A2O, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['volume_disposal_MLE_A2O'], correlation_distributions, correlation_parameters, n_samples)
normalize_factor, correlation_distributions, correlation_parameters = lhs.lhs_distribution(design_assumptions.loc['normalize_factor'], correlation_distributions, correlation_parameters, n_samples)

# capture only sampled inputs for sensitivity
def _is_sample_vec(x):
    return isinstance(x, np.ndarray) and x.ndim in (1, 2) and x.shape[0] == n_samples and (x.ndim == 1 or x.shape[1] == 1)

inputs_for_sensitivity = {
    name: np.ravel(val)
    for name, val in locals().items()
    if _is_sample_vec(val)
}
input_name_order = list(inputs_for_sensitivity.keys())  
 #%%
# daily masses in tonne per day
PTFE_mass= total_mass_input_waste * mass_fraction_PTFE # tonne of PTFE/day
mass_fraction_metal = 1 - mass_fraction_PTFE
metal_mass = total_mass_input_waste * mass_fraction_metal # tonne of iron oxide/day



#total wet mass and dry fraction
solids_mass_total_disposal = total_mass_input_waste # tonne/day
solids_conc_disposal = 1.0 - moisture_frac                         # dry matter fraction
solids_mass_disposal = solids_mass_total_disposal * solids_conc_disposal  # tDM/day

#component volumes then total volume flow
ptfe_vol_m3_d  = (PTFE_mass * 1000.0) / rho_ptfe                      # m3/day
metal_vol_m3_d = (metal_mass * 1000.0) / rho_metal                    # m3/day
solids_volume_disposal = ptfe_vol_m3_d + metal_vol_m3_d               # m3/day



cost_incineration = []
impacts_incineration = []

driver_wages = driver_wages_US
diesel_cost = diesel_cost_US
landfill_distance = landfill_distance_US
        
solids_volume_disposal_incineration = solids_volume_disposal 
solids_mass_disposal_incineration =  solids_mass_disposal
solids_mass_total_disposal_incineration = solids_mass_total_disposal



# incineration
incineration_construction_cost = (3E+06 * (solids_volume_disposal_incineration * normalize_factor)**0.271)  /normalize_factor  # $
incin_maintenance = (6523.9 * (solids_volume_disposal_incineration* normalize_factor) + 16047)  /normalize_factor  # $/yr
incin_material = (15360 + (solids_volume_disposal_incineration * normalize_factor)**0.2703)  /normalize_factor  # $/yr

number_of_incinerators = np.ceil ((total_mass_input_waste *1000) / (burn_rate * operation_time)) # number
fuel_demand = fuel_consumption * operation_time * 365 * number_of_incinerators * 0.264 # gallon/yr #propane
fuel_cost_annual = fuel_demand * fuel_price # $/yr

incin_energy_cost = solids_mass_disposal_incineration * electricity_use_incin * electricity_cost * 365 # $/yr
incin_capital_cost_annualized = incineration_construction_cost * ((discount_rate * (1+discount_rate)**analysis_period)
                                      / ((1+discount_rate)**analysis_period - 1))

number_of_trucks_solids_incineration = (solids_mass_total_disposal_incineration / 100) / truck_weight
transport_solids_incineration_annual_cost = ((((landfill_distance / average_speed) * driver_wages  * number_of_trucks_solids_incineration) 
                                + (landfill_distance * diesel_transport * (solids_mass_total_disposal_incineration / 100) * diesel_cost)) * 2 * 365)


 # # sum and normalize cost to m3 treated 
capital_cost = ((incin_capital_cost_annualized) / (total_mass_input_waste * 365)) # USD/tonne
OM_cost = ((incin_maintenance + incin_material + transport_solids_incineration_annual_cost + fuel_cost_annual) / (total_mass_input_waste * 365)) # USD/tonne
energy_cost = incin_energy_cost / (total_mass_input_waste * 365) # USD/tonne
total_cost = capital_cost + OM_cost + energy_cost #USD/tonne


steel_incin_construction = (incinerator_weight * steel_fraction) / (incinerator_weight / 1000) #kg/tonnes
concrete_incin_construction = (incinerator_weight * concrete_fraction) / (incinerator_weight / 1000) #kg/tonnes
incineration_construction_LCA = ((solids_mass_disposal_incineration  * steel_incin_construction * stainless_steel_unit_LCA) 
+ (solids_mass_disposal_incineration * concrete_incin_construction * concrete_unit_LCA))
incineration_construction_LCA_annual = incineration_construction_LCA / analysis_period


incineration_annual_LCA = (((solids_mass_disposal_incineration * sand_fluidized_bed_incinerator * sand_LCA)
                            + (solids_mass_disposal_incineration * lime_incin * lime_LCA)
                            + (solids_mass_disposal_incineration * ash_incin * incineration_LCA)
                            + (fuel_demand * 3.785 * 0.493 * fuel_LCA/365))) * 365


direct_emissions_from_waste = ((solids_mass_disposal_incineration * PM_to_air_incin * PM_to_air_LCA)
                            + (solids_mass_disposal_incineration * NOx_to_air_incin * NOx_to_air_LCA)
                            + (solids_mass_disposal_incineration * mass_fraction_PTFE * pollution_control_factor * CO2_fossil_to_air_incin * CO2_fossil_to_air_LCA)
                            + (solids_mass_disposal_incineration  * mass_fraction_PTFE * pollution_control_factor * C2F6_to_air_incin * C2F6_to_air_LCA)
                            + (solids_mass_disposal_incineration  * mass_fraction_PTFE * pollution_control_factor * CF4_to_air_incin * CF4_to_air_LCA)
                            + (solids_mass_disposal_incineration * mass_fraction_PTFE * pollution_control_factor * CO_to_air_incin * CO_to_air_LCA)) * 365


transport_solids_incineration_annual_LCA = (landfill_distance * transport_unit_LCA) * 2 * number_of_trucks_solids_incineration * 365



total_impacts_capital = incineration_construction_LCA_annual / (total_mass_input_waste * 365)
total_impacts_OM = (incineration_annual_LCA + transport_solids_incineration_annual_LCA) / (total_mass_input_waste * 365)
total_impacts_energy = ((solids_mass_disposal_incineration * electricity_use_incin * electricity_unit_LCA) * 365)/(total_mass_input_waste * 365)
total_impacts_direct_emissions_from_waste = direct_emissions_from_waste / (total_mass_input_waste * 365)

        
total_impacts= total_impacts_capital + total_impacts_OM + total_impacts_energy + total_impacts_direct_emissions_from_waste




cost_incineration.extend(np.ravel(total_cost))
impacts_incineration.extend(np.ravel(total_impacts))



gwp_impacts = total_impacts[:, 1]

#Save  to excel
df_impacts = pd.DataFrame(total_impacts, columns=[
    'AP',    # Acidification Potential
    'GWP',   # Global Warming Potential
    'EFW',   # Ecotoxicity 
    'EP',    # Eutrophication Potential 
    'HTC',   # Human Toxicity, carcinogenic
    'HTNC',  # Human Toxicity, non-carcinogenic
    'ODP',   # Ozone Depletion Potential
    'PMFP',  # Particulate Matter Formation Potential
    'MIR'    # maximum incremental reactivity
])

df_results = pd.DataFrame({
    'cost_incineration': np.ravel(total_cost),
    'impacts_GWP_incineration': np.ravel(gwp_impacts)
})


df_cost = pd.DataFrame({
    'capital_cost': np.ravel(capital_cost),
    'OM_cost': np.ravel(OM_cost),
    'energy_cost': np.ravel(energy_cost),
    'total_cost': np.ravel(total_cost),
})

df_total_impacts = pd.DataFrame({
    'total_impacts': np.ravel(total_impacts)
})

df_total_impacts_capital = pd.DataFrame(
    total_impacts_capital,
    columns=df_impacts.columns
)

df_total_impacts_OM = pd.DataFrame(
    total_impacts_OM,
    columns=df_impacts.columns
)

df_total_impacts_energy = pd.DataFrame(
    total_impacts_energy,
    columns=df_impacts.columns
)

df_total_impacts_direct_emissions_from_waste = pd.DataFrame(
    total_impacts_direct_emissions_from_waste,
    columns=df_impacts.columns
)


#np.percentile(incineration_construction_LCA_annual, 50, axis=0)

#%% sensitivity
dfinputs = pd.DataFrame(inputs_for_sensitivity)[input_name_order]



sensitivity_cost = dfinputs.corrwith(df_cost.total_cost, method='spearman')


sensitivity_total_impacts_AP = dfinputs.corrwith(df_impacts.iloc[:, 0], method='spearman')  # Accessing the first column
sensitivity_total_impacts_GWP = dfinputs.corrwith(df_impacts.iloc[:, 1], method='spearman') 
sensitivity_total_impacts_EFW = dfinputs.corrwith(df_impacts.iloc[:, 2], method='spearman')  
sensitivity_total_impacts_EP = dfinputs.corrwith(df_impacts.iloc[:, 3], method='spearman') 
sensitivity_total_impacts_HTC = dfinputs.corrwith(df_impacts.iloc[:, 4], method='spearman')  
sensitivity_total_impacts_HTNC = dfinputs.corrwith(df_impacts.iloc[:, 5], method='spearman') 
sensitivity_total_impacts_ODP = dfinputs.corrwith(df_impacts.iloc[:, 6], method='spearman')  
sensitivity_total_impacts_PMFP = dfinputs.corrwith(df_impacts.iloc[:, 7], method='spearman') 
sensitivity_total_impacts_MIR = dfinputs.corrwith(df_impacts.iloc[:, 8], method='spearman') 

sensitivity = pd.DataFrame({
     'cost' :sensitivity_cost, 'AP' : sensitivity_total_impacts_AP,
     'GWP' : sensitivity_total_impacts_GWP, 'EFW' : sensitivity_total_impacts_EFW,
     'EP' : sensitivity_total_impacts_EP, 'HTC' : sensitivity_total_impacts_HTC,
     'HTNC' : sensitivity_total_impacts_HTNC, 'ODP' : sensitivity_total_impacts_ODP,
     'PMFP' : sensitivity_total_impacts_PMFP,'MIR' : sensitivity_total_impacts_MIR,
})


with pd.ExcelWriter("incin_PTFE.xlsx") as writer:
    df_results.to_excel(writer, sheet_name="results", index=False)
    df_total_impacts_capital.to_excel(writer, sheet_name="capital_impacts", index=False)
    df_total_impacts_OM.to_excel(writer, sheet_name="OM_impacts", index=False)
    df_total_impacts_energy.to_excel(writer, sheet_name="energy_impacts", index=False)
    df_total_impacts_direct_emissions_from_waste.to_excel(writer, sheet_name="direct_emissions_impacts", index=False)
    df_cost.to_excel(writer, sheet_name="cost", index=False)
    df_impacts.to_excel(writer, sheet_name="impacts", index=False)
    sensitivity.to_excel(writer, sheet_name="sensitivity")
    dfinputs.to_excel(writer, sheet_name='inputs')