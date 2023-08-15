from eppa_exp1jr_scenario_discovery_main import SD
import pandas as pd
import matplotlib.pyplot as plt

sd_obj_pes = SD("GLB_RAW", "2C_USA_RENEW_SHARE")
sd_obj_pes.output_df = pd.read_excel("Copy of Full Data for Paper3+USAcons2050.xlsx", "A1.5C_pes_USA_cons")
print(sd_obj_pes.output_df)