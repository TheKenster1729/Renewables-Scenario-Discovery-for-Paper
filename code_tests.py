from eppa_exp1jr_scenario_discovery_main import InputOutputSD
import os

os.chdir(r"G:\My Drive\School\College\Senior Year\Senior Spring\UROP\Renewables Scenario Discovery for Paper")
generator = InputOutputSD('GLB_RAW', 'REF_GLB_RENEW_SHARE')
generator.parallel_plot_most_important_inputs(2050)
# generator.classification_with_time_series_clusters(max_depth = 3)

# regional GDP and population - add for the relevant region