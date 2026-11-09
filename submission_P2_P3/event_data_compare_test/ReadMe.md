## File Description
`comare_events_plot_multi_ev_rate.py` : python script for plotting and comparison of multiple event logs
`plot_cmds.sh` : Commands example for comparing and plotting for various test cases

## Plotting and profiling captured event csv files
- For multi event captures comparison run, just giving the first filename of event csv files (for example ) :
```
    python3 ./compare_events_plot_multi_ev_rate.py <event file1 first name> <event file2 first name> ...
    python3 ./compare_events_plot_multi_ev_rate.py 
    clahe_rgba_custom_sensor clahe_raw_custom_sensor_k1_7 clahe_rgba_custom_sensor_deterministic clahe_raw_custom_sensor_deterministic
```
- This will generate two plots, `compare_tau.png` and `compare_rate.png`
    - `compare_tau.png` - plots inter event time interval, time since last event at same pixel as probability density function
    - `compare_rate.png` - plots events count per 100 msec time binning, which helps to visualize event rates