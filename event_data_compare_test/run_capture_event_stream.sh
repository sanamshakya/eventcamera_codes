



#for raw clahe
EVSIM_INPUT_STATS=1 ./sample_camera --rig rig_raw.json --enable-events=1 --event-display=1 --event-input=raw --event-deterministic=0 --event-max=20000000 --event-sensor=custom_sensor_raw --event-clahe=1 --event-height=720 --event-width=1280 --event-csv=clahe_raw_custom_sensor >test_drivework_visualize.log

#for raw rgba
EVSIM_INPUT_STATS=1 ./sample_camera --rig rig.json --enable-events=1 --event-display=1 --event-input=rgba --event-deterministic=0 --event-max=20000000 --event-sensor=custom_sensor_rgba --event-clahe=1 --event-height=720 --event-width=1280 --event-csv=clahe_rgba_custom_sensor >test_drivework_visualize.log

#for default raw (without clahe)
EVSIM_INPUT_STATS=1 ./sample_camera --rig rig_raw.json --enable-events=1 --event-display=1 --event-input=raw --event-deterministic=0 --event-max=20000000 --event-sensor=custom_sensor_raw --event-clahe=0 --event-height=720 --event-width=1280 --event-csv=default_raw_custom_sensor >test_drivework_visualize.log

#for default rgba (without clahe)
EVSIM_INPUT_STATS=1 ./sample_camera --rig rig.json --enable-events=1 --event-display=1 --event-input=rgba --event-deterministic=0 --event-max=20000000 --event-sensor=custom_sensor_rgba --event-clahe=0 --event-height=720 --event-width=1280 --event-csv=default_rgba_custom_sensor >test_drivework_visualize.log


#For comparing above 4 captures raw clahe, raw rgba, default raw, 
python3 ./compare_events_plot_multi_ev_rate.py clahe_rgba_custom_sensor clahe_raw_custom_sensor default_rgba_custom_sensor default_raw_custom_sensor
