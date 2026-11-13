## File Desscriptions
`main.cpp` - Main application based on sample camera application from DriveWorks SDK
`EventGeneratorCuda.cu, EventGeneratorCuda.cuh` - Cuda kernel for event gneration
`ClaheCuda.cu, ClaheCuda.cuh` - Cuda kernel for CLAHE processing
`ResizeRaw.cu, ResizeRaw.cuh` - Cuda kernel for RAW resizing
`dvs_types.h` - Sensor K parameters
`EventLogger.cpp, Eventloger.h` - Events data logging helper files
`GenProfiler.h` - Helper file for profiling CUDA kernels
`Event.h, EventPacket.h, IEventGenerator.h` - Data structures used for Event Generation

## CMakeLists.txt updates
- adding source files for CUDA kernel and event logging *.cpp at SOURCE variable 
```
...
set(SOURCES
    main.cpp
    EventLogger.cpp
    ClaheCuda.cu 
    ResizeRaw.cu 
    EventGeneratorCUDA.cu
)
...
```

## Build steps
- Copy and replace the code at `/usr/local/driverworks/samples/src/sensors/camera/camera` contained in this repo
- After code update, go to `/usr/local/driverworks/samples`
- create `build` directory and run cmake command with parameter for building for Orin architecture :
```
cd build
cmake ../ -DCMAKE_TOOLCHAIN_FILE=/usr/local/driveworks/samples/cmake/Toolchain-V5L.cmake

```
- Run make command for building driveworks `sample_camera` application
```
make sample_camera
```
- This will create `sample_camera` application in build directory at `<build dir>/src/sensors/camera/camera/`
- Copy the built application to the Dev kit. Here is the example using `scp`
```
scp src/sensors/camera/camera/sample_camera nvidia@192.168.1.1:
```

## Test and run
- Command line parameters for driveworks `sample_camera` application:
```
--rig: rig.json file for camera hardware pipeline configuration
--enable-events : Set to 1 for running event generation 
--event-display : Set to 1 to display event frames
--event-input : Set to 'raw' or 'rgba' depending upon available input stream from camera
--event-deterministic : Set to '1' for fast deterministic CUDA kernel without noise modeling and set to '0' for stocastic CUDA kernel
--event-max : Controls memory for maximum event count per frame
--event-sensor : K parameter presets as sensor types stored in 'dvs_types.hpp`
--event-clahe : Set to 1 to enable CLAHE CUDA kernel based processing for event generator input. Toggle at runtime with 'H' key
--event-csv : if set with <prefix>, log events with <prefix>.events.csv and <prefix>.frames.csv
```
More other options can be found by running `./sample camera --help`

-Commands for running in `RAW` and `RGBA` mode along with capturing event packets in *.csv files
```
    #for raw clahe
    EVSIM_INPUT_STATS=1 ./sample_camera --rig rig_raw.json --enable-events=1 --event-display=1 --event-input=raw --event-deterministic=0 --event-max=20000000 --event-sensor=custom_sensor_raw --event-clahe=1 --event-height=720 --event-width=1280 --event-csv=clahe_raw_custom_sensor >test_drivework_visualize.log

    #for raw rgba
    EVSIM_INPUT_STATS=1 ./sample_camera --rig rig.json --enable-events=1 --event-display=1 --event-input=rgba --event-deterministic=0 --event-max=20000000 --event-sensor=custom_sensor_rgba --event-clahe=1 --event-height=720 --event-width=1280 --event-csv=clahe_rgba_custom_sensor >test_drivework_visualize.log

    #for default raw (without clahe)
    EVSIM_INPUT_STATS=1 ./sample_camera --rig rig_raw.json --enable-events=1 --event-display=1 --event-input=raw --event-deterministic=0 --event-max=20000000 --event-sensor=custom_sensor_raw --event-clahe=0 --event-height=720 --event-width=1280 --event-csv=default_raw_custom_sensor >test_drivework_visualize.log

    #for default rgba (without clahe)
    EVSIM_INPUT_STATS=1 ./sample_camera --rig rig.json --enable-events=1 --event-display=1 --event-input=rgba --event-deterministic=0 --event-max=20000000 --event-sensor=custom_sensor_rgba --event-clahe=0 --event-height=720 --event-width=1280 --event-csv=default_rgba_custom_sensor >test_drivework_visualize.log

```

- Above run will display RGBA and Event frame side by side along with input statistics and CUDA kernel profiling in the terminal and creates `*.events.csv and *.frames.csv` are created with suffix as set for `--event-csv` in command line argument



## DVS sensor model parameters 
In current sensor model, event output depends upon three parameters :
```
drift  = K1 * rate of change in intensity + K4 + K5 * (average intensity)
Here K4 : constant drift (dark current) term,
and K5 : illumination-dependent drift term

modeled actual change in intensity = previous_residual_value + drift * delta_time

```
So even if rate of change in intensity = 0, there will be drift in intesity due to    K4 + K5 * (average intensity) term. 
So setting K4 and K5 to zero will remove the event accumulation due to these parameter and decrease noise when there is no any intesity change or motion in camera frame.

These parameters are stored in `dvs_types.h` file. Corresponding to `sensorType` as `Raw2DVS346` in `Config.h` corresponding camera model parameters in `dvs_types.h`.
```
if (camera_type == "Raw2DVS346")
        return SensorK{2.388, 4.166e-7, 1.541e-6, 9.768e-8, 1.466e-11, 9.824e-6};
//So in current config setting k4 and k5 terms to zero for removing DVS sensor's dark current / illumination-dependent leakage above line becomes :
		return SensorK{2.388, 4.166e-7, 1.541e-6, 0.0, 0.0, 9.824e-6};		
```



## Plotting and profiling captured event csv files
- Scripts for profiling are located at `<repo>/event_data_compare_test`
- For multi event captures comparison run, just giving the first filename of event csv files (for example ) :
```
    python3 ./compare_events_plot_multi_ev_rate.py <event file1 first name> <event file2 first name> ...
    python3 ./compare_events_plot_multi_ev_rate.py 
    clahe_rgba_custom_sensor clahe_raw_custom_sensor_k1_7 clahe_rgba_custom_sensor_deterministic clahe_raw_custom_sensor_deterministic
```
- This will generate two plots, `compare_tau.png` and `compare_rate.png`
    - `compare_tau.png` - plots inter event time interval, time since last event at same pixel as probability density function
    - `compare_rate.png` - plots events count per 100 msec time binning, which helps to visualize event rates


## Commands for various test conditions 

- Test 1 run commands for comparing  effect of CLAHE pre processing in stochastic RAW and RGBA pipeline

- with events csv capture for raw capture pipeline
```
timeout 30s ./sample_camera --rig rig_raw.json --enable-events=1 --event-display=1 --event-input=raw --event-deterministic=0 --event-max=20000000 --event-sensor=custom_sensor_raw --event-clahe=1 --event-height=720 --event-width=1280 --event-csv=clahe_raw_custom_sensor
```

- with events csv capture logging for rgba clahe
```
timeout 30s ./sample_camera --rig rig.json --enable-events=1 --event-display=1 --event-input=rgba --event-deterministic=0 --event-max=20000000 --event-sensor=custom_sensor_rgba --event-clahe=1 --event-height=720 --event-width=1280 --event-csv=clahe_rgba_custom_sensor
```

- with events csv capture for raw capture pipeline
```
timeout 30s ./sample_camera --rig rig_raw.json --enable-events=1 --event-display=1 --event-input=raw --event-deterministic=0 --event-max=20000000 --event-sensor=custom_sensor_raw --event-clahe=0 --event-height=720 --event-width=1280 --event-csv=default_raw_custom_sensor
```

- with events csv capture logging for rgba clahe

```
timeout 30s ./sample_camera --rig rig.json --enable-events=1 --event-display=1 --event-input=rgba --event-deterministic=0 --event-max=20000000 --event-sensor=custom_sensor_rgba --event-clahe=0 --event-height=720 --event-width=1280 --event-csv=default_rgba_custom_sensor
```

- Test 2 run commands (For comparing deterministic vs stochastic with CLAHE enabled)

- First two runs already run in Test 1
```
#./sample_camera --rig rig_raw.json --enable-events=1 --event-display=1 --event-input=raw --event-deterministic=0 --event-max=20000000 --event-sensor=custom_sensor_raw --event-clahe=1 --event-height=720 --event-width=1280 --event-csv=clahe_raw_custom_sensor
```
- with events csv capture logging for rgba clahe
```
#./sample_camera --rig rig.json --enable-events=1 --event-display=1 --event-input=rgba --event-deterministic=0 --event-max=20000000 --event-sensor=custom_sensor_rgba --event-clahe=1 --event-height=720 --event-width=1280 --event-csv=clahe_rgba_custom_sensor
```
```
timeout 30s ./sample_camera --rig rig_raw.json --enable-events=1 --event-display=1 --event-input=raw --event-deterministic=1 --event-max=20000000 --event-sensor=custom_sensor_raw --event-clahe=1 --event-height=720 --event-width=1280 --event-csv=clahe_raw_custom_sensor_deterministic

timeout 30s ./sample_camera --rig rig.json --enable-events=1 --event-display=1 --event-input=rgba --event-deterministic=1 --event-max=20000000 --event-sensor=custom_sensor_rgba --event-clahe=1 --event-height=720 --event-width=1280 --event-csv=clahe_rgba_custom_sensor_deterministic
```

- Test 3 run commands (For deterministic vs stochastic with CLAHE disabled)
- First two runs already run in Test 1
```
#./sample_camera --rig rig_raw.json --enable-events=1 --event-display=1 --event-input=raw --event-deterministic=0 --event-max=20000000 --event-sensor=custom_sensor_raw --event-clahe=0 --event-height=720 --event-width=1280 --event-csv=default_raw_custom_sensor

#with events csv capture logging for rgba clahe
#./sample_camera --rig rig.json --enable-events=1 --event-display=1 --event-input=rgba --event-deterministic=0 --event-max=20000000 --event-sensor=custom_sensor_rgba --event-clahe=0 --event-height=720 --event-width=1280 --event-csv=default_rgba_custom_sensor

timeout 30s ./sample_camera --rig rig_raw.json --enable-events=1 --event-display=1 --event-input=raw --event-deterministic=1 --event-max=20000000 --event-sensor=custom_sensor_raw --event-clahe=0 --event-height=720 --event-width=1280 --event-csv=default_raw_custom_sensor_deterministic

#with events csv capture logging for rgba clahe
timeout 30s ./sample_camera --rig rig.json --enable-events=1 --event-display=1 --event-input=rgba --event-deterministic=0 --event-max=20000000 --event-sensor=custom_sensor_rgba --event-clahe=0 --event-height=720 --event-width=1280 --event-csv=default_rgba_custom_sensor_deterministic
```
- All the above commands are present in `run_capture_event_stream.sh`

## For capturing RAW, RGBA and Events input capture for Dynamic range profiling
- Script for capturing RAW and Events : `capture_raw_input_event_stream.sh`
- Script for capturing RGBA and Events : `capture_rgba_input_event_stream.sh`


