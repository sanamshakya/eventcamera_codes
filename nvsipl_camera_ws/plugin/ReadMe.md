## Sample Plugin for Auto exposure/gain/HDR and Auto white balance

- Plugin uses `ISiplControlAuto` class for controlling gain and exposure of camera sensor
- Update in code for printing gain and exposure by adding following lines at the end of procesAE function
```
printf("Sensor exp[0]: applied=%.6f applying=%.6f gain applied=%.2f applying=%.2f\n",
        parsedEmbData.sensorExpInfo.exposureTime[0], nextExpTime[0],
        parsedEmbData.sensorExpInfo.sensorGain[0], nextExpGain[0]);
```
- Next build the pluging by running make command, which will create `libnvsipl_sampleplugin.so` file
    - This shared library (*.so) file is called by main code at : 
    ```
    ...
      } else {
                    pluginlib_handle = dlopen("libnvsipl_sampleplugin.so", RTLD_LAZY);
                    if (!pluginlib_handle) {
                        LOG_ERR("Failed to open lib libnvsipl_sampleplugin.so\n");
                        return -1;
                    }
                    nvsipl::ISiplControlAuto* (*libCreatePlugin)() = (nvsipl::ISiplControlAuto*(*)())dlsym(pluginlib_handle, "CreatePlugin");
                    if (!libCreatePlugin) {
                        LOG_ERR("Failed to create function pointer for CreatePlugin\n");
                        return -1;
                    }
                    upCustomPlugins[uSensor] = libCreatePlugin();
                    CHK_PTR_AND_RETURN(upCustomPlugins[uSensor], "AutoControl plugin creation");

                    status = upMaster->RegisterAutoControl(uSensor, CUSTOM_PLUGIN0,
                                                           upCustomPlugins[uSensor], blob);
                    if (status != NVSIPL_STATUS_OK) {
                        LOG_ERR("SetAutoControl(CUST0) failed for ISP output of sensor:%u\n", uSensor);
                        return -1;
                    }
    ...
    ```

    And above code runs when one of the ISP is enabled.

## Build and Running the plugin
Build the plugin by running the `make` command.
This will create `libnvsipl_sampleplugin.so` shared plugin object file.
In `main.cpp` give full path of the above shared object file as below : 
```
...
else {
                    pluginlib_handle = dlopen("<full path>/libnvsipl_sampleplugin.so", RTLD_LAZY);
                    if (!pluginlib_handle) {
                        LOG_ERR("Failed to open lib libnvsipl_sampleplugin.so\n");
                        return -1;
                    }
                    nvsipl::ISiplControlAuto* (*libCreatePlugin)() = (nvsipl::ISiplControlAuto*(*)())dlsym(pluginlib_handle, "CreatePlugin");
                    if (!libCreatePlugin) {
                        LOG_ERR("Failed to create function pointer for CreatePlugin\n");
                        return -1;
                    }
                    upCustomPlugins[uSensor] = libCreatePlugin();
                    CHK_PTR_AND_RETURN(upCustomPlugins[uSensor], "AutoControl plugin creation");

                    status = upMaster->RegisterAutoControl(uSensor, CUSTOM_PLUGIN0,
                                                           upCustomPlugins[uSensor], blob);
                    if (status != NVSIPL_STATUS_OK) {
                        LOG_ERR("SetAutoControl(CUST0) failed for ISP output of sensor:%u\n", uSensor);
                        return -1;
                    }
                }
...                
```

Then build the main.cpp and run the application with one of the ISP enabled by removing `--disableISPXOutput` while running nvsipl_camera application.


