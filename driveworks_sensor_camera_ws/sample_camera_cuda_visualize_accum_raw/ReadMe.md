## Driveworks sample camera application with CUDA event generator

- Copy and replace the camera sample application located at `<driveworks src>/samples/src/sensors/camera/camera` with the current source content in this repo
- Build the appication by running cmake at the sample main directory :
```
cd <driverworks src>/samples
mkdir build
cd build
cmake ..
make VERBOSE=1 sample_camera
```
Check that EventGeneratorCUDA.cu is compiled by nvcc with -arch=sm_87 (or --generate-code=arch=compute_87,code=sm_87) in the build log.