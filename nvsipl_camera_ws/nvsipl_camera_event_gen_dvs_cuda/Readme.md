## Event Generation from RAW Bayer capture using nvsipl_camera  

### File Description :

`EventGeneratorCUDA.cu, EventGeneratorCUDA.cuh` : Class description and declarations for `EventGeneratorCuda` class. It implements the event generation algorithm in CUDA kernels. It also handles  initialization, input/output memory buffers and runners for runnin CUDA kernels.

`EventGenerator.h, EventGenerator.cpp `: CPU based threaded implementation for `EventGenerator` class.

`IEventGenerator.h `: Adapter code for interfacing EventGenerator code

`Event.h, PixelState.h, EventPacket.h `: Data structure used by `EventGenerator` and `EventGeneratorCuda` class.


### Main code highlights : 

### CUDA kernel class:

- Two cuda kernel definitions :

```
// Cuda kernel for input initialization for cuda kernels
__global__ void InitFirstFrameKernel(PixelStateGPU *states,
                                      const uint16_t *image,
                                      int width, int height, int stride,
                                      double timestamp)

``` 

```
// Main Cuda kernel for event generation
void GenerateEventsKernel(PixelStateGPU *states,
                                      const uint16_t *image,
                                      int width, int height, int stride,
                                      double timestamp,
                                      float positiveThreshold,
                                      float negativeThreshold,
                                      int maxEvents,
                                      uint8_t *outputBuffer /* [int count][Event...] */)

```

- Cuda kernel runners 
```
EventPacket EventGeneratorCUDA::generate(
    const uint16_t *image,
    int width,
    int height,
    int stridePixels,
    double timestamp)
```
- Function for allocating ids for running cuda kernel and running the cuda kernel
	- Sets number of 256 parallel threads, running in 32 blocks each running 8 threads
	```
	const int cuda_blocks = 32;
    	const int cuda_threads = 8;
    	const dim3 block(cuda_blocks, cuda_threads);
    	const dim3 grid((width_ + block.x - 1) / block.x, (height_ + block.y - 1) / block.y);
    	...
    	 GenerateEventsKernel<<<grid, block, 0, stream_>>>(
        d_pixelStates_, d_image_, width_, height_, width_,
        timestamp, config_.positiveThreshold, config_.negativeThreshold,
        maxEvents_, d_outputBuffer_);
        ...
	```
	

### NVSIPL consumer integration
- In NVSIPL framework, processing of camera frames is handles by consumer code. In case of nvsipl_camera application `CNvSIPLConsumer.hpp` consists of the consumer code.
- Event generation code is integrated in `CNvSIPLConsumer.hpp` code.
### Major consumer code customisation highlights :

- Funcition for enabling event generation 
```
void EnableEventGeneration(const evsim::Config &evConfig,
                               const string &sEventFilename = "",
                               bool useCuda = true,
                               int maxEventsPerFrame = 200000)
```
	- Function creating the pointer for CPU or CUDA based event generation object on basis of `useCuda` flag :
	```
	#ifdef EVSIM_ENABLE_CUDA
        	if (useCuda)
        	{
            		m_pEventGenerator.reset(new evsim::EventGeneratorCUDA(evConfig, maxEventsPerFrame));
            		LOG_ERR("EventGenerator: useCuda enabled, running cuda based event generator\n");
        	}
        	else
	#else
        	if (useCuda)
        	{
            		LOG_ERR("EventGenerator: useCuda requested but this binary was built "
                    "without EVSIM_ENABLE_CUDA - falling back to the CPU backend\n");
        	}
        	(void)maxEventsPerFrame;
	#endif
        	{
            		m_pEventGenerator.reset(new evsim::EventGenerator(evConfig));
        	}
	```
	- This function is called in main.cpp to enable the event generation
	```
	//update in main.cpp for enabling event generation
	...
	evsim::Config evConfig;
                
        upCons->EnableEventGeneration(evConfig,"events.bin");
        upCons->SetTscFrequency(31250000.0);
	...
	
	```
- Main callback called on each new camera raw frame 
```
SIPLStatus OnFrameAvailable(INvSIPLClient::INvSIPLBuffer *pBuffer,
                                NvSciSyncCpuWaitContext cpuWaitContext)
``` 
	- In this function RAW Bayer frame is captured and passed to the event generation thread using following code : 
	```
	EnqueueRawFrame(std::move(raw16), eventFrameWidth,
                                 eventFrameHeight, eventFrameStride, timestamp);
	```
	
- Main event generation thread which calls event genertion function :
```
EventProcessingThreadFunc()
```
	- Calls the event generation :
	```
	...
	 evsim::EventPacket packet = m_pEventGenerator->generate(
                frame.data.data(), frame.width, frame.height,
                frame.stridePixels, frame.timestamp);
	...
	```

## Code Update for stochastic event generator 
- In `EventGeneratorCUDA.cu`, two event generation logic is implemented : 
	1) Deterministic : Similar to previous but considering camera parameters
	```
	void GenerateEventsKernelDVSFast(
    PixelStateGPU *states,
    const uint16_t *image,
    int width, int height, int stride,
    float k1_over_dt_us,   // k1 / dt_frame_in_microseconds -- see the .cu note on units
    float k2, float k4, float k5,
    float thresholdOn, float thresholdOff,
    float dtFrameUs,         // for the drift accumulation
    double dtFrameSeconds,   // for producing real-world event timestamps
    double frameStartTime,   // seconds -- previous frame's timestamp
    int maxEvents,
    uint8_t *outputBuffer /* [int count][Event...] */)
	
	```
	2) Stochastic : Added Brownian Motion based analytic pdf for polarity estimation and Inverse Gaussian pdf for Timestamp estimation of events
	```
	void GenerateEventsKernelDVSStochastic(
    PixelStateGPU *states, curandState *rngStates,
    const uint16_t *image, int width, int height, int stride,
    float k1, float k2, float k3, float k4, float k5, float k6,
    float thresholdOn, float thresholdOff,
    float dtFrameUs, double dtFrameSeconds, double frameStartTime,
    int maxEvents, uint8_t *outputBuffer)
	```
- Next added  sensorType, fastDeterministicMode, contrastThresholdOn/Off parameters used by cuda kernels
	- sensorType : Model parameters for event cameras
	- fastDeterministicMode : for enabling and disabling stochastic mode
	- contrastThresholdOn/Off : threshold values for polarity and time stamp estimation

## DVS sensor model parameters 
Deterministic mode is enabled, after setting `bool fastDeterministicMode = true;` in `Config.h`. 
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
## Running Stochastic or Deterministic Cuda Kernel
To run the deterministic kernel, set the following in `Config.h`
```
bool fastDeterministicMode = true; //for enabling and disabling stochastic mode
```
Next for running stochastic event generation Cuda kernel, set 
```
bool fastDeterministicMode = false; //for enabling and disabling stochastic mode
```
After setting the `fastDeterministicMode`, rebuild the `EventGenratorCUDA.cu` cuda kernel code.

	
## Build steps
- Copy  all files to  nvsipl_camera source application
- Add the display rendering class header files 
- Use the updated make file with OpenCV support
	- If cuda kernel compilation fails use `nvcc` compiler for compiling the cuda kernel
	```
	nvcc -O2 -std=c++17 -arch=sm_87 EventGeneratorCUDA.cu EventGeneratorCUDA.o
	```
- Update the `main.cpp` for enabling the event generation by adding following lines at `upCons` initialization : 
```
if (cmdline.bShowMetadata) {
                    upCons->EnableMetadataLogging();
                }
                
                evsim::Config evConfig;
                
                upCons->EnableEventGeneration(evConfig,"events.bin");
                upCons->SetTscFrequency(31250000.0);
                
                upCons->EnableRawCapture("data.raw");
#if !NV_IS_SAFETY
                if (cmdline.bAutoLEDControl) {
                    upCons->EnableLEDControl();
                }
#endif // !NV_IS_SAFETY

```
- Next build the code by running the make command : 
```
make
```

## Running the application :
```
export LD_LIBRARY_PATH=/home/nvidia/ws_grv/lib/:$LD_LIBRARY_PATH

//first disable gui
sudo systemctl stop gdm

./nvsipl_camera -c V1SIM728S3RU4120NC00_CPHY_x4
    -m "0x0001 0x0000 0x0000 0x0000 0x0000"  --enableRawOutput --disableISP0Output --disableISP1Output --disableISP2Output -d 0
```

	 



























