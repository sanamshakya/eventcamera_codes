## Steps to Build 

- Add the codes into nvsipl_camera default code
- Update Makefile to build eventgenerator
	- Makefile updates 
	```
	#change include to 
	include /drive/drive-linux/make/nvdefs.mk

	TARGETS = nvsipl_camera_event_gen_save_bin

	CXXFLAGS := $(NV_PLATFORM_OPT) $(NV_PLATFORM_CFLAGS) -std=c++14 -fexceptions -frtti -fPIC
	CPPFLAGS := -I/drive/drive-linux/include/nvmedia_6x \
            -I/drive/drive-linux/plugin \
            $(NV_PLATFORM_CPPFLAGS) \
            $(NV_PLATFORM_SDK_INC) \
            -DWIN_INTERFACE_CUSTOM \
            -D_POSIX_C_SOURCE=200112L

	LDFLAGS  := $(NV_PLATFORM_SDK_LIB) $(NV_PLATFORM_TARGET_LIB) $(NV_PLATFORM_LDFLAGS)

	OBJS   := CUtils.o
	OBJS   += main.o
	OBJS   += EventGenerator.o
	...
	```
- Enable event generation in main.cpp 

```
		...
		if (cmdline.bShowMetadata) {
                    upCons->EnableMetadataLogging();
                }
                
 		evsim::Config evConfig;
                
                upCons->EnableEventGeneration(evConfig,"events.bin");
                upCons->SetTscFrequency(31250000.0);
                
                upCons->EnableRawCapture("data.raw");
                ...
```
