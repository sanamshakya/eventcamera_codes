project(sample_camera)

set(SOURCES
    main.cpp
    EventLogger.cpp
    ClaheCuda.cu 
    ResizeRaw.cu 
    EventGeneratorCUDA.cu
)

set(LIBRARIES
    samples_framework
    samples_framework_checks
    samples_framework_program_arguments
    lodepng-src
    ${Driveworks_LIBRARIES}   # already contains CUDART::CUDART
)

add_executable(${PROJECT_NAME} ${SOURCES})

# dvs_types.hpp, Config.h, EventGeneratorCUDA.cuh ... live next to main.cpp
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBRARIES})

# Orin (DRIVE AGX) = sm_87. Without this, CMake picks its own default arch
# and the kernels may fail to launch on the device.
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_ARCHITECTURES 87
    CUDA_SEPARABLE_COMPILATION OFF
)

sdk_add_sample(${PROJECT_NAME})
