cmake_minimum_required(VERSION 3.10)
project(evt3_tools LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to an optimized build when no build type is given
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(MSVC)
  set(EVT3_WARN /W4)
else()
  set(EVT3_WARN -Wall -Wextra)
endif()

# CSV -> EVT 3.0
add_executable(evt3_encoder evt3_encoder.cpp)
target_compile_options(evt3_encoder PRIVATE ${EVT3_WARN})

# EVT 3.0 -> decoded events / accumulated frames (interactive window if OpenCV is found)
add_executable(evt3_viewer evt3_viewer.cpp evt3_decoder.hpp)
target_include_directories(evt3_viewer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(evt3_viewer PRIVATE ${EVT3_WARN})

option(EVT3_USE_OPENCV "Build the interactive viewer window with OpenCV" ON)
if(EVT3_USE_OPENCV)
  find_package(OpenCV QUIET)
endif()
if(OpenCV_FOUND)
  message(STATUS "evt3_viewer: OpenCV ${OpenCV_VERSION} found, interactive window enabled")
  target_include_directories(evt3_viewer PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(evt3_viewer PRIVATE ${OpenCV_LIBS})
  target_compile_definitions(evt3_viewer PRIVATE HAVE_OPENCV)
else()
  message(STATUS "evt3_viewer: OpenCV not found, headless mode only (--dump / --csv)")
endif()

install(TARGETS evt3_encoder evt3_viewer RUNTIME DESTINATION bin)
