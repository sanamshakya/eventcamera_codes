#pragma once

#include <string>

namespace evsim
{

struct Config
{
    bool useCamera = true;

    int cameraID = 2;

    std::string videoFile;

    int width = 1280;

    int height = 720;

    double fps = 30.0;

    int threshold = 20;

    float positiveThreshold = 15.0;
    float negativeThreshold = -15.0;

    double accumulationTime = 0.03;

    bool saveEvents = false;

    std::string outputCSV = "events.csv";

    // --- DVS-Voltmeter model fields, required by EventGeneratorCUDA ---
    // Must be a key understood by dvs::sensor_k_preset() (see dvs_types.hpp).
    std::string sensorType = "DVS346";
    // true  -> GenerateEventsKernelDVSFast (no RNG, floor-divide crossings)
    // false -> GenerateEventsKernelDVSStochastic (Bernoulli + IG sampling)
    bool fastDeterministicMode = true;
    double contrastThresholdOn = 1.0;
    double contrastThresholdOff = 1.0;
};

}
