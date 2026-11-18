#include "VehicleDetection.h"
#include <cmath>

//------------------------------------
// Detection Thresholds
//------------------------------------

namespace 
{
    // FRONT DETECTION REQUIREMENTS:
    // 400 +/- 40 FEET, APPROXIMATELY 122 M.
    constexpr float FRONT_RANGE_METERS = 121.92f;


    // REAR DETECTION REQUIREMENTS:
    // 15 TO 200 FEET.
    constexpr float REAR_MIN_METERS = 4.572f;
    constexpr float REAR_MAX_METERS = 60.96f;

    // INITIAL REAR WARNING THRESHOLDS.
    // THESE ARE PROVISIONAL AND REQUIRE TESTING.
    constexpr float WARNING_DISTANCE = 9.144f; // 30 FT
    constexpr float CAUTION_DISTANCE = 22.86f; // 75 FT

    bool validDistance(float distance)
    {
        return std::isfinite(distance) && distance > 0.0f;
    }

}

//------------------------------------
// Constructor
//------------------------------------

VehicleDetection::VehicleDetection()
    : frontStatus(DetectionStatus::UNKNOWN),
      rearWarning(RearWarning::UNKNOWN),
      frontDistance(0.0f),
      rearDistance(0.0f) 
      {

      }

//------------------------------------
// Front Vehicle Detection
//------------------------------------

void VehicleDetection::updateFront(const VehicleData& data)
{
    if(!data.radarValid || !validDistance(data.distanceMeters))
    {
        frontStatus = DetectionStatus::UNKNOWN;
        frontDistance = 0.0f;
        return;
    }

    frontDistance = data.distanceMeters;

    // A camera-confirmed vehicle within the
    // front detection range means the lane
    // is occupied
    if (frontDistance <= FRONT_RANGE_METERS && data.cameraValid && data.cameraDetected)
    {
        frontStatus = DetectionStatus::VEHICLE_DETECTED;
    }

    else
    {
        // NO CONFIRMED VEHICLE IS NOT THE
        // SAME AS A VERIFIED CLEAR LANE
        frontStatus = DetectionStatus::UNKNOWN;
    }
}

//------------------------------------
// Rear Follwing-Distance Detection
//------------------------------------

void VehicleDetection::updateRear(const VehicleData& data)
{
    if (!data.radarValid || !validDistance(data.distanceMeters))
    {
        rearWarning = RearWarning::UNKNOWN;
        rearDistance = 0.0f;
        return;
    }

    rearDistance = data.distanceMeters;

    // OUTSIDE THE SPECIFIED SENSING RANGE.
    if(rearDistance < REAR_MIN_METERS || rearDistance > REAR_MAX_METERS)
    {
        rearWarning = RearWarning::UNKNOWN;
        return;
    }

    // RADAR MEASUREMENT DRIVES THE WARNING.
    // CAMERA CONFIRMATION CAN BE INTEGRATED
    // AFTER THE CAMERA DRIVER IS IMPLEMENTED.

    if (rearDistance <= WARNING_DISTANCE)
    {
        rearWarning = RearWarning::WARNING;
    }
    else if (rearDistance <= CAUTION_DISTANCE)
    {
        rearWarning = RearWarning::CAUTION;
    }
    else 
    {
        rearWarning = RearWarning::SAFE;
    }
}

//------------------------------------
// Status Getters
//------------------------------------

DetectionStatus VehicleDetection::getFrontStatus() const
{
    return frontStatus;
}

RearWarning VehicleDetection::getRearWarning() const
{
    return rearWarning;
}

//------------------------------------
// Distange Getters
//------------------------------------

float VehicleDetection::getFrontDistance() const
{
    return frontDistance;
}

float VehicleDetection::getRearDistance() const
{
    return rearDistance;
}