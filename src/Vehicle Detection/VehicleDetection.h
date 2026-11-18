#pragma once

#include <cstdint>

enum class DetectionStatus {
    UNKNOWN,
    CLEAR,
    VEHICLE_DETECTED,
    SENSOR_ERROR
};

enum class RearWarning {
    UNKNOWN,
    SAFE,
    CAUTION,
    WARNING
};

struct VehicleData {
    float distanceMeters = 0.0f;
    bool radarValid = false;
    bool cameraDetected = false;
    bool cameraValid = false;
};

class VehicleDetection {
public:
    VehicleDetection();

    void updateFront(const VehicleData& data);
    void updateRear(const VehicleData& data);

    DetectionStatus getFrontStatus() const;
    RearWarning getRearWarning() const;

    float getFrontDistance() const;
    float getRearDistance() const;

private:
    DetectionStatus frontStatus;
    RearWarning rearWarning;

    float frontDistance;
    float rearDistance;
};