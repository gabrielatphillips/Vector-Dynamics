#include <iostream>
#include "Vehicle Detection/VehicleDetection.h"

int main()
{
    VehicleDetection detector;

    // Simulate a vehicle 20 feet behind the truck.
    VehicleData rearData;
    rearData.distanceMeters = 6.096f;
    rearData.radarValid = true;

    detector.updateRear(rearData);

    std::cout << "Rear distance: "
              << detector.getRearDistance()
              << " meters\n";

    switch (detector.getRearWarning())
    {
        case RearWarning::SAFE:
            std::cout << "Rear status: SAFE\n";
            break;

        case RearWarning::CAUTION:
            std::cout << "Rear status: CAUTION\n";
            break;

        case RearWarning::WARNING:
            std::cout << "Rear status: WARNING\n";
            break;

        default:
            std::cout << "Rear status: UNKNOWN\n";
            break;
    }

    return 0;
}