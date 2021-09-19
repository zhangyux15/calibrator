## Dependency
Visual Studio 2019

Opencv 4.X(opencv 4.4.0) with contrib(to use aruco)

JsonCpp

Ceres

Eigen

## Features
ArucoCalibrator and CheckboardCalibrator are both available, you can switch them by the template function in main.cpp

The directory tree should look like this:
```
|-- calibrator 
|-- src
|-- x64
|-- data
    |-- external
    |   |-- frameidx_SN.jpeg
    |   |-- ...
    |-- internal
    |   |-- SN
    |       |-- xxx.jpeg
    |-- calibration.json
|--calibrator.sln
```