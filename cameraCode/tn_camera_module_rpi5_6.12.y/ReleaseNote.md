# TecchNexion Camera Driver Installation

## Release Note

| Release Date | Remark |
| --- | --- |
| 2026.02.23 | Update driver and device tree. </br></br>**Feature:** </br> 1. Support new tev cameras. </br> 2. Updated driver module version to 2.0. </br> 3. Fixed BSL mode bugs. |
| 2025.12.16 | Update driver and device tree. </br></br>**Feature:** </br> 1. Support kernel version 6.12.47+rpt-rpi-2712. </br> 2. Updated the v4l2 sub-device callback function changes. </br> 3. Added a function to get the chip ID of a sensor and check if the chip ID is supported by the driver. </br> 4. Removed mcu bsl command. |
| 2025.09.11 | Update driver and device tree. </br></br>**Feature:** </br> 1. Support kernel version 6.6.51+rpt-rpi-2712. </br> 2. Added v4l2 control for denoise. </br> 3. Added v4l2 control for auto exposure time range. </br> 4. Added v4l2 control for trigger mode. </br> 5. Added virtual ID control. </br> 6. Modified trigger mode control to support more modes. </br> 7. Modified tevs device tree blob to suit the driver. </br> 8. Fixed some flow bugs. |
| 2024.09.24 | Update driver. </br></br>**Feature:** </br> 1. Added ar0145 image sensor. </br> 2. Added max fps control. </br> 3. Added AGC mode. </br> 4. Added link frequency control. </br> 5. Added pixel rate control. </br> 6. Modified sensor setting when stream on. </br> 7. Modified frame interval control. </br> 8. Modified v4l2 ctrls inti function. </br> 9. Modifed pm control. |
| 2024.04.08 | First Release. </br></br>**Feature:** </br> 1. Support TEVS Cameras. </br> 2. Auto-install Script for Raspberry Pi 5. |
|   |   |
