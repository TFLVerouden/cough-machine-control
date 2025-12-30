*To do in this document:*
- [ ] Check and add laser safety label/regulations here.
- [ ] Add link to (micro)rheology repo(s) once applicable.
- [ ] Add link to SprayTec lift (and other) class(es) once properly sorted.
- [ ] Finish adding all electronics pdfs.
- [ ] Add pump cable pinouts below.


# Component list
## Airflow
Connected by means of, among other components, Swagelok male connector fittings and tube adapters (see [catalogue & installation instructions](/docs/airflow/Swagelok_fittings_ms-01-140_catalogue.pdf)). Advantage of connecting via these tubes is that the parts can be turned in any desired direction, rather than the angle being limited to keeping seals tight.

[![Cough machine airflow diagram](/docs/schematics/cough_machine_airflow.jpg)](https://github.com/TFLVerouden/cough-machine-control/blob/main/docs/cough_machine_airflow.pdf?raw=true)

### Pressure regulator
*FESTO VPPE-3-1-1/8-10-420-E1*
- [Website manufacturer](https://www.festo.com/nl/nl/a/557776)
- [Datasheet](/docs/airflow/FESTO_pressureregulator_VPPE-3-1-18-10-420-E1_datasheet.pdf)
- [Installation manual](/docs/airflow/FESTO_pressureregulator_VPPE-3-1-18-10-420-E1_manual.pdf)
- [Catalogue/info](/docs/airflow/FESTO_pressureregulator_info.pdf)

The pressure regulator has three parameter presets that range from slow (Pr1) to fast with overshoot (Pr3). Preset 2 seems to suit our needs. 

Connected using a *Pepperl+Fuchs V15-W-2M-PVC* female M12 to 5-way unterminated cable.
- [Website manufacturer](https://nl.rs-online.com/web/p/sensor-actuator-cables/8919448)
- [Datasheet](/docs/electronics/PepperlFuchs_cable_pressureregulator_V15-W-2M-PVC_datasheet.pdf)

Device-side front view of connector:

![Pressure regulator connector](/docs/electronics/screenshots/FESTO_pressureregulator_cable_pins.png)

On breadboard left-to-right:

| Pin | Cable colour | Assignment                    |
| --- | ------------ | ----------------------------- |
| 1   | Brown (BN)   | 24 V power supply             |
| 2   | White (WH)   | −, 0 mA setpoint current      |
| 3   | Blue (BU)    | Ground (0 V)                  |
| 4   | Black (BK)   | +, 4-20 mA setpoint current   |
| 5   | Gray (GY)    | 4-20 mA analog out            |
| 6   |              | N.C. for 180° turn protection |


### Air reservoir
*FESTO CRVZS-10*
- [Website manufacturer](https://www.festo.com/nl/nl/a/160237)
- [Catalogue/info/specifications](/docs/airflow/FESTO_reservoir_CRVS10.pdf)

### Valve #1 (solenoid valve)
*FESTO VUVS-LT25-M32C-MZD-G14-F8-1C1*
- [Website manufacturer](https://www.festo.com/nl/nl/a/8035170/)
- [Datasheet](/docs/airflow/FESTO_valve1_VUVS-LT25-M32C-MZD-G14-F8-1C1%20datasheet.pdf)

### Valve #2 (proportional valve)
*FESTO MPYE-5-1/4-420-B*
- [Website manufacturer](https://www.festo.com/nl/nl/a/161980/)
- [Datasheet](/docs/airflow/FESTO_valve2_MPYE-5-14-420-B_datasheet.pdf)
- [Installation manual](/docs/airflow/FESTO_valve2_manual.pdf)
- [Catalogue/info](/docs/airflow/FESTO_valve2_info.pdf)

Connected using *Pepperl+Fuchs V1-G-2M-PUR* 4-way M12 to 4-way unterminated cable.
- [Website manufacturer](https://nl.rs-online.com/web/p/sensor-actuator-cables/8171585)
- [Datasheet](/docs/electronics/PepperlFuchs_cable_valve2_V1-G-2M-PUR_datasheet.pdf)

Device-side front view of connector:

![Valve 2 connector](/docs/electronics/screenshots/FESTO_valve2_cable_pins.png)

On breadboard left-to-right:
| Pin | Cable colour | Assignment                    |
| --- | ------------ | ----------------------------- |
| 1   | Brown (BN)   | 24 V power supply             |
| 2   | White (WH)   | Ground/- (0 V)                |
| 3   | Blue (BU)    | +, 12-20 mA setpoint current  |
| 4   | Black (BK)   | −, 0 mA setpoint current      |
| 5   |              | N.C. for 180° turn protection |

### Obsolete
- ~~Bronkhorst EL-FLOW Select F-113AC flow meter~~

## Electronics
[![Cough machine wiring diagram](/docs/schematics/cough_machine_wiring.jpg)](https://github.com/TFLVerouden/cough-machine-control/blob/main/docs/cough_machine_wiring.pdf?raw=true)

Note: when making any physical changes to the current loop of R-click or T-click, reset the microcontroller because the current will be incorrect (~ 2 mA) otherwise.

Breadboard top view:
![Cough machine control breadboard top view](/docs/breadboard_top_view.jpeg)

Breadboard bottom view:
![Cough machine control breadboard bottom view](/docs/breadboard_bottom_view.jpeg)

### Power supplies
*TracoPower TBLC-06-105* 5 V power supply
- [Data sheet](/docs/electronics/TracoPower_tblc06_datasheet.pdf)
- [Manual](/docs/electronics/TracoPower_tblc06_installation.pdf)

*TracoPower TBLC-15-124* 24 V power supply
- [Data sheet](/docs/electronics/TracoPower_tblc15_datasheet.pdf)
- [Manual](/docs/electronics/TracoPower_tblc15_installation.pdf)



- Adafruit ItsyBitsy M4 Express: https://learn.adafruit.com/introducing-adafruit-itsybitsy-m4/pinouts
    - ATSAMD51G19A chip: https://www.microchip.com/en-us/product/ATSAMD51G19A
- MIKROE 4-20 mA T-click: https://www.mikroe.com/4-20ma-t-click
    - Calibration values for both pressure regulator and proportional valve setpoint: bitval 796 <=> 3.97 mA; bitval 3982 <=> 19.90 mA.
- MIKROE 4-20 mA R-click: https://www.mikroe.com/4-20ma-r-click
   - Modified to act as passive current receiver (not supplying any voltage), see notes in the [MIKROE_4_20mA_RT_Click library](https://github.com/Dennis-van-Gils/MIKROE_4_20mA_RT_Click/tree/main/active_to_passive_R_click_modification)
   - Calibration value for pressure regulator readout:
   bitval 806 <=> 4.04 mA, bitval 2191 <=> 10.98 mA.
- Adafruit MOSFET driver: https://learn.adafruit.com/adafruit-mosfet-driver
- Adafruit SHT45 temperature \& humidity sensor: https://learn.adafruit.com/adafruit-sht40-temperature-humidity-sensor
    - SHT4X chip: https://sensirion.com/products/catalog/SHT45
- Traco Power TBLC power supplies: https://www.tracopower.com/int/series/tblc
- Isocom 6N138 Optocoupler: https://nl.rs-online.com/web/p/optocouplers/1610989?srsltid=AfmBOopHfB2N8esy3AUzJydobraa8phtpUnF_R_vVBQSlnlFK72Xo9mU

- StarTech ICUSB2324I USB-RS232 hub: https://www.startech.com/nl-nl/kaarten-peripherals/icusb2324i?srsltid=AfmBOopFQCteDwa-6x80QMKLGdYEHS63esR50nsnaFysBw_U2jR4mku1

## PIV add-on
### Manual pressure regulator
*SMC IR2000-F02*
- [Website manufacturer](https://www.smcpneumatics.com/IR2000-F02.html)
- [Manual](/docs/piv_addon/SMC_pressureregulator_IR2000-F02_manual.pdf)

Pressure at the output is read out by *FESTO SPAB-P10R-F-PB-L1*
- [Website manufacturer](https://www.festo.com/nl/nl/a/8035539/)
- [Datasheet](/docs/piv_addon/FESTO_pressuresensor_SPAB-P10R-F-PB-L1_datasheet.pdf)
- [Manual](/docs/piv_addon/FESTO_pressuresensor_SPAB-P10R-F-PB-L1_manual.pdf)
- [Catalogue/info](/docs/piv_addon/FESTO_pressuresensor_info.pdf)

This is connected using the *FESTO NEBS-L1G4-K-2.5-LE4* cable.
- [Website manufacturer](https://www.festo.com/nl/nl/a/572576/)

Device-side front view of connector:

![Pressure sensor connector](/docs/electronics/screenshots/FESTO_pressuresensor_cable_pins.png)

On breadboard left-to-right:

| Breadboard pin | Device pin | Cable colour | Assignment         |
| -------------- | ---------- | ------------ | ------------------ |
| N.C. or 1      | 2          | Black (BK)   | N.C. (digital out) |
| 2              | 4          | Blue (BU)    | Ground (0 V)       |
| 3              | 3          | White (WH)   | 4-20 mA analog out |
| 4              | 1          | Brown (BN)   | 24 V power supply  |

### Laser
*Thorlabs CPS635R*
- [Website manufacturer](https://www.thorlabs.com/thorproduct.cfm?partnumber=CPS635R)
- [Datasheet](/docs/piv_addon/Thorlabs_laser_CPS635R-specsheet.pdf)


Mounted using a [Thorlabs MK11F/M laser diode mount](https://www.thorlabs.com/thorproduct.cfm?partnumber=MK11F/M)

### Photodiode
*Thorlabs PDA36A2*
- [Website manufacturer](https://www.thorlabs.com/thorproduct.cfm?partnumber=PDA36A2)
- [Manual](/docs/piv_addon/Thorlabs_photodiode_PDA36A2-manual.pdf)


## Peripherals
### Syringe pump
*Harvard Apparatus PHD 2000*
- [Manual](/docs/peripherals/Harvard_PHD2000_syringe_pump_manual.pdf)
- [Recommended Python serial control code](https://github.com/WetenSchaap/pumpy3)

The pump is connected using a custom RJ12 to DB9 cable, which consists of an [Assmann AT-S-26-6-6-S-7](/docs/peripherals/pump_cable/Assmann_AT-S-26-6-6-S-7_RJ12_straight_through_cable.pdf) RJ12 straight-through cable and an [L-COM RBA096F](/docs/peripherals/pump_cable/L-COM_RBA096F_adaptor_DB9F-RJ12.pdf) RJ12 to DB9 female adapter.

Make sure to set the correct syringe diameter. Commonly used Hamilton-Microliter Series Gastight syringes:
| Volume (mL) | Diameter (mm) |
| ----------- | ------------- |
| 1.0         | 4.61          |
| 2.5         | 7.28          |
| 5.0         | 10.3          |

### Laser diffraction particle sizer
*Malvern SprayTec*
- [Website manufacturer](https://www.malvernpanalytical.com/en/products/product-range/spraytec)
- [Manual](/docs/peripherals/Malvern_SprayTec_manual.pdf)
- [Presentation before installation](/docs/peripherals/Malvern_SprayTec_installation_slides.pdf)
- [Brochure](/docs/peripherals/Malvern_SprayTec_brochure.pdf)
- [Troubleshooting: set trigger for dilute sprays](/docs/peripherals/spraytec/SprayTec_support_set_trigger_dilute_spray.pdf)
- [Troubleshooting: ambient light](/docs/peripherals/spraytec/SprayTec_support_ambient_light.pdf)
- [Troubleshooting: alarms at low signal](/docs/peripherals/spraytec/SprayTec_support_alarm_at_low_signal.pdf)
    
The SprayTec sits on top of a lift that can be controlled over serial.

### Optics
The distortion targets we use in our lab for calibrating high-speed video recordings are the [Thorlabs R2L2S3P3 (500 μm)](https://www.thorlabs.com/thorproduct.cfm?partnumber=R2L2S3P3) and [R2L2S3P4 (1 mm)](https://www.thorlabs.com/thorproduct.cfm?partnumber=R2L2S3P4) glass targets. They are positioned using the [B3C/M](https://www.thorlabs.com/thorproduct.cfm?partnumber=B3C/M) platform and [FFM1](https://www.thorlabs.com/thorproduct.cfm?partnumber=FFM1) holder. 

### Rheometry
Characterisation of the liquids is done using an Anton Paar rheometer.
- [Software quickstart](/docs/peripherals/AntonPaar_RheoCompass_quickstart.pdf)
- [Software manual](/docs/peripherals/AntonPaar_RheoCompass_manual.pdf)