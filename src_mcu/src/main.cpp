/*
 * Cough Machine Control System
 *
 * Controls a solenoid valve for atomisation experiments with precise timing.
 * Monitors pressure and environmental conditions.
 */

#include "DvG_StreamCommand.h"
#include "MIKROE_4_20mA_RT_Click.h"
#include <Adafruit_DotStar.h>
#include <Adafruit_SHT4x.h>
#include <Arduino.h>

// TODO: Add/change function to adjust all delays via serial command
// TODO: Make it so the opening procedure using loaded protocol can be used with
// droplet detection
// TODO: Streamline function names (can they all be one character, consistent
// question mark, etc?)
// TODO: Enable/disable debug mode via serial command
// TODO: Allowed set pressure range is slightly above 0.00 bar, change mA range
// to 3.99?

// ============================================================================
// DEBUG CONFIGURATION
// ============================================================================
// Set to 1 to enable debug messages, 0 to disable for maximum speed
#define DEBUG 1

#if DEBUG
#define DEBUG_PRINT(x) Serial.print(x)
#define DEBUG_PRINTLN(x) Serial.println(x)
#else
#define DEBUG_PRINT(x)
#define DEBUG_PRINTLN(x)
#endif

// ============================================================================
// PIN DEFINITIONS
// ============================================================================
const int PIN_VALVE = 7;       // MOSFET gate pin for solenoid valve control
const int PIN_PROP_VALVE = 11; // Chip select for proportional valve
const int PIN_PRES_REG = 10;   // Chip select for pressure regulator
const int PIN_CS_RCLICK = 2;   // Chip select for R-Click pressure sensor (SPI)
const int PIN_TRIG = 9; // Trigger output for peripheral devices synchronization
const int PIN_LASER = 12; // Laser MOSFET gate pin for droplet detection
const int PIN_PDA = A2;   // Analog input from photodetector
// Note: PIN_DOTSTAR_DATA and PIN_DOTSTAR_CLK are already defined in variant.h

// ============================================================================
// INITIALIZE DVG_STREAMCOMMAND AND FLOW CURVE DATASETS
// ============================================================================
const int MAX_DATA_LENGTH = 2000; // Max serial dataset size
const uint16_t CMD_BUF_LEN =
    32000;                       // RAM size allocation for Serial buffer size
int incomingCount = 0;           // Declare incoming dataset length globally
char cmd_buf[CMD_BUF_LEN]{'\0'}; // Instantiate empty Serial buffer
uint32_t time_array[MAX_DATA_LENGTH]; // Time dataset
float value_array[MAX_DATA_LENGTH];   // mA dataset
// Create DvG_StreamCommand object on Serial stream
DvG_StreamCommand sc(Serial, cmd_buf, CMD_BUF_LEN);

// ============================================================================
// DATASET PROCESSING & EXECUTION VARIABLES
// ============================================================================
int sequenceIndex = 0;     // Index of dataset to execute on time
int dataIndex = 0;         // Number of datapoints of dataset stored
int datasetDuration = 0.0; // Duration of the uploaded flow profile

// ============================================================================
// TIMING PARAMETERS
// ============================================================================
const uint32_t TRIGGER_WIDTH = 10000; // Trigger pulse width [µs] (10ms)
uint32_t tick = 0;                    // Timestamp for timing events [µs]
uint32_t tick_delay = 59500;          // Delay before opening valve [µs]
uint32_t pda_delay = 10000; // Delay before photodiode starts detecting [µs]
uint32_t valve_delay_open =
    11000; // Delay between solenoid valve and proportional valve opening [µs]
           // (positive is sol first)
int32_t valve_delay_close =
    -40000; // Delay between proportional valve and solenoid valve closing [µs]
            // (negative is sol first)
uint32_t runCalltTime = 0; // Time elapsed since "RUN" command [µs]

// ============================================================================
// SENSOR CONFIGURATION
// ============================================================================
// Pressure sensor (4-20mA R-Click) with exponential moving average filtering
const uint32_t EMA_INTERVAL = 500; // Sampling interval for EMA [µs]
const float EMA_LP_FREQ = 200.;    // Low-pass filter cutoff frequency [Hz]
// Initialize with calibration values: p1_mA, p2_mA, p1_bitval, p2_bitval
R_Click R_click(PIN_CS_RCLICK, RT_Click_Calibration{4.04, 10.98, 806, 2191});

// Temperature & humidity sensor (SHT4x I2C)
Adafruit_SHT4x sht4;

// Photodetector configuration for droplet detection
const float PDA_R1 = 6710.0; // Voltage divider resistor [Ohm]
const float PDA_R2 = 3260.0; // Voltage divider resistor [Ohm]
const float PDA_THR = 4.5;   // Droplet detection threshold [V]

// ============================================================================
// T CLICK CONFIGURATION (proportional valve and pressure regulator)
// ============================================================================
T_Click valve(PIN_PROP_VALVE, RT_Click_Calibration{3.97, 19.90, 796, 3982});
T_Click pressure(PIN_PRES_REG, RT_Click_Calibration{3.97, 19.90, 796, 3982});

// Define default T Click values
const float max_mA = 20.0;
const float min_mA_valve = 12.0;
const float min_mA_pres_reg = 4.0;
const float default_valve = 12.0;   // mA
const float default_pressure = 4.0; // mA

// ============================================================================
// LED CONFIGURATION
// ============================================================================
// DotStar RGB LED (using board's built-in DotStar on pins 8 and 6)
Adafruit_DotStar led(1, PIN_DOTSTAR_DATA, PIN_DOTSTAR_CLK, DOTSTAR_BGR);

// LED color definitions (avoiding pure red for laser goggle compatibility)
// Colors use BGR format: Blue, Green, Red
const uint32_t COLOR_IDLE = 0x001000;       // Dim green - system ready
const uint32_t COLOR_VALVE_OPEN = 0x00FF00; // Bright green - valve active
const uint32_t COLOR_ERROR = 0xFF4000;      // Orange - error state
const uint32_t COLOR_READING = 0xFF0040;    // Cyan - taking measurement
const uint32_t COLOR_LASER = 0x100000;      // Dim blue - started detection
const uint32_t COLOR_DROPLET = 0xFF0000;    // Bright blue - droplet detected
const uint32_t COLOR_WAITING = 0x400040;   // Purple - waiting for valve opening
const uint32_t COLOR_RECEIVING = 0x100000; // Dim red - receiving dataset
const uint32_t COLOR_EXECUTING =
    0xFF0000;                        // Bright red - executing loaded dataset
const uint32_t COLOR_OFF = 0x000000; // Off

// ============================================================================
// LED HELPER FUNCTION
// ============================================================================

void setLedColor(uint32_t color) {
  // Set the DotStar LED to a specific color
  led.setPixelColor(0, color);
  led.show();
}

// ============================================================================
// INITIALIZATION
// ============================================================================

void setup() {
  // Configure pin modes
  pinMode(PIN_LED, OUTPUT);
  pinMode(PIN_VALVE, OUTPUT);
  pinMode(PIN_CS_RCLICK, INPUT);
  pinMode(PIN_TRIG, OUTPUT);
  pinMode(PIN_LASER, OUTPUT);
  pinMode(PIN_PDA, INPUT);

  // Set all outputs to safe initial state (off)
  digitalWrite(PIN_LED, LOW);
  digitalWrite(PIN_VALVE, LOW);
  digitalWrite(PIN_TRIG, LOW);
  digitalWrite(PIN_LASER, LOW);

  // Initialize T Clicks (proportional valve and pressure regulator)
  valve.begin();
  valve.set_mA(default_valve);

  pressure.begin();
  pressure.set_mA(default_pressure);

  // Initialize DotStar LED
  led.begin();
  led.setBrightness(255); // Set brightness to full
  led.show();             // Initialize all pixels to 'off'

  // Initialize serial communication at 115200 baud
  Serial.begin(115200);
  Serial.setTimeout(10); // Set timeout to 10ms instead of default 1000ms

  // TODO: Implement averaging?
  // Set ADC resolution for photodetector
  analogReadResolution(12); // 12-bit ADC (0-4095)

  // Initialize pressure sensor
  R_click.begin();

  // Initialize the SHT4x temperature & humidity sensor
  if (!sht4.begin()) {
    Serial.println("Failed to find SHT4x sensor!");
    // Blink orange for fatal error
    while (1) {
      setLedColor(COLOR_ERROR);
      delay(200);
      setLedColor(COLOR_OFF);
      delay(200);
    }
  }

  // Configure SHT4x for high precision, no heater
  sht4.setPrecision(SHT4X_HIGH_PRECISION);
  sht4.setHeater(SHT4X_NO_HEATER);

  // Show idle color to indicate system is ready
  setLedColor(COLOR_IDLE);
}

// ============================================================================
// VALVE AND SENSOR FUNCTIONS
// ============================================================================

void openValveTrigger() {
  // Open valve and trigger using direct PORT register access for speed
  // Equivalent to digitalWrite(PIN_VALVE, HIGH); digitalWrite(PIN_TRIG, HIGH);
  PORT->Group[g_APinDescription[PIN_VALVE].ulPort].OUTSET.reg =
      ((1 << g_APinDescription[PIN_VALVE].ulPin) |
       (1 << g_APinDescription[PIN_TRIG].ulPin));

  DEBUG_PRINTLN(
      "Solenoïd valve opened using openValveTrigger()"); // Valve opened
                                                         // confirmation (debug
                                                         // only for speed)
}

void closeValve() {
  // Close valve using direct PORT register access for speed
  // Equivalent to digitalWrite(PIN_VALVE, LOW);
  PORT->Group[g_APinDescription[PIN_VALVE].ulPort].OUTCLR.reg =
      (1 << g_APinDescription[PIN_VALVE].ulPin);

  DEBUG_PRINT(
      "Solenoïd valve closed using closeValve()"); // Valve closed confirmation
                                                   // (debug only for speed)
  Serial.println("!");
}

void startLaser() {
  // Turn on laser using direct PORT register access for speed
  // Equivalent to digitalWrite(PIN_LASER, HIGH);
  PORT->Group[g_APinDescription[PIN_LASER].ulPort].OUTSET.reg =
      (1 << g_APinDescription[PIN_LASER].ulPin);

  DEBUG_PRINTLN("Laser ON");
}

void stopLaser() {
  // Turn off laser using direct PORT register access for speed
  // Equivalent to digitalWrite(PIN_LASER, LOW);
  PORT->Group[g_APinDescription[PIN_LASER].ulPort].OUTCLR.reg =
      (1 << g_APinDescription[PIN_LASER].ulPin);

  DEBUG_PRINTLN("Laser OFF");
}

void stopTrigger() {
  // Turn off trigger using direct PORT register access for speed
  // Equivalent to digitalWrite(PIN_TRIG, LOW);
  PORT->Group[g_APinDescription[PIN_TRIG].ulPort].OUTCLR.reg =
      (1 << g_APinDescription[PIN_TRIG].ulPin);
}

void printError(const char *message) {
  // Print error message to serial for debugging
  Serial.print("ERROR: ");
  Serial.println(message);

  // Flash orange briefly to indicate error
  setLedColor(COLOR_ERROR);
  delay(300);
  setLedColor(COLOR_IDLE);
}

void readPressure(bool valveOpen) {
  // Read current pressure from R-Click sensor
  // Conversion formula: Pressure = 0.6249 * I[mA] - 2.4882
  // where I is the 4-20mA current output
  setLedColor(COLOR_READING); // Show color during reading
  Serial.print("P");
  Serial.print(0.62350602 * R_click.get_EMA_mA() - 2.51344790);
  Serial.println();
  // Restore LED color based on valve state
  setLedColor(valveOpen ? COLOR_VALVE_OPEN : COLOR_IDLE);
  DEBUG_PRINT("R Click bitvalue: ");
  DEBUG_PRINTLN(R_click.get_EMA_bitval());
}

void readTemperature(bool valveOpen) {
  // Read temperature and relative humidity from SHT4x sensor
  setLedColor(COLOR_READING); // Show color during reading
  sensors_event_t humidity, temp;
  sht4.getEvent(&humidity, &temp);

  // Send temperature reading
  Serial.print("T");
  Serial.println(temp.temperature);

  // Send humidity reading
  Serial.print("H");
  Serial.println(humidity.relative_humidity);

  // Restore LED color based on valve state
  setLedColor(valveOpen ? COLOR_VALVE_OPEN : COLOR_IDLE);
}

float readPhotodetector() {
  // Read photodetector voltage with resistor divider compensation
  int adcValue = analogRead(PIN_PDA);        // 0-4095 (12-bit)
  float voltage = (adcValue / 4095.0) * 3.3; // Convert to voltage
  float signalVoltage = voltage * ((PDA_R1 + PDA_R2) / PDA_R2);

  // DEBUG_PRINTLN(signalVoltage);

  return signalVoltage;
}

void resetDataArrays() {
  memset(time_array, 0, sizeof(time_array));
  memset(value_array, 0, sizeof(value_array));
  incomingCount = 0;
  // Added these three resets after testing, need reviewing!
  dataIndex = 0;
  sequenceIndex = 0;
  datasetDuration = 0;
}

// ============================================================================
// MAIN LOOP
// ============================================================================

void loop() {
  // Static variables persist across loop iterations
  static uint32_t duration = 0;     // How long valve should stay open [µs]
  static bool solValveOpen = false; // Tracks if valve is currently open
  static bool propValveOpen =
      false; // Tracks if proportional valve is currently open
  static bool performingTrigger = false; // Tracks if trigger pulse is active
  static bool detectingDroplet = false;  // Tracks if in droplet detection mode
  static bool belowThreshold = false;    // Tracks if signal is below threshold
  static bool waitingToOpenValve =
      false; // Tracks if waiting for delay before opening valve
  static uint32_t waitStartTime =
      0; // When waiting for valve opening started [µs]
  static uint32_t detectionStartTime =
      0; // When laser/detection was started [µs]
  static bool continuousDetection =
      false;                       // Tracks if in continuous detection mode
  static bool isExecuting = false; // Tracks if waiting to run loaded sequence
  static bool setPressure =
      false; // Tracks if pressure regulator has been set at least once

  // -------------------------------------------------------------------------
  // Handle trigger pulse timing
  // -------------------------------------------------------------------------
  // The trigger pulse is a short signal sent to peripheral devices when the
  // valve opens. It turns off after TRIGGER_WIDTH microseconds.
  if (performingTrigger && (micros() - tick >= TRIGGER_WIDTH)) {
    stopTrigger();
    performingTrigger = false;
  }

  // -------------------------------------------------------------------------
  // Handle valve timing
  // -------------------------------------------------------------------------
  // Close valve after duration (duration=0 means stay open)
  if (solValveOpen && duration > 0 && (micros() - tick >= duration)) {
    DEBUG_PRINTLN("Solenoïd valve closed after duration check.");
    closeValve();
    solValveOpen = false;

    // If in continuous detection mode, restart detection immediately
    if (continuousDetection) {
      // Turn on laser and restart detection
      startLaser();
      setLedColor(COLOR_LASER);
      detectingDroplet = true;
      belowThreshold = false;
      detectionStartTime = micros();
      DEBUG_PRINTLN("Restarting droplet detection");
    } else {
      setLedColor(COLOR_IDLE);
    }
  }

  // -------------------------------------------------------------------------
  // Update pressure sensor
  // -------------------------------------------------------------------------
  // Must be called regularly to maintain the exponential moving average
  R_click.poll_EMA();

  // -------------------------------------------------------------------------
  // Droplet detection monitoring
  // -------------------------------------------------------------------------
  if (detectingDroplet) {
    uint32_t elapsedSinceStart = micros() - detectionStartTime;

    // Only start checking photodiode after the configured delay
    if (elapsedSinceStart >= pda_delay) {
      float signalVoltage = readPhotodetector();

      // Falling edge: droplet detected (signal drops below threshold)
      if (!belowThreshold && signalVoltage < PDA_THR) {
        belowThreshold = true;
        waitingToOpenValve = true;
        waitStartTime = micros();

        // Turn off laser immediately when droplet is detected
        stopLaser();
        detectingDroplet = false;

        setLedColor(COLOR_DROPLET);
        DEBUG_PRINTLN("Droplet detected!");
      }
    }
  }

  // -------------------------------------------------------------------------
  // Handle delay and valve opening (for both droplet detection and O command)
  // -------------------------------------------------------------------------
  if (waitingToOpenValve && !solValveOpen) {
    uint32_t elapsed = micros() - waitStartTime;

    // Show purple LED during delay period
    if (elapsed < tick_delay) {
      setLedColor(COLOR_WAITING);
    }

    // Open valve after delay has elapsed
    if (elapsed >= tick_delay) {
      // Open valve and trigger
      openValveTrigger();

      setLedColor(COLOR_VALVE_OPEN);
      solValveOpen = true;
      performingTrigger = true;
      tick = micros();

      // Reset flag after opening valve
      waitingToOpenValve = false;
    }
  }

  // Execute loaded dataset
  if (isExecuting) {
    // Caclulate time since start execution
    uint32_t now = (micros() - runCalltTime); // Time since RUN is called [µs]

    // If valve isn't open and timing of first datapoint has been reached open
    // solenoid valve
    if (!solValveOpen && sequenceIndex == 0 && (now / 1000) >= time_array[0]) {
      openValveTrigger();       // Open solenoid valve and trigger
      performingTrigger = true; // Set trigger flag
      tick = micros();
      DEBUG_PRINT("Time to opening solenoid valve: ");
      DEBUG_PRINT(now / 1000);
      DEBUG_PRINTLN(" ms.");
      solValveOpen = true;
      propValveOpen = true;
      setLedColor(COLOR_VALVE_OPEN);
    }

    int32_t solValveCloseTime = datasetDuration + (valve_delay_close / 1000);

    // Check if solenoid valve needs to be closed
    if (solValveOpen && (now / 1000) >= (uint32_t)solValveCloseTime) {
      closeValve();
      solValveOpen = false;
      DEBUG_PRINT("Solenoïd valve closed after: ");
      DEBUG_PRINT(now / 1000);
      DEBUG_PRINT("ms, time goal: ");
      DEBUG_PRINTLN(solValveCloseTime);
    }

    // If time since start execution >= (dataset index time + valve timing
    // delay) -> set mA value of valve to dataset index value
    if ((now / 1000) >= time_array[sequenceIndex] + (valve_delay_open / 1000)) {
      // If whole dataset has been executed, exit execution state
      if (sequenceIndex >= dataIndex) {
        isExecuting = false;         // Reset executing flag
        sequenceIndex = 0;           // Reset dataset index
        valve.set_mA(default_valve); // Close proportional valve
        propValveOpen = false;
        setLedColor(COLOR_OFF);
        return;
      } else {
        valve.set_mA(value_array[sequenceIndex]);
        // This Serial.print(); shows that the timing is accurate to within a ms
        // and that the DEBUG prints delay the code Serial.println((now / 1000)
        // - time_array[sequenceIndex]); Debug print dataset execution, expected
        // time and value vs actual time and value
        DEBUG_PRINT("Sequence index: ");
        DEBUG_PRINT(sequenceIndex);
        DEBUG_PRINT(", time elapsed: ");
        DEBUG_PRINT(now / 1000);
        DEBUG_PRINT("ms, time goal: ");
        DEBUG_PRINT(time_array[sequenceIndex]);
        DEBUG_PRINT("ms, difference: ");
        DEBUG_PRINT((now / 1000) - time_array[sequenceIndex]);
        DEBUG_PRINT("ms, setpoint: ");
        DEBUG_PRINTLN(value_array[sequenceIndex]);

        sequenceIndex++;
      }
    }
  }

  // =========================================================================
  // Process serial commands
  // =========================================================================
  if (sc.available()) {
    char *command =
        sc.getCommand(); // Pointer to memory location of serial buffer contents

    DEBUG_PRINT("CMD: ");
    DEBUG_PRINTLN(command);

    if (strncmp(command, "SV", 2) == 0) {
      // Command: SV <mA>
      // Set milli amps of proportional valve to <mA>

      float current = parseFloatInString(
          command, 2); // Parse float from char array 'command'

      // Handle out of allowable range inputs, defaults to specified value
      if (!current || current < min_mA_valve || current > max_mA) {
        valve.set_mA(default_valve);
        DEBUG_PRINT("ERROR: input outside of allowable range (");
        DEBUG_PRINT(min_mA_valve);
        DEBUG_PRINT(" - ");
        DEBUG_PRINT(max_mA);
        DEBUG_PRINTLN("), valve set to default value.");
        setLedColor(COLOR_ERROR);
        delay(300);
        setLedColor(COLOR_OFF);

        // Set T_Click to input mA
      } else {
        valve.set_mA(current);
        DEBUG_PRINT("Last set bitvalue of proportional valve: ");
        DEBUG_PRINTLN(valve.get_last_set_bitval());
      }

    } else if (strncmp(command, "SP", 2) == 0) {
      // Command: SP <mA>
      // Set milli amps of pressure regulator to <mA>

      if (!setPressure) {
        setPressure = true;
      }

      float bar =
          parseFloatInString(command, 2); // Parse float from char array command
      float current = (bar + 2.48821429) / 0.62242857;

      // Handle out of allowable range inputs, defaults to specified value
      // TODO: Put calculation in function
      if (!current || current < min_mA_pres_reg || current > max_mA) {
        pressure.set_mA(default_pressure);
        DEBUG_PRINT("ERROR: input outside of allowable range (");
        DEBUG_PRINT(min_mA_pres_reg);
        DEBUG_PRINT(" - ");
        DEBUG_PRINT(max_mA);
        DEBUG_PRINTLN(" mA), valve set to default value.");
        setLedColor(COLOR_ERROR);
        delay(300);
        setLedColor(COLOR_OFF);

        // Set T_Click to input mA
      } else {
        pressure.set_mA(current);
        DEBUG_PRINT("Last set bitvalue of pressure regulator: ");
        DEBUG_PRINTLN(pressure.get_last_set_bitval());
      }

    } else if (strncmp(command, "SHOW", 4) == 0) {

      if (dataIndex == 0) {
        Serial.println("No dataset in memory! LOAD one first.");
      } else {
        Serial.print("Saved dataset is: ");
        Serial.print(incomingCount);
        Serial.print(" datapoints long and takes ");
        Serial.print(datasetDuration);
        Serial.println(" ms.");
      }

    } else if (strncmp(command, "LOAD", 4) == 0) {
      // Parse incomming dataset. Command: "LOAD <N_datapoints>
      // <Time0>,<mA0>,<Time1>,<mA1>,<TimeN>,<mAN>"

      setLedColor(COLOR_RECEIVING);

      const char *delim = ","; // Serial dataset delimiter

      if (strlen(command) < 6) {
        DEBUG_PRINTLN("ERROR: \"LOAD\" command is not followed by dataset");
        setLedColor(COLOR_ERROR);
        delay(300);
        setLedColor(COLOR_OFF);
        return;
      }

      // read dataset length from char 5 until space (_)
      // "LOAD_<length>_<dataset>" and instantialize position to start reading
      // data from in strtok
      char *idx = strtok(command + 5, " ");
      incomingCount = atoi(idx); // Dataset length (int)

      idx = strtok(NULL, " ");
      datasetDuration = atoi(idx); // Dataset duration

      // Check if data length is acceptable
      if (incomingCount > MAX_DATA_LENGTH || incomingCount <= 0) {
        DEBUG_PRINT("ERROR: data length is not allowed: 0 < N < ");
        DEBUG_PRINT(MAX_DATA_LENGTH);
        DEBUG_PRINTLN(", upload new dataset!");
        resetDataArrays();
        setLedColor(COLOR_ERROR);
        delay(300);
        setLedColor(COLOR_OFF);
        return;
        // Check if dataset duration minus valve delay is not negative (needs to
        // be compared to uint32_t later)
      } else if ((datasetDuration + (valve_delay_close / 1000)) < 0) {
        DEBUG_PRINT("ERROR: dataset duration is too short, must be at least ");
        DEBUG_PRINT(-valve_delay_close / 1000);
        DEBUG_PRINTLN(" ms, upload new dataset!");
        resetDataArrays();
        setLedColor(COLOR_ERROR);
        delay(300);
        setLedColor(COLOR_OFF);
        return;
      }

      dataIndex = 0; // Used later to only read valuable data from data arrays

      // Parsing rest of the dataset after handshake
      for (int i = 0; i < incomingCount; i++) {

        idx = strtok(NULL, delim); // Get next item from buffer (str_cmd). This
                                   // item is the timestamp
        // If the item is NULL, break
        if (idx == NULL) {
          DEBUG_PRINT("ERROR: token was NULL, breaking CSV parsing. Upload new "
                      "dataset! (error at data index: ");
          DEBUG_PRINT(dataIndex);
          DEBUG_PRINTLN(")");
          resetDataArrays();
          setLedColor(COLOR_ERROR);
          delay(300);
          setLedColor(COLOR_OFF);
          break;
        }
        // Convert incoming csv buffer index from string to int and add to time
        // array
        time_array[i] = atoi(idx);

        idx = strtok(
            NULL,
            delim); // Get next csv buffer index. This item is the mA value
        // Check again if item is not NULL
        if (idx == NULL) {
          DEBUG_PRINT("ERROR: token was NULL, breaking CSV parsing. Upload new "
                      "dataset! (data index: ");
          DEBUG_PRINT(dataIndex);
          DEBUG_PRINTLN(")");
          resetDataArrays();
          setLedColor(COLOR_ERROR);
          delay(300);
          setLedColor(COLOR_OFF);
          break;
        }
        // Convert incoming csv buffer index from string to float and add to
        // value array
        value_array[i] = parseFloatInString(idx, 0);

        // Debug print whole received dataset
        DEBUG_PRINT("Timestamp: ");
        DEBUG_PRINT(time_array[i]);
        DEBUG_PRINT(", mA: ");
        DEBUG_PRINTLN(value_array[i]);

        // Increase working index, used later to only read valuable data from
        // data arrays
        dataIndex++;
      }

      // LED color off when whole dataset is received
      setLedColor(COLOR_OFF);

    } else if (strncmp(command, "RUN", 3) == 0) {
      if (dataIndex == 0) {
        printError("Dataset is empty! Upload first using LOAD command.");
        setLedColor(COLOR_ERROR);
        delay(300);
        setLedColor(COLOR_OFF);
      } else if (!setPressure) {
        printError(
            "Pressure regulator not set! Set it first using SP command.");
      } else {
        isExecuting = true;
        runCalltTime = micros();
        sequenceIndex = 0;
        setLedColor(COLOR_EXECUTING);
      }

    } else if (strncmp(command, "O", 1) == 0) {
      // Command: O or O <duration_ms>
      // O = open indefinitely, O <ms> = open for specified time

      if (strlen(command) == 1) {
        duration = 0; // 0 means stay open
        DEBUG_PRINTLN("Opening valve indefinitely");
      } else {
        duration = 1000 * parseFloatInString(command, 1);
        DEBUG_PRINT("Opening valve for ");
        DEBUG_PRINT(duration);
        DEBUG_PRINTLN(" µs");
      }

      // Start waiting period before opening valve
      waitingToOpenValve = true;
      waitStartTime = micros();

      if (tick_delay > 0) {
        setLedColor(COLOR_WAITING);
        DEBUG_PRINT("Waiting ");
        DEBUG_PRINT(tick_delay);
        DEBUG_PRINTLN(" µs before opening");
      } else {
        // If no delay, proceed immediately in next loop iteration
        setLedColor(COLOR_VALVE_OPEN);
      }

    } else if (strncmp(command, "C", 1) == 0) {
      // Command: C
      // Manually close valve (override)
      closeValve();
      solValveOpen = false;

      // Stop detection if it was running
      if (detectingDroplet) {
        stopLaser();
        detectingDroplet = false;
      }

      // Stop continuous detection mode
      continuousDetection = false;

      setLedColor(COLOR_IDLE);

    } else if (strncmp(command, "D", 1) == 0) {
      // Command: D or D <duration_ms>
      // D = detect droplet and open indefinitely, D <ms> = open for specified
      // time

      if (strlen(command) == 1) {
        duration = 0; // 0 means stay open
        DEBUG_PRINTLN("Droplet detection: valve will stay open");
      } else {
        duration = 1000 * parseIntInString(command, 1);
        DEBUG_PRINT("Droplet detection: valve will open for ");
        DEBUG_PRINT(duration);
        DEBUG_PRINTLN(" µs");
      }

      // Turn on laser
      startLaser();

      setLedColor(COLOR_LASER);
      detectingDroplet = true;
      belowThreshold = false;
      detectionStartTime = micros();
      continuousDetection = true; // Enable continuous detection mode

      DEBUG_PRINTLN("Detecting droplets");

    } else if (strncmp(command, "L", 1) == 0) {
      // Command: L <delay_us>
      // Set delay before opening valve (applies to both O and D commands)
      tick_delay = parseIntInString(command, 1);
      DEBUG_PRINT("Delay before opening valve: ");
      DEBUG_PRINT(tick_delay);
      DEBUG_PRINTLN(" µs");

    } else if (strncmp(command, "P?", 2) == 0) {
      // Command: P?
      // Read and return current pressure
      readPressure(solValveOpen);

    } else if (strncmp(command, "T?", 2) == 0) {
      // Command: T?
      // Read and return temperature & humidity
      readTemperature(solValveOpen);

    } else if (strncmp(command, "?", 1) == 0) {
      // Command: ?
      // Print help menu
      DEBUG_PRINTLN("\n=== Available Commands ===");
      DEBUG_PRINTLN("O       - Open valve indefinitely");
      DEBUG_PRINTLN("O <ms>  - Open valve for <ms> milliseconds (e.g., O 100)");
      DEBUG_PRINTLN("C       - Close valve immediately");
      DEBUG_PRINTLN("D       - Detect droplet, open valve indefinitely");
      DEBUG_PRINTLN("D <ms>  - Detect droplet, open for <ms> milliseconds");
      DEBUG_PRINTLN(
          "L <us>  - Set delay before valve opening to <us> microseconds");
      DEBUG_PRINTLN("SV <mA> - Set proportional valve milliamps to <mA>");
      DEBUG_PRINTLN("SP <bar> - Set pressure regulator to <bar>");
      DEBUG_PRINTLN("LOAD <N_datapoints> <csv dataset> - Load dataset, format: "
                    "<ms0>,<mA0>,<ms1>,<mA1>,<msN>,<mAN>");
      DEBUG_PRINTLN("RUN     - Execute loaded dataset");
      DEBUG_PRINTLN("P?      - Read pressure");
      DEBUG_PRINTLN("T?      - Read temperature & humidity");
      DEBUG_PRINTLN("S?      - System status");
      DEBUG_PRINTLN("?       - Show this help");

    } else if (strncmp(command, "S?", 2) == 0) {
      // Command: S?
      // Print system status (debug only)
      DEBUG_PRINTLN("\n=== System Status ===");
      DEBUG_PRINT("Solenoïd valve: ");
      DEBUG_PRINTLN(solValveOpen ? "OPEN" : "CLOSED");
      if (solValveOpen) {
        DEBUG_PRINT("Time remaining: ");
        uint32_t elapsed = micros() - tick;
        if (elapsed < duration) {
          DEBUG_PRINT((duration - elapsed) / 1000);
          DEBUG_PRINTLN(" ms");
        } else if (duration == 0) {
          DEBUG_PRINTLN("inf");
        }
      }
      DEBUG_PRINT("Proportional valve: ");
      DEBUG_PRINTLN(propValveOpen ? "OPEN" : "CLOSED");
      DEBUG_PRINT("Dataset in memory: ");
      DEBUG_PRINTLN((dataIndex == 0) ? "FALSE" : "TRUE");
      DEBUG_PRINT("Executing dataset: ");
      DEBUG_PRINTLN(isExecuting ? "TRUE" : "FALSE");
      DEBUG_PRINT("Trigger: ");
      DEBUG_PRINTLN(performingTrigger ? "ACTIVE" : "IDLE");
      DEBUG_PRINT("Droplet detection: ");
      DEBUG_PRINTLN(detectingDroplet ? "ACTIVE" : "IDLE");
      if (detectingDroplet) {
        DEBUG_PRINT("Photodetector: ");
        DEBUG_PRINT(readPhotodetector());
        DEBUG_PRINTLN(" V");
      }
      DEBUG_PRINT("Delay before opening valve: ");
      DEBUG_PRINT(tick_delay);
      DEBUG_PRINTLN(" µs");
      DEBUG_PRINT("Photodiode detection delay: ");
      DEBUG_PRINT(pda_delay);
      DEBUG_PRINTLN(" µs");
      DEBUG_PRINT("Pressure (raw): ");
      DEBUG_PRINT(R_click.get_EMA_mA());
      DEBUG_PRINTLN(" mA");
      DEBUG_PRINT("Uptime: ");
      DEBUG_PRINT(millis() / 1000);
      DEBUG_PRINTLN(" s");

    } else {
      // Unknown command
      printError("Unknown command");
    }
  }
}
