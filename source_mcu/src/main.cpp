/*
 * Cough Machine Control System
 *
 * Controls a solenoid valve for atomisation experiments with precise timing.
 * Monitors pressure and environmental conditions.
 */

#include "MIKROE_4_20mA_RT_Click.h"
#include <Adafruit_DotStar.h>
#include <Adafruit_SHT4x.h>
#include <Arduino.h>

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
const int PIN_VALVE = 7;     // MOSFET gate pin for solenoid valve control
const int PIN_CS_RCLICK = 2; // Chip select for R-Click pressure sensor (SPI)
const int PIN_TRIG = 9; // Trigger output for peripheral devices synchronization
const int PIN_LASER = 12; // Laser MOSFET gate pin for droplet detection
const int PIN_PDA = A2;   // Analog input from photodetector
// Note: PIN_DOTSTAR_DATA and PIN_DOTSTAR_CLK are already defined in variant.h

// ============================================================================
// TIMING PARAMETERS
// ============================================================================
const uint32_t TRIGGER_WIDTH = 10000; // Trigger pulse width [µs] (10ms)
uint32_t tick = 0;                    // Timestamp for timing events [µs]
uint32_t tick_delay = 0;              // Delay before opening valve [µs]
uint32_t pda_delay = 10000; // Delay before photodiode starts detecting [µs]

// ============================================================================
// SENSOR CONFIGURATION
// ============================================================================
// Pressure sensor (4-20mA R-Click) with exponential moving average filtering
const uint32_t EMA_INTERVAL = 500; // Sampling interval for EMA [µs]
const float EMA_LP_FREQ = 200.;    // Low-pass filter cutoff frequency [Hz]
// Initialize with calibration values: p1_mA, p2_mA, p1_bitval, p2_bitval
R_Click R_click(PIN_CS_RCLICK, RT_Click_Calibration{3.99, 9.75, 795, 1943});

// Temperature & humidity sensor (SHT4x I2C)
Adafruit_SHT4x sht4;

// Photodetector configuration for droplet detection
const float PDA_R1 = 6710.0; // Voltage divider resistor [Ohm]
const float PDA_R2 = 3260.0; // Voltage divider resistor [Ohm]
const float PDA_THR = 1.5;   // Droplet detection threshold [V]

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
const uint32_t COLOR_WAITING = 0x400040; // Purple - waiting for valve opening
const uint32_t COLOR_OFF = 0x000000;     // Off

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

  // Initialize DotStar LED
  led.begin();
  led.setBrightness(255); // Set brightness to full
  led.show();             // Initialize all pixels to 'off'

  // Initialize serial communication at 115200 baud
  Serial.begin(115200);

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
      "Valve opened"); // Valve opened confirmation (debug only for speed)
}

void closeValve() {
  // Close valve using direct PORT register access for speed
  // Equivalent to digitalWrite(PIN_VALVE, LOW);
  PORT->Group[g_APinDescription[PIN_VALVE].ulPort].OUTCLR.reg =
      (1 << g_APinDescription[PIN_VALVE].ulPin);

  DEBUG_PRINTLN(
      "Valve closed"); // Valve closed confirmation (debug only for speed)
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
  Serial.print(0.6249 * R_click.get_EMA_mA() - 2.4882);
  Serial.println();
  // Restore LED color based on valve state
  setLedColor(valveOpen ? COLOR_VALVE_OPEN : COLOR_IDLE);
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

  DEBUG_PRINTLN(signalVoltage);

  return signalVoltage;
}

// ============================================================================
// MAIN LOOP
// ============================================================================

void loop() {
  // Static variables persist across loop iterations
  static uint32_t duration = 0;          // How long valve should stay open [µs]
  static bool valveOpen = false;         // Tracks if valve is currently open
  static bool performingTrigger = false; // Tracks if trigger pulse is active
  static bool detectingDroplet = false;  // Tracks if in droplet detection mode
  static bool dropletDetected = false;   // Tracks if droplet has been detected
  static bool belowThreshold = false;    // Tracks if signal is below threshold
  static uint32_t dropletDetectTime = 0; // When droplet was detected [µs]
  static bool waitingToOpen =
      false; // Tracks if waiting for delay before opening
  static uint32_t openCommandTime = 0; // When open command was received [µs]
  static uint32_t detectionStartTime =
      0; // When laser/detection was started [µs]

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
  if (valveOpen && duration > 0 && (micros() - tick >= duration)) {
    closeValve();
    valveOpen = false;

    // Stop droplet detection after valve closes
    // TODO: Could add a separate mode for continuous detection with repeated
    // experiments
    if (detectingDroplet) {
      stopLaser();
      detectingDroplet = false;
    }

    setLedColor(COLOR_IDLE);
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
        dropletDetected = true;
        dropletDetectTime = micros();

        // Turn off laser immediately when droplet is detected
        stopLaser();
        detectingDroplet = false;

        setLedColor(COLOR_DROPLET);
        DEBUG_PRINTLN("Droplet detected!");
      }
    }
  }

  // -------------------------------------------------------------------------
  // Handle delay and valve opening after droplet detection
  // -------------------------------------------------------------------------
  if (dropletDetected && !valveOpen) {
    uint32_t elapsed = micros() - dropletDetectTime;

    // Show purple LED during delay period
    if (elapsed < tick_delay) {
      setLedColor(COLOR_WAITING);
    }

    // Open valve after delay has elapsed
    if (elapsed >= tick_delay) {
      // Open valve and trigger
      openValveTrigger();

      setLedColor(COLOR_VALVE_OPEN);
      valveOpen = true;
      performingTrigger = true;
      tick = micros();

      dropletDetected = false; // Reset after opening valve
    }
  }

  // -------------------------------------------------------------------------
  // Handle delay and valve opening after O command
  // -------------------------------------------------------------------------
  if (waitingToOpen && !valveOpen) {
    uint32_t elapsed = micros() - openCommandTime;

    // Show purple LED during delay period
    if (elapsed < tick_delay) {
      setLedColor(COLOR_WAITING);
    }

    // Open valve after delay has elapsed
    if (elapsed >= tick_delay) {
      // Open valve and trigger
      openValveTrigger();

      setLedColor(COLOR_VALVE_OPEN);
      valveOpen = true;
      performingTrigger = true;
      tick = micros();

      waitingToOpen = false; // Reset after opening valve
    }
  }

  // -------------------------------------------------------------------------
  // Process serial commands
  // -------------------------------------------------------------------------
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim(); // Remove any whitespace

    DEBUG_PRINT("CMD: ");
    DEBUG_PRINTLN(command);

    if (command == "O" || command.startsWith("O ")) {
      // Command: O or O <duration_ms>
      // O = open indefinitely, O <ms> = open for specified time

      if (command == "O") {
        duration = 0; // 0 means stay open
        DEBUG_PRINTLN("Opening valve indefinitely");
      } else {
        duration = 1000 * command.substring(2).toInt();
        DEBUG_PRINT("Opening valve for ");
        DEBUG_PRINT(duration);
        DEBUG_PRINTLN(" µs");
      }

      // Start waiting period before opening valve
      waitingToOpen = true;
      openCommandTime = micros();

      if (tick_delay > 0) {
        setLedColor(COLOR_WAITING);
        DEBUG_PRINT("Waiting ");
        DEBUG_PRINT(tick_delay);
        DEBUG_PRINTLN(" µs before opening");
      } else {
        // If no delay, proceed immediately in next loop iteration
        setLedColor(COLOR_VALVE_OPEN);
      }

    } else if (command == "C") {
      // Command: C
      // Manually close valve (override)
      closeValve();
      valveOpen = false;

      // Stop detection if it was running
      if (detectingDroplet) {
        stopLaser();
        detectingDroplet = false;
      }

      setLedColor(COLOR_IDLE);

    } else if (command == "D" || command.startsWith("D ")) {
      // Command: D or D <duration_ms>
      // D = detect droplet and open indefinitely, D <ms> = open for specified
      // time

      if (command == "D") {
        duration = 0; // 0 means stay open
        DEBUG_PRINTLN("Droplet detection: valve will stay open");
      } else {
        duration = 1000 * command.substring(2).toInt();
        DEBUG_PRINT("Droplet detection: valve will open for ");
        DEBUG_PRINT(duration);
        DEBUG_PRINTLN(" µs");
      }

      // Turn on laser
      startLaser();

      setLedColor(COLOR_LASER);
      detectingDroplet = true;
      dropletDetected = false;
      belowThreshold = false;
      detectionStartTime = micros();

      DEBUG_PRINTLN("Detecting droplets");

    } else if (command.startsWith("L ")) {
      // Command: L <delay_us>
      // Set delay before opening valve (applies to both O and D commands)
      tick_delay = command.substring(2).toInt();
      DEBUG_PRINT("Delay before opening valve: ");
      DEBUG_PRINT(tick_delay);
      DEBUG_PRINTLN(" µs");

    } else if (command == "P?") {
      // Command: P?
      // Read and return current pressure
      readPressure(valveOpen);

    } else if (command == "T?") {
      // Command: T?
      // Read and return temperature & humidity
      readTemperature(valveOpen);

    } else if (command == "?") {
      // Command: ?
      // Print help menu
      DEBUG_PRINTLN("\n=== Available Commands ===");
      DEBUG_PRINTLN("O      - Open valve indefinitely");
      DEBUG_PRINTLN("O <ms> - Open valve for <ms> milliseconds (e.g., O 100)");
      DEBUG_PRINTLN("C      - Close valve immediately");
      DEBUG_PRINTLN("D      - Detect droplet, open valve indefinitely");
      DEBUG_PRINTLN("D <ms> - Detect droplet, open for <ms> milliseconds");
      DEBUG_PRINTLN(
          "L <us> - Set delay before valve opening to <us> microseconds");
      DEBUG_PRINTLN("P?     - Read pressure");
      DEBUG_PRINTLN("T?     - Read temperature & humidity");
      DEBUG_PRINTLN("S?     - System status");
      DEBUG_PRINTLN("?      - Show this help");

      // TODO: Implement continuous droplet detection & triggering

    } else if (command == "S?") {
      // Command: S?
      // Print system status (debug only)
      DEBUG_PRINTLN("\n=== System Status ===");
      DEBUG_PRINT("Valve: ");
      DEBUG_PRINTLN(valveOpen ? "OPEN" : "CLOSED");
      if (valveOpen) {
        DEBUG_PRINT("Time remaining: ");
        uint32_t elapsed = micros() - tick;
        if (elapsed < duration) {
          DEBUG_PRINT((duration - elapsed) / 1000);
          DEBUG_PRINTLN(" ms");
        }
      }
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
