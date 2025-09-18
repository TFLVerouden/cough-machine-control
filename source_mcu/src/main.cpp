#include <Arduino.h>
#include <Adafruit_SHT4x.h>
#include "MIKROE_4_20mA_RT_Click.h"

// Connections
const int PIN_VALVE = 7;         // Pin to control the MOSFET for valve control
const int PIN_CS_RCLICK = 2;          // Cable select pin for RClick pressure reading
const int PIN_TRIG = 9;           // Pin for trigger out 

// Trigger parameters
const uint32_t TRIGGER_WIDTH = 10000; // Trigger pulse width [µs]
uint32_t tick = 0;  // Trigger for the tick interval
bool performingTrigger = false;

// Valve opening logic parameters (these are not static as they are called by functions)
uint32_t duration = 0;                // Variable to store the duration
bool valveOpen = false;          // Flag to check if the valve is open

// Exponential moving average (EMA) parameters & calibration of the R Click readings
const uint32_t EMA_INTERVAL = 500; // Desired oversampling interval [µs]
const float EMA_LP_FREQ = 200.;      // Low-pass filter cut-off frequency [Hz]
R_Click R_click(PIN_CS_RCLICK, RT_Click_Calibration{3.99, 9.75, 795, 1943});

// Humidity+temperature sensor
Adafruit_SHT4x sht4;  // Declare the T+RH sensor object

enum State {
  IDLE,
  ERROR
};

State currentState = IDLE;

void setup() {
  pinMode(PIN_LED, OUTPUT);
  pinMode(PIN_VALVE, OUTPUT);
  pinMode(PIN_CS_RCLICK, INPUT);
  pinMode(PIN_TRIG, OUTPUT);

  digitalWrite(PIN_LED, LOW);
  digitalWrite(PIN_VALVE, LOW);
  digitalWrite(PIN_TRIG, LOW);
  
  Serial.begin(115200);
  R_click.begin();

  // Initialize the SHT4x sensor
  if (!sht4.begin()) {
    Serial.println("Failed to find SHT4x sensor!");
    while (1) delay(10);  // Halt execution if sensor is not found
  }

  // Set sensor precision and heater mode
  sht4.setPrecision(SHT4X_HIGH_PRECISION);
  sht4.setHeater(SHT4X_NO_HEATER);
} 

void closeValve() {
  // digitalWrite(PIN_VALVE, LOW);
  PORT->Group[g_APinDescription[PIN_VALVE].ulPort].OUTCLR.reg =
      (1 << g_APinDescription[PIN_VALVE].ulPin);
  digitalWrite(PIN_LED, LOW);
  valveOpen = false;
  Serial.println("!");
}

// TODO: Print error rather than occupying microcontroller with error state -> state machine can then be removed altogether
void blinkError() {
  for (int i = 0; i < 5; i++) {
    digitalWrite(PIN_LED, HIGH);
    delay(100);
    digitalWrite(PIN_LED, LOW);
    delay(100);
  }
}

void readPressure() {
  // Reads out R-click board and converts to pressure
  Serial.print("P");
  Serial.print(0.6249*R_click.get_EMA_mA() - 2.4882);
  Serial.println();
}

void readTemperature() {
  // Reads Temperature and RH on command;
  sensors_event_t humidity, temp;
  
  // Read temperature and humidity
  sht4.getEvent(&humidity, &temp);
  
  Serial.print("T");
  Serial.println(temp.temperature);
  
  Serial.print("RH");
  Serial.println(humidity.relative_humidity);
}

 // TODO: Move to loop so variables don't have to be global
void handleCommand(String command) {
  if (command.startsWith("O")) {
    // Extract duration (us) from command (ms)
    duration = 1000 * command.substring(2).toInt();
    if (duration > 0) {

      // digitalWrite(PIN_VALVE, HIGH);
      // digitalWrite(PIN_TRIG, HIGH);
      PORT->Group[g_APinDescription[PIN_VALVE].ulPort].OUTSET.reg =
          ((1 << g_APinDescription[PIN_VALVE].ulPin) |
           (1 << g_APinDescription[PIN_TRIG].ulPin));
      
      digitalWrite(PIN_LED, HIGH);
      valveOpen = true;
      performingTrigger = true;

      tick = micros();
    } else {
      currentState = ERROR;
    }

  } else if (command == "C") {
    closeValve();

  } else if (command == "P?") {
    readPressure();

  } else if (command == "T?") {
    readTemperature();
  } else {
    currentState = ERROR;
  }
}

void loop() {
  // Handle trigger
  if (performingTrigger && (micros() - tick >= TRIGGER_WIDTH)) {
    // digitalWrite(PIN_TRIG, LOW);
    PORT->Group[g_APinDescription[PIN_TRIG].ulPort].OUTCLR.reg =
      (1 << g_APinDescription[PIN_TRIG].ulPin);
    performingTrigger = false;
  }

  // Handle MOSFET valve control
  if (valveOpen && (micros() - tick >= duration)) {
    closeValve();
    valveOpen = false;
    currentState = IDLE;
  }

  // Poll R-click board (pressure sensor)
  R_click.poll_EMA();

  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    handleCommand(command);
  }

  switch (currentState) {
    case IDLE:
      // Do nothing
      break;

    case ERROR:
      blinkError();
      currentState = IDLE;
      break;
  }
}

