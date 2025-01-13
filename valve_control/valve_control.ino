const int ledPin = LED_BUILTIN;  // Use the built-in LED
const int powerPin = 5;  // Use pin 5 for 5.0V control
int duration = 0;  // Variable to store the duration
bool timeSet = false;  // Flag to check if time has been set
bool valveOpen = false;  // Flag to check if the valve is open

void setup() {
  pinMode(ledPin, OUTPUT);  // Set the LED pin as an output
  pinMode(powerPin, OUTPUT);  // Set the power pin as an output
  digitalWrite(ledPin, LOW);  // Ensure the LED is off initially
  digitalWrite(powerPin, LOW);  // Ensure the power pin is off initially
  Serial.begin(9600);  // Start the serial communication
  Serial.println("Arduino is ready");
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    Serial.print("Received command: ");
    Serial.println(command);
    if (command.startsWith("TIME")) {
      duration = command.substring(5).toInt();
      Serial.print("Setting time to: ");
      Serial.print(duration);
      Serial.println(" ms");
      timeSet = true;
    } else if (command == "OPEN") {
      if (timeSet) {
        Serial.println("Valve is OPEN");
        digitalWrite(ledPin, HIGH);  // Turn the LED on
        digitalWrite(powerPin, HIGH);  // Turn the power pin on
        valveOpen = true;
        unsigned long startTime = millis();
        while (millis() - startTime < duration) {
          if (Serial.available() > 0) {
            String interruptCommand = Serial.readStringUntil('\n');
            if (interruptCommand == "CLOSE" || interruptCommand == "RESET") {
              Serial.println("Manually closing the valve");
              digitalWrite(ledPin, LOW);  // Turn the LED off
              digitalWrite(powerPin, LOW);  // Turn the power pin off
              Serial.println("Valve is CLOSED");
              valveOpen = false;
              if (interruptCommand == "RESET") {
                Serial.println("Closing valve and resetting time");
                timeSet = false;
              }
              break;
            }
          }
        }
        if (valveOpen) {
          digitalWrite(ledPin, LOW);  // Turn the LED off
          digitalWrite(powerPin, LOW);  // Turn the power pin off
          Serial.println("Valve is CLOSED");
          valveOpen = false;
        }
      } else {
        Serial.println("Error: TIME not set");
        // Blink the LED to indicate an error
        for (int i = 0; i < 5; i++) {
          digitalWrite(ledPin, HIGH);
          delay(100);
          digitalWrite(ledPin, LOW);
          delay(100);
        }
      }
    } else if (command == "CLOSE") {
      if (valveOpen) {
        Serial.println("Manually closing the valve");
        digitalWrite(ledPin, LOW);  // Turn the LED off
        digitalWrite(powerPin, LOW);  // Turn the power pin off
        Serial.println("Valve is CLOSED");
        valveOpen = false;
      } else {
        Serial.println("Valve is already closed");
      }
    } else if (command == "RESET") {
      Serial.println("Resetting time");
      timeSet = false;
    } else {
      Serial.println("Error: Unknown command");
    }
  }
}