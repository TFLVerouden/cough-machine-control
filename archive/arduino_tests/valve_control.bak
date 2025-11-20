const int ledPin = LED_BUILTIN;  // Use the built-in LED
const int powerPin = 5;  // Use pin 5 for 5.0V control
int duration = 0;  // Variable to store the duration
bool timeSet = false;  // Flag to check if time has been set
bool valveOpen = false;  // Flag to check if the valve is open

void setup() {
  // Set the outputs
  pinMode(ledPin, OUTPUT);
  pinMode(powerPin, OUTPUT);

  // Ensure initial off states
  digitalWrite(ledPin, LOW);
  digitalWrite(powerPin, LOW);

  // Start the serial communication
  Serial.begin(9600); 
  Serial.println("ARDUINO ready");
}

void loop() {

  // If serial communication has started...
  if (Serial.available() > 0) {

    // ...await commands
    String command = Serial.readStringUntil('\n');
    Serial.print("ARDUINO received command: ");
    Serial.println(command);

    if (command.startsWith("TIME")) {

      // Get duration from TIME command
      duration = command.substring(5).toInt();
      Serial.print("ARDUINO setting opening duration to: ");
      Serial.print(duration);
      Serial.println(" ms");
      timeSet = true;

    } else if (command == "OPEN") {

      // Accept the OPEN command if the duration has been set
      if (timeSet) {

        // Turn on pins
        Serial.println("ARDUINO opened the valve");
        digitalWrite(ledPin, HIGH);
        digitalWrite(powerPin, HIGH);
        valveOpen = true;

        // Start counting
        unsigned long startTime = millis();
        while (millis() - startTime < duration) {

          // Listen for interruptions
          if (Serial.available() > 0) {
            String interruptCommand = Serial.readStringUntil('\n');
            if (interruptCommand == "CLOSE") {

              // Closing the valve
              digitalWrite(ledPin, LOW);  // Turn the LED off
              digitalWrite(powerPin, LOW);  // Turn the power pin off
              Serial.println("ARDUINO closed the valve");
              valveOpen = false;

            } else if (interruptCommand == "RESET") {

              // Closing the valve and resetting time
              digitalWrite(ledPin, LOW);  // Turn the LED off
              digitalWrite(powerPin, LOW);  // Turn the power pin off
              Serial.println("ARDUINO closed the valve and has reset duration");
              valveOpen = false;
              timeSet = false;
            }

            // Interrupt counting
            break;
          }
        }

        // After counting
        if (valveOpen) {
          digitalWrite(ledPin, LOW);  // Turn the LED off
          digitalWrite(powerPin, LOW);  // Turn the power pin off
          Serial.println("ARDUINO closed the valve");
          valveOpen = false;
        }

      // Error if duration has not been set beforehand
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