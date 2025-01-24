const int ledPin = LED_BUILTIN;  // Use the built-in LED
const int powerPin = 5;  // Use pin 5 for 5.0V control
int duration = 0;  // Variable to store the duration
bool timeSet = false;  // Flag to check if time has been set
bool valveOpen = false;  // Flag to check if the valve is open
unsigned long startTime = 0;  // Variable to store the start time

enum State {
  IDLE,
  OPENING,
  CLOSING,
  ERROR
};

State currentState = IDLE;

void setup() {
  pinMode(ledPin, OUTPUT);
  pinMode(powerPin, OUTPUT);
  digitalWrite(ledPin, LOW);
  digitalWrite(powerPin, LOW);
  Serial.begin(9600);
  Serial.println("ARDUINO ready");
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    handleCommand(command);
  }

  switch (currentState) {
    case IDLE:
      // Do nothing
      break;

    case OPENING:
      if (millis() - startTime >= duration) {
        closeValve();
        currentState = IDLE;
      }
      break;

    case CLOSING:
      closeValve();
      currentState = IDLE;
      break;

    case ERROR:
      blinkError();
      currentState = IDLE;
      break;
  }
}

void handleCommand(String command) {
  Serial.print("ARDUINO received command: ");
  Serial.println(command);

  if (command.startsWith("TIME")) {
    duration = command.substring(5).toInt();
    Serial.print("ARDUINO setting opening duration to: ");
    Serial.print(duration);
    Serial.println(" ms");
    timeSet = true;

  } else if (command == "OPEN") {
    if (timeSet) {
      openValve();
      currentState = OPENING;
    } else {
      Serial.println("ARDUINO error: duration not set");
      currentState = ERROR;
    }

  } else if (command == "CLOSE") {
    if (valveOpen) {
      currentState = CLOSING;
    } else {
      Serial.println("ARDUINO warning: valve is already closed");
    }

  } else if (command == "RESET") {
    Serial.println("ARDUINO resetting time");
    timeSet = false;

  } else {
    Serial.println("ARDUINO error: Unknown command");
  }
}

void openValve() {
  Serial.println("ARDUINO opened the valve");
  digitalWrite(ledPin, HIGH);
  digitalWrite(powerPin, HIGH);
  valveOpen = true;
  startTime = millis();
}

void closeValve() {
  Serial.println("ARDUINO closed the valve");
  digitalWrite(ledPin, LOW);
  digitalWrite(powerPin, LOW);
  valveOpen = false;
}

void blinkError() {
  for (int i = 0; i < 5; i++) {
    digitalWrite(ledPin, HIGH);
    delay(100);
    digitalWrite(ledPin, LOW);
    delay(100);
  }
}