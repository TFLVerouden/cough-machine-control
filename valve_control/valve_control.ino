const int ledPin = LED_BUILTIN;  // Use the built-in LED
const int powerPin = 5;  // Use pin 5 for 5.0V control
const int pressurePin = A0;  // Placeholder pin for pressure reading
int duration = 0;  // Variable to store the duration
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
  pinMode(pressurePin, INPUT);
  digitalWrite(ledPin, LOW);
  digitalWrite(powerPin, LOW);
  Serial.begin(9600);
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
  if (command.startsWith("O")) {
    duration = command.substring(2).toInt();
    if (duration > 0) {
      openValve();
      currentState = OPENING;
    } else {
      currentState = ERROR;
    }

  } else if (command == "C") {
    if (valveOpen) {
      currentState = CLOSING;
    }

  } else if (command == "P?") {
    readPressure();

  } else {
    currentState = ERROR;
  }
}

void openValve() {
  digitalWrite(ledPin, HIGH);
  digitalWrite(powerPin, HIGH);
  valveOpen = true;
  startTime = millis();
}

void closeValve() {
  digitalWrite(ledPin, LOW);
  digitalWrite(powerPin, LOW);
  valveOpen = false;
  Serial.println("!");
}

void blinkError() {
  for (int i = 0; i < 5; i++) {
    digitalWrite(ledPin, HIGH);
    delay(100);
    digitalWrite(ledPin, LOW);
    delay(100);
  }
}

void readPressure() {
  // int pressureValue = analogRead(pressurePin);
  // TODO: connect 4-20 mA board to read out sensor
  int pressureValue = 0;
  Serial.println(pressureValue);
}