// ======================================================
// ARDUINO NANO 33 BLE SENSE - SENSOR FUSION TRAINER - v01
// (nano33-ble-trainer-v004b)
//
// ONE board, no SD card, no second MCU. The web page (index.html)
// does all the collecting and training over Web Bluetooth OR Web Serial
// (USB). When training is done the page downloads "myNanoModel.h".
// Put that file in THIS sketch folder, upload, and the Nano boots with
// the finished model baked into its code. No file needed on the board.
//
// What this sketch does:
//   - Samples every onboard sensor into one fused window of 17 channels x 40 steps:
//       accel(3) gyro(3) mag(3) mic(1) pressure temp humidity prox colorR colorG colorB
//     (slow sensors are re-read every few steps and held in between)
//   - Every channel has an "effect" 0..1 set in the web page (0 = ignore that sense,
//     1 = full). It is baked into myNanoModel.h along with the weights.
//   - Streams all 17 live sensor values (BLE heartbeat or USB "L," lines)
//   - Sends a captured window to the page (for training)
//   - Receives a model from the page (BLE or USB) for quick testing
//   - Runs inference on the Nano and reports the result
//   - If a baked model exists and no computer is talking to it, it runs
//     on its own and shows the class on the RGB LED (MY_AUTORUN)
//
// Training NEVER happens on the Nano. Only a forward pass.
//
// BAKING A MODEL IN (nothing to edit):
//   1. In the web page click "Download myNanoModel.h"
//   2. Save it in this sketch folder next to nano33-ble-trainer.ino
//   3. Upload. The sketch auto-detects the file (see __has_include below).
//   A model pushed later from the page simply replaces the baked one in
//   RAM until the next reboot.
//
// Arduino IDE libraries (Library Manager): ArduinoBLE, Arduino_APDS9960,
// Arduino_LPS22HB, plus PDM (built in). Rev1: Arduino_LSM9DS1 + Arduino_HTS221.
// Rev2: Arduino_BMI270_BMM150 + Arduino_HS300x (set MY_NANO_REV below).
// Board: "Arduino Nano 33 BLE" from the Arduino Mbed OS Nano Boards core.
//
// FILE LAYOUT RULE: every variable is declared at the top of this file and
// every function is pre-declared in PART 3, so the Arduino IDE never hits
// an "undeclared" error no matter what order the functions are in.
//
// By Jeremy Ellis - MIT license - use at your own risk
// https://github.com/hpssjellis  https://github.com/webmcu-ai
// ======================================================

#include <PDM.h>
#include <Arduino_APDS9960.h>
#include <Arduino_LPS22HB.h>
#include <ArduinoBLE.h>

// ======================================================
// BOARD REVISION: change this ONE number.
//   1 = original Nano 33 BLE Sense  (LSM9DS1 IMU, HTS221 temp/humidity)
//   2 = Nano 33 BLE Sense Rev2      (BMI270+BMM150 IMU, HS300x temp/humidity)
// Rev2 needs the libraries Arduino_BMI270_BMM150 and Arduino_HS300x.
// A model trained on one revision is only approximate on the other
// (different sensor chips, magnetometer axes), so train on the board you deploy on.
// ======================================================
#define MY_NANO_REV 1

#if MY_NANO_REV == 2
  #include <Arduino_BMI270_BMM150.h>
  #include <Arduino_HS300x.h>
  #define HTS HS300x               // lets the rest of this sketch use HTS.readTemperature() on both revisions
#else
  #include <Arduino_LSM9DS1.h>
  #include <Arduino_HTS221.h>
#endif

// ======================================================
// PART 1: SETTINGS (defines only, no variables)
// ======================================================
#define MY_BLE_DEVICE_NAME         "Nano33-Trainer"
#define MY_BLE_SERVICE_UUID        "7e400001-b2c3-5d4e-af60-9b3c7d8eaf20"
#define MY_BLE_CONTROL_CHAR_UUID   "7e400002-b2c3-5d4e-af60-9b3c7d8eaf20"  // write:    page -> Nano  (CAPTURE / INFER / STATUS)
#define MY_BLE_HEARTBEAT_CHAR_UUID "7e400003-b2c3-5d4e-af60-9b3c7d8eaf20"  // notify:   Nano -> page  (17 live sensor values)
#define MY_BLE_BINARY_CHAR_UUID    "7e400004-b2c3-5d4e-af60-9b3c7d8eaf20"  // indicate: Nano -> page  (captured window, chunked)
#define MY_BLE_MODEL_CHAR_UUID     "7e400005-b2c3-5d4e-af60-9b3c7d8eaf20"  // write:    page -> Nano  (model, chunked)
#define MY_BLE_RESULT_CHAR_UUID    "7e400006-b2c3-5d4e-af60-9b3c7d8eaf20"  // notify:   Nano -> page  ("R,..." result, "T,..." status, "E,..." error)

#define MY_BLE_CHUNK_SIZE        180    // bytes per binary chunk
#define MY_BLE_CONTROL_MAXLEN     32
#define MY_BLE_LIVE_MAXLEN       128
#define MY_BLE_RESULT_MAXLEN     160

#define MY_AUTORUN                 1    // 1 = run on its own (LED shows class) when no computer is connected, 0 = never
#define MY_HOST_QUIET_MS       15000    // how long without a page command before autorun may start
#define MY_LIVE_INTERVAL_MS      200    // live sensor stream rate (5 Hz)
#define MY_AUTORUN_GAP_MS        200    // pause between autorun inferences

#define IMU_TIMESTEPS         40
#define CHANNELS              17        // ax ay az gx gy gz mx my mz mic pressure temp humidity prox r g b
#define SAMPLE_INTERVAL_MS    25        // 40 Hz -> 40 steps = 1 second
#define LIVE_VALUES           CHANNELS

#define PACKET_FLOATS (IMU_TIMESTEPS * CHANNELS)                  // 680
#define PACKET_BYTES  (PACKET_FLOATS * 4)                         // 2720

// ---- network architecture - MUST match index.html exactly ----
//   Conv1D(k=5, 17 channels -> 8 filters) -> MaxPool/2 -> flatten(144)
//   -> Dense(32) -> Dense(16) -> Output(N classes, softmax)
#define MY_MAX_CLASSES         7        // 7 = number of LED colours (R,G,B,Y,C,M,W)
#define MY_LABEL_LEN          13        // 12 characters + NUL
#define CONV1_KERNEL           5
#define CONV1_FILTERS          8
#define CONV1_OUT_STEPS       (IMU_TIMESTEPS - CONV1_KERNEL + 1)  // 36
#define POOL1_STEPS           (CONV1_OUT_STEPS / 2)               // 18
#define CONV1_FLAT            (POOL1_STEPS * CONV1_FILTERS)       // 144
#define COMBINED_SIZE         CONV1_FLAT                          // 144
#define CONV1_WEIGHTS         (CONV1_KERNEL * CHANNELS * CONV1_FILTERS)   // 680
#define DENSE1_SIZE           32
#define DENSE2_SIZE           16
#define DENSE1_WEIGHTS        (COMBINED_SIZE * DENSE1_SIZE)       // 4608
#define DENSE2_WEIGHTS        (DENSE1_SIZE * DENSE2_SIZE)         // 512
#define OUTPUT_WEIGHTS_MAX    (DENSE2_SIZE * MY_MAX_CLASSES)      // 112

// Model package (page -> Nano), float32 little endian, then label text:
//   [numClasses] conv1_w(680) conv1_b(8) dense1_w(4608) dense1_b(32)
//   dense2_w(512) dense2_b(16) output_w(16*N) output_b(N)
//   calibMean(17) calibStd(17) effect(17)   then   "label0|label1|...|labelN-1"
#define MY_LABEL_REGION_BYTES  (MY_MAX_CLASSES * MY_LABEL_LEN + 4)
#define MY_MODEL_MAX_FLOATS    (1 + CONV1_WEIGHTS + CONV1_FILTERS + DENSE1_WEIGHTS + DENSE1_SIZE + \
                                DENSE2_WEIGHTS + DENSE2_SIZE + OUTPUT_WEIGHTS_MAX + MY_MAX_CLASSES + \
                                3 * CHANNELS)
#define MY_MODEL_MAX_BYTES     (MY_MODEL_MAX_FLOATS * 4 + MY_LABEL_REGION_BYTES)

#define MY_LINE_MAX          200        // longest USB command line
#define MY_SERIAL_HEX_BYTES   64        // model bytes per "H" line from the page

// ======================================================
// BAKED MODEL: auto-detected. Drop myNanoModel.h next to this file.
// ======================================================
#if defined(__has_include)
  #if __has_include("myNanoModel.h")
    #define USE_BAKED_WEIGHTS
  #endif
#endif
//#define USE_BAKED_WEIGHTS     // <- uncomment ONLY if your IDE does not auto-detect the file above

#ifdef USE_BAKED_WEIGHTS
  #include "myNanoModel.h"
  // The header carries a format number. Anything made by an older page has none (or a different one).
  #if !defined(MY_MODEL_FORMAT) || (MY_MODEL_FORMAT != 4)
    #error "myNanoModel.h was made by an older web page. Open index-v004b.html, train again, download a NEW myNanoModel.h and replace the one in this sketch folder (or delete it to run without a baked model)."
  #endif
  // If any of these fail, the header was made for a different network: re-download it from the page.
  static_assert(MY_BAKED_NUM_CLASSES >= 1 && MY_BAKED_NUM_CLASSES <= MY_MAX_CLASSES, "myNanoModel.h: class count must be 1..7");
  static_assert(sizeof(myModel_conv1_w)  / sizeof(float) == CONV1_WEIGHTS,  "myNanoModel.h: conv1_w size mismatch");
  static_assert(sizeof(myModel_dense1_w) / sizeof(float) == DENSE1_WEIGHTS, "myNanoModel.h: dense1_w size mismatch");
  static_assert(sizeof(myModel_dense2_w) / sizeof(float) == DENSE2_WEIGHTS, "myNanoModel.h: dense2_w size mismatch");
  static_assert(sizeof(myModel_output_w) / sizeof(float) == DENSE2_SIZE * MY_BAKED_NUM_CLASSES, "myNanoModel.h: output_w size mismatch");
  static_assert(sizeof(myModel_calibMean) / sizeof(float) == CHANNELS, "myNanoModel.h: calibMean size mismatch");
  static_assert(sizeof(myModel_calibStd)  / sizeof(float) == CHANNELS, "myNanoModel.h: calibStd size mismatch");
  static_assert(sizeof(myModel_effect)    / sizeof(float) == CHANNELS, "myNanoModel.h: effect size mismatch");
#endif

// ======================================================
// PART 2: ALL GLOBAL VARIABLES (declared once, up here)
// ======================================================

// ---- BLE objects ----
BLEService        myFusionService(MY_BLE_SERVICE_UUID);
BLECharacteristic myControlChar(MY_BLE_CONTROL_CHAR_UUID,     BLEWrite, MY_BLE_CONTROL_MAXLEN);
BLECharacteristic myHeartbeatChar(MY_BLE_HEARTBEAT_CHAR_UUID, BLENotify, MY_BLE_LIVE_MAXLEN);
BLECharacteristic myBinaryChar(MY_BLE_BINARY_CHAR_UUID,       BLEIndicate, MY_BLE_CHUNK_SIZE);
BLECharacteristic myModelChar(MY_BLE_MODEL_CHAR_UUID,         BLEWrite | BLEWriteWithoutResponse, MY_BLE_CHUNK_SIZE);
BLECharacteristic myResultChar(MY_BLE_RESULT_CHAR_UUID,       BLENotify, MY_BLE_RESULT_MAXLEN);

// ---- sensor values ----
// order: ax ay az gx gy gz mx my mz mic pressure temp humidity prox r g b
float myLatest[LIVE_VALUES];
short mySampleBuffer[256];
volatile int mySamplesRead = 0;
float myMicLevel = 0;
float myWindow[PACKET_FLOATS];

// ---- model weights ----
float myConv1_w[CONV1_WEIGHTS];
float myConv1_b[CONV1_FILTERS];
float myDense1_w[DENSE1_WEIGHTS];
float myDense1_b[DENSE1_SIZE];
float myDense2_w[DENSE2_WEIGHTS];
float myDense2_b[DENSE2_SIZE];
float myOutput_w[OUTPUT_WEIGHTS_MAX];
float myOutput_b[MY_MAX_CLASSES];
float myCalibMean[CHANNELS];
float myCalibStd[CHANNELS];
float myEffect[CHANNELS];        // per-sense effect 0..1 (0 = ignore this sense)
int   myNumClasses = 0;
bool  myModelLoaded = false;
bool  myModelIsBaked = false;
char  myLabels[MY_MAX_CLASSES][MY_LABEL_LEN];
int   myLastPred = -1;

// ---- forward pass scratch ----
float myConv1_output[CONV1_OUT_STEPS * CONV1_FILTERS];
float myPool1_output[CONV1_FLAT];
float myCombined[COMBINED_SIZE];
float myDense1_output[DENSE1_SIZE];
float myDense2_output[DENSE2_SIZE];
float myFinal_output[MY_MAX_CLASSES];

// ---- model receive (BLE chunks or USB hex lines) ----
uint8_t myModelRxBuf[MY_MODEL_MAX_BYTES] __attribute__((aligned(4)));
volatile bool          myModelRxActive = false;
volatile uint32_t      myModelRxTotal = 0;
volatile uint32_t      myModelRxGot = 0;
volatile unsigned long myModelRxLastByteMs = 0;

// ---- requests and state ----
volatile bool myPendingCapture = false;
volatile bool myPendingInfer   = false;
volatile bool myPendingStatus  = false;
bool myReplySerial    = false;     // true = answer the page over USB, false = over BLE
bool myBusy           = false;     // true while sampling a window (pauses live stream)
bool myCentralConnected = false;
bool myLiveSerial     = false;     // page asked for the USB live stream
bool myHostEverSeen   = false;     // has any page/computer talked to us yet
unsigned long myHostSeenMs  = 0;
unsigned long myLastLiveMs  = 0;
unsigned long myLastAutoMs  = 0;
char myLineBuf[MY_LINE_MAX];
int  myLineLen = 0;

// ======================================================
// PART 3: FUNCTION PRE-DECLARATIONS (keep in sync with PART 5)
// ======================================================
void  onPDMdata();
void  myUpdateMic();
void  myUpdateFast();
void  myUpdateSlow(bool env, bool optical);
void  mySampleWindow();
int   myBuildLiveText(char* out, int cap);
void  mySendText(const char* text);
void  mySendWindow();
void  myBLESendBuffer(const uint8_t* raw, uint32_t totalBytes);
void  mySendStatus();
void  myActionCapture();
void  myActionInfer();
void  myReportResult(int pred);
float myLeakyRelu(float x);
float myClip(float v, float mn, float mx);
void  mySoftmax(float* x, int size);
void  myNormalizeWindow(float* buf);
void  myForwardPass(float* normalizedWindow);
int   myRunInference(const float* rawWindow);
void  mySetLed(int cls);
void  myLoadBakedModel();
void  myModelRxBegin(uint32_t total);
void  myModelRxAppend(const uint8_t* data, int len);
void  myUnpackModel();
void  myOnModelWrite(BLEDevice central, BLECharacteristic characteristic);
void  myOnControlWrite(BLEDevice central, BLECharacteristic characteristic);
void  myCheckModelRxWatchdog();
int   myHexVal(char c);
void  myHandleHexChunk(const char* hex);
void  myHandleLine(char* line);
void  myReadSerial();
void  myPrintHelp();
void  myLiveTick();
void  myAutoRunTick();
void  myCheckBleConnection();
void  myRunPending();

// ======================================================
// PART 4: SETUP AND LOOP
// ======================================================
void setup() {
  Serial.begin(115200);
  unsigned long t0 = millis();
  while (!Serial && millis() - t0 < 1500) { /* wait briefly, never hang */ }

  Serial.println("\n=== Nano 33 BLE Sense - Sensor Fusion Trainer v004b ===");

  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);
  pinMode(LEDB, OUTPUT);
  mySetLed(-1);

  for (int i = 0; i < LIVE_VALUES; i++) myLatest[i] = 0;
  myLatest[2] = 1.0f;   // az starts at 1 g
  for (int i = 0; i < CHANNELS; i++) myEffect[i] = 1.0f;

  PDM.onReceive(onPDMdata);
  if (!PDM.begin(1, 16000)) Serial.println("ERROR: PDM microphone failed");
  if (!IMU.begin())   Serial.println("ERROR: IMU failed");           else Serial.println("IMU OK");
  if (!APDS.begin())  Serial.println("ERROR: APDS9960 failed");      else Serial.println("APDS9960 OK");
  if (!BARO.begin())  Serial.println("ERROR: LPS22HB failed");       else Serial.println("LPS22HB OK");
  if (!HTS.begin())   Serial.println("ERROR: temp/humidity sensor failed"); else Serial.println("Temp/humidity OK");

  if (!BLE.begin()) {
    Serial.println("FATAL: BLE failed to start");
    while (1) { delay(1000); }
  }
  BLE.setLocalName(MY_BLE_DEVICE_NAME);
  BLE.setDeviceName(MY_BLE_DEVICE_NAME);
  BLE.setAdvertisedService(myFusionService);
  myFusionService.addCharacteristic(myControlChar);
  myFusionService.addCharacteristic(myHeartbeatChar);
  myFusionService.addCharacteristic(myBinaryChar);
  myFusionService.addCharacteristic(myModelChar);
  myFusionService.addCharacteristic(myResultChar);
  BLE.addService(myFusionService);
  myControlChar.setEventHandler(BLEWritten, myOnControlWrite);
  myModelChar.setEventHandler(BLEWritten, myOnModelWrite);
  BLE.advertise();
  Serial.print("BLE advertising as \"");
  Serial.print(MY_BLE_DEVICE_NAME);
  Serial.println("\"");

  #ifdef USE_BAKED_WEIGHTS
    myLoadBakedModel();
  #else
    Serial.println("No myNanoModel.h in the sketch folder - waiting for a model from the web page.");
  #endif

  myPrintHelp();
}

void loop() {
  BLE.poll();
  myCheckModelRxWatchdog();
  myReadSerial();
  myCheckBleConnection();
  myRunPending();
  myLiveTick();
  myAutoRunTick();
}

// ======================================================
// PART 5: FUNCTIONS
// ======================================================

// ---------- sensors ----------
void onPDMdata() {
  int bytesAvailable = PDM.available();
  PDM.read(mySampleBuffer, bytesAvailable);
  mySamplesRead = bytesAvailable / 2;
}

void myUpdateMic() {
  if (mySamplesRead > 0) {
    long total = 0;
    for (int i = 0; i < mySamplesRead; i++) total += abs(mySampleBuffer[i]);
    myMicLevel = (float)total / (float)mySamplesRead;
    mySamplesRead = 0;
  }
}

// fast sensors: accel, gyro, mag, mic -> myLatest[0..9]
void myUpdateFast() {
  float x, y, z;
  if (IMU.accelerationAvailable()) { IMU.readAcceleration(x, y, z);  myLatest[0] = x; myLatest[1] = y; myLatest[2] = z; }
  if (IMU.gyroscopeAvailable())    { IMU.readGyroscope(x, y, z);     myLatest[3] = x; myLatest[4] = y; myLatest[5] = z; }
  if (IMU.magneticFieldAvailable()){ IMU.readMagneticField(x, y, z); myLatest[6] = x; myLatest[7] = y; myLatest[8] = z; }
  myUpdateMic();
  myLatest[9] = myMicLevel;
}

// slow sensors -> myLatest[10..16]. Pressure is read every call; temperature/humidity
// (env) and proximity/colour (optical) only when asked, otherwise the last value is held.
// APDS9960 only gives fresh data every second call, so its last value is kept too.
void myUpdateSlow(bool env, bool optical) {
  myLatest[10] = BARO.readPressure();          // kPa
  if (env) {
    myLatest[11] = HTS.readTemperature();      // C
    myLatest[12] = HTS.readHumidity();         // %
  }
  if (optical) {
    if (APDS.proximityAvailable()) {
      int p = APDS.readProximity();            // 0 = near ... 255 = far
      if (p >= 0) myLatest[13] = (float)p;
    }
    if (APDS.colorAvailable()) {
      int r, g, b;
      if (APDS.readColor(r, g, b) && r >= 0) { myLatest[14] = (float)r; myLatest[15] = (float)g; myLatest[16] = (float)b; }
    }
  }
}

// one fused window: 40 steps x 25 ms = 1 second, all 17 channels every step
void mySampleWindow() {
  myBusy = true;
  for (int t = 0; t < IMU_TIMESTEPS; t++) {
    unsigned long tStart = millis();
    BLE.poll();
    myUpdateSlow(t % 10 == 0, t % 5 == 0);
    myUpdateFast();
    memcpy(myWindow + t * CHANNELS, myLatest, CHANNELS * sizeof(float));
    long elapsed = (long)(millis() - tStart);
    if (elapsed < SAMPLE_INTERVAL_MS) delay(SAMPLE_INTERVAL_MS - elapsed);
  }
  myBusy = false;
}

// ---------- sending to the page ----------
// "v0,v1,...,v16" - same text used for the BLE heartbeat and the USB "L," lines
int myBuildLiveText(char* out, int cap) {
  static const char* fmts[LIVE_VALUES] = { "%.3f","%.3f","%.3f", "%.1f","%.1f","%.1f", "%.1f","%.1f","%.1f",
                                           "%.0f", "%.2f","%.1f","%.1f", "%.0f","%.0f","%.0f","%.0f" };
  int n = 0;
  for (int i = 0; i < LIVE_VALUES && n < cap - 12; i++) {
    if (i > 0) out[n++] = ',';
    n += snprintf(out + n, cap - n, fmts[i], myLatest[i]);
  }
  out[n] = 0;
  return n;
}

// short text reply: USB line if the request came over USB, else BLE result notify
void mySendText(const char* text) {
  if (myReplySerial) {
    Serial.println(text);
  } else if (myCentralConnected) {
    myResultChar.writeValue((const uint8_t*)text, strlen(text));
  }
}

void mySendWindow() {
  if (myReplySerial) {
    Serial.print("W");
    for (int i = 0; i < PACKET_FLOATS; i++) { Serial.print(","); Serial.print(myWindow[i], 4); }
    Serial.println();
  } else {
    myBLESendBuffer((const uint8_t*)myWindow, PACKET_BYTES);
  }
}

// BLE: 4-byte little-endian length, then raw chunks
void myBLESendBuffer(const uint8_t* raw, uint32_t totalBytes) {
  uint8_t header[4];
  header[0] = (uint8_t)(totalBytes & 0xFF);
  header[1] = (uint8_t)((totalBytes >> 8) & 0xFF);
  header[2] = (uint8_t)((totalBytes >> 16) & 0xFF);
  header[3] = (uint8_t)((totalBytes >> 24) & 0xFF);
  myBinaryChar.writeValue(header, 4);
  BLE.poll();
  for (uint32_t offset = 0; offset < totalBytes; offset += MY_BLE_CHUNK_SIZE) {
    int n = min((uint32_t)MY_BLE_CHUNK_SIZE, totalBytes - offset);
    myBinaryChar.writeValue(raw + offset, n);
    BLE.poll();
  }
}

// "T,modelLoaded,baked,numClasses,label0|label1|..."
void mySendStatus() {
  char out[MY_BLE_RESULT_MAXLEN];
  int n = snprintf(out, sizeof(out), "T,%d,%d,%d,", myModelLoaded ? 1 : 0, myModelIsBaked ? 1 : 0, myNumClasses);
  for (int i = 0; i < myNumClasses && n < (int)sizeof(out) - MY_LABEL_LEN - 2; i++) {
    n += snprintf(out + n, sizeof(out) - n, "%s%s", (i > 0 ? "|" : ""), myLabels[i]);
  }
  mySendText(out);
}

// ---------- actions ----------
void myActionCapture() {
  Serial.println(">>> capture window");
  mySampleWindow();
  mySendWindow();
}

void myActionInfer() {
  if (!myModelLoaded) {
    Serial.println("Infer requested but no model is loaded yet.");
    mySendText("E,no model loaded");
    return;
  }
  mySampleWindow();
  int pred = myRunInference(myWindow);
  myReportResult(pred);
}

// "R,pred,conf,p0,p1,...,pN-1" for the page + a readable line + LED colour
void myReportResult(int pred) {
  char out[MY_BLE_RESULT_MAXLEN];
  int n = snprintf(out, sizeof(out), "R,%d,%.3f", pred, myFinal_output[pred]);
  for (int j = 0; j < myNumClasses && n < (int)sizeof(out) - 9; j++) {
    n += snprintf(out + n, sizeof(out) - n, ",%.3f", myFinal_output[j]);
  }
  mySendText(out);

  Serial.print("On-device result: ");
  Serial.print(myLabels[pred]);
  Serial.print(" (");
  for (int j = 0; j < myNumClasses; j++) {
    Serial.print(myLabels[j]);
    Serial.print("=");
    Serial.print(myFinal_output[j] * 100.0f, 0);
    Serial.print(j < myNumClasses - 1 ? "%, " : "%)\n");
  }
  mySetLed(pred);
}

// ---------- inference (forward pass only) ----------
float myLeakyRelu(float x) { return x > 0 ? x : 0.1f * x; }

float myClip(float v, float mn, float mx) {
  if (isnan(v) || isinf(v)) return 0;
  return v < mn ? mn : (v > mx ? mx : v);
}

void mySoftmax(float* x, int size) {
  float maxVal = x[0];
  for (int i = 1; i < size; i++) if (x[i] > maxVal) maxVal = x[i];
  float sum = 0;
  for (int i = 0; i < size; i++) { x[i] = exp(x[i] - maxVal); sum += x[i]; }
  for (int i = 0; i < size; i++) x[i] /= sum;
}

// buf = PACKET_FLOATS long, normalized in place: standardize, clip, then scale by the sense's effect
void myNormalizeWindow(float* buf) {
  for (int t = 0; t < IMU_TIMESTEPS; t++) {
    for (int c = 0; c < CHANNELS; c++) {
      int idx = t * CHANNELS + c;
      buf[idx] = myClip((buf[idx] - myCalibMean[c]) / (myCalibStd[c] + 1e-8f), -5.0f, 5.0f) * myEffect[c];
    }
  }
}

void myForwardPass(float* normalizedWindow) {
  for (int s = 0; s < CONV1_OUT_STEPS; s++) {            // Conv1D
    for (int f = 0; f < CONV1_FILTERS; f++) {
      float sum = myConv1_b[f];
      for (int k = 0; k < CONV1_KERNEL; k++) {
        for (int c = 0; c < CHANNELS; c++) {
          sum += normalizedWindow[(s + k) * CHANNELS + c] *
                 myConv1_w[(k * CHANNELS + c) * CONV1_FILTERS + f];
        }
      }
      myConv1_output[s * CONV1_FILTERS + f] = myLeakyRelu(sum);
    }
  }
  for (int s = 0; s < POOL1_STEPS; s++) {                // MaxPool /2
    for (int f = 0; f < CONV1_FILTERS; f++) {
      float a = myConv1_output[(s * 2)     * CONV1_FILTERS + f];
      float b = myConv1_output[(s * 2 + 1) * CONV1_FILTERS + f];
      myPool1_output[s * CONV1_FILTERS + f] = a > b ? a : b;
    }
  }
  memcpy(myCombined, myPool1_output, CONV1_FLAT * sizeof(float));                        // flatten
  for (int j = 0; j < DENSE1_SIZE; j++) {                // Dense1
    float sum = myDense1_b[j];
    for (int i = 0; i < COMBINED_SIZE; i++) sum += myCombined[i] * myDense1_w[i * DENSE1_SIZE + j];
    myDense1_output[j] = myLeakyRelu(sum);
  }
  for (int j = 0; j < DENSE2_SIZE; j++) {                // Dense2
    float sum = myDense2_b[j];
    for (int i = 0; i < DENSE1_SIZE; i++) sum += myDense1_output[i] * myDense2_w[i * DENSE2_SIZE + j];
    myDense2_output[j] = myLeakyRelu(sum);
  }
  for (int j = 0; j < myNumClasses; j++) {               // Output
    float sum = myOutput_b[j];
    for (int i = 0; i < DENSE2_SIZE; i++) sum += myDense2_output[i] * myOutput_w[i * myNumClasses + j];
    myFinal_output[j] = sum;
  }
  mySoftmax(myFinal_output, myNumClasses);
}

// returns the predicted class index, probabilities stay in myFinal_output
int myRunInference(const float* rawWindow) {
  float buf[PACKET_FLOATS];
  memcpy(buf, rawWindow, PACKET_BYTES);
  myNormalizeWindow(buf);
  myForwardPass(buf);
  int pred = 0;
  for (int j = 1; j < myNumClasses; j++) if (myFinal_output[j] > myFinal_output[pred]) pred = j;
  myLastPred = pred;
  return pred;
}

// RGB LED shows the class: 0 red, 1 green, 2 blue, 3 yellow, 4 cyan, 5 magenta, 6 white, -1 off
// (the Nano's RGB LED is active LOW)
void mySetLed(int cls) {
  static const uint8_t rgb[MY_MAX_CLASSES][3] = { {1,0,0}, {0,1,0}, {0,0,1}, {1,1,0}, {0,1,1}, {1,0,1}, {1,1,1} };
  bool r = false, g = false, b = false;
  if (cls >= 0 && cls < MY_MAX_CLASSES) { r = rgb[cls][0]; g = rgb[cls][1]; b = rgb[cls][2]; }
  digitalWrite(LEDR, r ? LOW : HIGH);
  digitalWrite(LEDG, g ? LOW : HIGH);
  digitalWrite(LEDB, b ? LOW : HIGH);
}

// ---------- model loading ----------
void myLoadBakedModel() {
  #ifdef USE_BAKED_WEIGHTS
    memcpy(myConv1_w,  myModel_conv1_w,  CONV1_WEIGHTS  * sizeof(float));
    memcpy(myConv1_b,  myModel_conv1_b,  CONV1_FILTERS  * sizeof(float));
    memcpy(myDense1_w, myModel_dense1_w, DENSE1_WEIGHTS * sizeof(float));
    memcpy(myDense1_b, myModel_dense1_b, DENSE1_SIZE    * sizeof(float));
    memcpy(myDense2_w, myModel_dense2_w, DENSE2_WEIGHTS * sizeof(float));
    memcpy(myDense2_b, myModel_dense2_b, DENSE2_SIZE    * sizeof(float));
    memcpy(myOutput_w, myModel_output_w, DENSE2_SIZE * MY_BAKED_NUM_CLASSES * sizeof(float));
    memcpy(myOutput_b, myModel_output_b, MY_BAKED_NUM_CLASSES * sizeof(float));
    memcpy(myCalibMean, myModel_calibMean, CHANNELS * sizeof(float));
    memcpy(myCalibStd,  myModel_calibStd,  CHANNELS * sizeof(float));
    memcpy(myEffect,    myModel_effect,    CHANNELS * sizeof(float));
    myNumClasses = MY_BAKED_NUM_CLASSES;
    for (int i = 0; i < myNumClasses; i++) {
      strncpy(myLabels[i], myModel_labels[i], MY_LABEL_LEN - 1);
      myLabels[i][MY_LABEL_LEN - 1] = 0;
    }
    myModelLoaded = true;
    myModelIsBaked = true;
    Serial.print("Baked model loaded: ");
    Serial.print(myNumClasses);
    Serial.println(" classes.");
  #endif
}

void myModelRxBegin(uint32_t total) {
  if (total < 4 || total > MY_MODEL_MAX_BYTES) {
    Serial.print("ERROR: model size not accepted: ");
    Serial.println(total);
    mySendText("E,model too large");
    return;
  }
  myModelRxTotal = total;
  myModelRxGot = 0;
  myModelRxActive = true;
  myModelRxLastByteMs = millis();
  Serial.print("Model receive starting, ");
  Serial.print(total);
  Serial.println(" bytes");
}

void myModelRxAppend(const uint8_t* data, int len) {
  if (!myModelRxActive) return;
  uint32_t room = myModelRxTotal - myModelRxGot;
  uint32_t n = min((uint32_t)len, room);
  memcpy(myModelRxBuf + myModelRxGot, data, n);
  myModelRxGot += n;
  myModelRxLastByteMs = millis();
  if (myModelRxGot >= myModelRxTotal) {
    myModelRxActive = false;
    Serial.println("Model received - unpacking...");
    myUnpackModel();
  }
}

void myUnpackModel() {
  float* f = (float*)myModelRxBuf;
  int idx = 0;
  int n = (int)(f[idx++] + 0.5f);
  if (n < 1 || n > MY_MAX_CLASSES) {
    Serial.print("ERROR: model class count out of range: ");
    Serial.println(n);
    mySendText("E,bad model");
    return;
  }
  uint32_t floatCount = 1 + CONV1_WEIGHTS + CONV1_FILTERS + DENSE1_WEIGHTS + DENSE1_SIZE +
                        DENSE2_WEIGHTS + DENSE2_SIZE + DENSE2_SIZE * n + n + 3 * CHANNELS;
  if (myModelRxTotal < floatCount * 4) {
    Serial.println("ERROR: model package shorter than expected");
    mySendText("E,bad model");
    return;
  }
  memcpy(myConv1_w,  f + idx, CONV1_WEIGHTS  * 4); idx += CONV1_WEIGHTS;
  memcpy(myConv1_b,  f + idx, CONV1_FILTERS  * 4); idx += CONV1_FILTERS;
  memcpy(myDense1_w, f + idx, DENSE1_WEIGHTS * 4); idx += DENSE1_WEIGHTS;
  memcpy(myDense1_b, f + idx, DENSE1_SIZE    * 4); idx += DENSE1_SIZE;
  memcpy(myDense2_w, f + idx, DENSE2_WEIGHTS * 4); idx += DENSE2_WEIGHTS;
  memcpy(myDense2_b, f + idx, DENSE2_SIZE    * 4); idx += DENSE2_SIZE;
  memcpy(myOutput_w, f + idx, DENSE2_SIZE * n * 4); idx += DENSE2_SIZE * n;
  memcpy(myOutput_b, f + idx, n * 4);               idx += n;
  memcpy(myCalibMean, f + idx, CHANNELS * 4);  idx += CHANNELS;
  memcpy(myCalibStd,  f + idx, CHANNELS * 4);  idx += CHANNELS;
  memcpy(myEffect,    f + idx, CHANNELS * 4);  idx += CHANNELS;
  myNumClasses = n;

  // labels: "a|b|c" after the floats
  for (int i = 0; i < MY_MAX_CLASSES; i++) snprintf(myLabels[i], MY_LABEL_LEN, "class%d", i);
  int labelBytes = (int)(myModelRxTotal - floatCount * 4);
  if (labelBytes > 0 && labelBytes <= MY_LABEL_REGION_BYTES) {
    const char* p = (const char*)(myModelRxBuf + floatCount * 4);
    int cls = 0, pos = 0;
    for (int i = 0; i < labelBytes && cls < n; i++) {
      char c = p[i];
      if (c == '|' || c == 0) { myLabels[cls][pos] = 0; cls++; pos = 0; if (c == 0) break; }
      else if (pos < MY_LABEL_LEN - 1) myLabels[cls][pos++] = c;
    }
    if (cls < n) myLabels[cls][pos] = 0;
  }
  myModelLoaded = true;
  myModelIsBaked = false;
  Serial.print("Model loaded from page: ");
  Serial.print(myNumClasses);
  Serial.println(" classes (RAM only - baking makes it permanent).");
  myPendingStatus = true;   // tell the page right away: the new model is already live, no reboot needed
}

// BLE model write: first packet = 4-byte length, then chunks
void myOnModelWrite(BLEDevice central, BLECharacteristic characteristic) {
  int len = myModelChar.valueLength();
  const uint8_t* data = myModelChar.value();
  if (!myModelRxActive) {
    if (len == 4) {
      uint32_t total = (uint32_t)data[0] | ((uint32_t)data[1] << 8) | ((uint32_t)data[2] << 16) | ((uint32_t)data[3] << 24);
      myReplySerial = false;
      myModelRxBegin(total);
    }
    return;
  }
  myModelRxAppend(data, len);
}

// if a push stalls for 5 s, free the receiver so a new push can start
void myCheckModelRxWatchdog() {
  if (myModelRxActive && (millis() - myModelRxLastByteMs > 5000)) {
    Serial.println("WARNING: model transfer stalled - aborted, ready for a new push");
    myModelRxActive = false;
  }
}

// ---------- BLE control ----------
void myOnControlWrite(BLEDevice central, BLECharacteristic characteristic) {
  int len = myControlChar.valueLength();
  char buf[MY_BLE_CONTROL_MAXLEN + 1];
  int n = min(len, MY_BLE_CONTROL_MAXLEN);
  memcpy(buf, myControlChar.value(), n);
  buf[n] = 0;
  for (char* p = buf; *p; p++) *p = toupper(*p);
  myReplySerial = false;
  if      (strncmp(buf, "CAPTURE", 7) == 0) myPendingCapture = true;
  else if (strncmp(buf, "INFER", 5) == 0)   myPendingInfer = true;
  else if (strncmp(buf, "STATUS", 6) == 0)  myPendingStatus = true;
  else if (strncmp(buf, "LED ", 4) == 0)    mySetLed(atoi(buf + 4));
}

void myCheckBleConnection() {
  bool now = BLE.connected();
  if (now && !myCentralConnected) {
    myCentralConnected = true;
    myHostEverSeen = true;
    myHostSeenMs = millis();
    Serial.println("BLE page connected");
  } else if (!now && myCentralConnected) {
    myCentralConnected = false;
    myHostSeenMs = millis();
    Serial.println("BLE page disconnected - advertising again");
    BLE.advertise();
  }
  if (myCentralConnected) myHostSeenMs = millis();
}

// ---------- USB serial commands (page or Serial Monitor) ----------
int myHexVal(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return 0;
}

// "H" + hex text from the page: model bytes over USB, answered with "A,bytesSoFar"
void myHandleHexChunk(const char* hex) {
  if (!myModelRxActive) { Serial.println("E,no model transfer in progress"); return; }
  uint8_t tmp[MY_SERIAL_HEX_BYTES + 8];
  int n = 0;
  while (hex[0] && hex[1] && n < (int)sizeof(tmp)) {
    tmp[n++] = (uint8_t)((myHexVal(hex[0]) << 4) | myHexVal(hex[1]));
    hex += 2;
  }
  myModelRxAppend(tmp, n);
  Serial.print("A,");
  Serial.println(myModelRxGot);
}

void myHandleLine(char* line) {
  myHostSeenMs = millis();
  myHostEverSeen = true;
  if (line[0] == 'H') { myHandleHexChunk(line + 1); return; }
  for (char* p = line; *p; p++) *p = toupper(*p);
  myReplySerial = true;
  if      (strcmp(line, "CAPTURE") == 0)       myPendingCapture = true;
  else if (strcmp(line, "INFER") == 0)         myPendingInfer = true;
  else if (strcmp(line, "STATUS") == 0)        myPendingStatus = true;
  else if (strcmp(line, "LIVE 1") == 0)        myLiveSerial = true;
  else if (strcmp(line, "LIVE 0") == 0)        myLiveSerial = false;
  else if (strncmp(line, "LED ", 4) == 0)      mySetLed(atoi(line + 4));   // page shows a class colour (-1 = off)
  else if (strncmp(line, "MODEL ", 6) == 0) {
    myModelRxBegin((uint32_t)strtoul(line + 6, NULL, 10));
    Serial.println("A,0");
  }
  else if (strcmp(line, "HELP") == 0 || strcmp(line, "?") == 0) myPrintHelp();
}

void myReadSerial() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      myLineBuf[myLineLen] = 0;
      if (myLineLen > 0) myHandleLine(myLineBuf);
      myLineLen = 0;
    } else if (myLineLen < MY_LINE_MAX - 1) {
      myLineBuf[myLineLen++] = c;
    } else {
      myLineLen = 0;   // line too long: drop it
    }
  }
}

void myPrintHelp() {
  Serial.println("Commands (end with Enter): CAPTURE, INFER, STATUS, LIVE 1, LIVE 0, LED n, HELP");
  Serial.println("The web page (index.html) sends these for you.");
}

// ---------- loop helpers ----------
void myRunPending() {
  if (myPendingStatus)  { myPendingStatus = false;  mySendStatus(); }
  if (myPendingCapture) { myPendingCapture = false; myActionCapture(); }
  if (myPendingInfer)   { myPendingInfer = false;   myActionInfer(); }
}

// live sensor stream: BLE heartbeat when connected, "L," lines when the page asked over USB
void myLiveTick() {
  if (myBusy || myModelRxActive) return;
  unsigned long now = millis();
  if (now - myLastLiveMs < MY_LIVE_INTERVAL_MS) return;
  bool serialOn = myLiveSerial && myHostEverSeen && (now - myHostSeenMs < MY_HOST_QUIET_MS);
  if (!myCentralConnected && !serialOn) return;
  myLastLiveMs = now;
  myUpdateFast();
  myUpdateSlow(true, true);
  char text[MY_BLE_LIVE_MAXLEN];
  int n = myBuildLiveText(text, sizeof(text));
  if (myCentralConnected) myHeartbeatChar.writeValue((const uint8_t*)text, n);
  if (serialOn) { Serial.print("L,"); Serial.println(text); }
}

// standalone mode: no page talking to us -> keep classifying and show the class on the LED
void myAutoRunTick() {
  #if MY_AUTORUN
    if (!myModelLoaded || myCentralConnected || myModelRxActive) return;
    unsigned long now = millis();
    if (myHostEverSeen && (now - myHostSeenMs < MY_HOST_QUIET_MS)) return;
    if (now - myLastAutoMs < MY_AUTORUN_GAP_MS) return;
    myReplySerial = false;
    myActionInfer();
    myLastAutoMs = millis();
  #endif
}
