# Nano 33 BLE Sense - Sensor Trainer (v006)

One board. No SD card. No second MCU.
A single web page collects data from **every onboard sensor** of the Arduino Nano 33 BLE Sense over **Web Bluetooth or Web Serial (USB)**, trains a small neural network **in the browser**, and downloads a header file. Put that header next to the sketch, upload, and the Nano boots with the finished model baked into its firmware.

```
index-v006.html                     <- the web page (no libraries, works offline)
nano33-ble-trainer-v006/
  nano33-ble-trainer-v006.ino       <- the firmware (Arduino IDE folder name must match the .ino name)
  myNanoModel.h                     <- you add this: downloaded from the page
```

## Quick start

1. **Arduino IDE**: install the *Arduino Mbed OS Nano Boards* core, then the libraries `ArduinoBLE`, `Arduino_LSM9DS1`, `Arduino_APDS9960`, `Arduino_LPS22HB`, `Arduino_HTS221`. Select **Arduino Nano 33 BLE**, open `nano33-ble-trainer/nano33-ble-trainer.ino`, Upload.
2. **Open `index.html`** in Chrome or Edge (from GitHub Pages, or `localhost`; Web Bluetooth/Serial need https or localhost).
3. **0 Connect**: Bluetooth or USB. The 17 live sensor values appear in step 1.
4. **2 Classes**: rename/add/remove classes (2 to 7). Click **Record** and do that action with the Nano. 30+ windows per class works well.
5. **3 Train**: takes a second or two.
6. **4 Test**: *Test in browser* runs the model in the page. *Send model to Nano* then *Test on Nano* shows what the Nano itself decides.
7. **5 Bake**: *Download myNanoModel.h*, save it in the sketch folder next to the `.ino`, press Upload. Done: no editing of the sketch needed, it detects the file.

After baking, power the Nano from any USB supply. With no computer talking to it, it classifies on its own and the **RGB LED shows the class** (the coloured dot next to each class name in the page). Press *Check Nano* in the page to confirm it holds the baked model.

## What the model sees (17 values)

| Sensor | Values |
|---|---|
| LSM9DS1 accelerometer | ax ay az (g) |
| LSM9DS1 gyroscope | gx gy gz (deg/s) |
| LSM9DS1 magnetometer | mx my mz (uT) |
| MP34DT05 microphone | mic level |
| LPS22HB | pressure (kPa) |
| HTS221 | temperature (C), humidity (%) |
| APDS9960 | proximity (0 near .. 255 far), colour R G B |

One sample = 40 steps x 25 ms (1 s) of all 17 channels (slow sensors are re-read every few steps and held in between).
Network: Conv1D(5, 8 filters) -> MaxPool -> Dense 32 -> Dense 16 -> softmax. About 5.9k weights (23 KB).

**Sense effect (0 to 1):** click any chart under "Last captured window" to set how much that sense counts. 0 = ignored, 1 = full. The effect is used in training, in the browser test and on the Nano, and is baked into the header. Change effects, then train again.

**Train from start:** untick it to keep training the last model with the current samples.

v006 changed the window layout (17 x 40). Projects and models saved by v003 or older cannot be loaded. Training happens only in the browser; the Nano only does a forward pass.

## Bluetooth or USB

Both transports do everything: live sensors, capture, send model, test. Bluetooth is wireless; USB is the most reliable (close the Arduino Serial Monitor first, only one program can hold the port). If Bluetooth is crowded at a venue, switch to USB.

USB commands (the page sends these; you can also type them in the Serial Monitor): `CAPTURE`, `INFER`, `STATUS`, `LIVE 1`, `LIVE 0`, `HELP`.

## Notes

- **Your work is autosaved** in the browser (samples, classes, model). Use *Save project (.json)* to move it to another computer.
- **Class names**: letters, numbers, `_` and `-`, max 12 characters (the Nano stores them for its Serial output).
- Changing the *number* of classes after training means you must train again (the page tells you).
- Set `MY_AUTORUN` to `0` near the top of the `.ino` if you do not want the standalone LED mode.
- Sending a model from the page only lasts until reboot. Baking is what makes it permanent.
- If the IDE does not auto-detect `myNanoModel.h`, uncomment `//#define USE_BAKED_WEIGHTS` in the `.ino`.
- If the header was made for a different network the compile stops with a clear `static_assert` message: just download it again.

## Troubleshooting

- *No Bluetooth device in the popup*: the Nano must be advertising as `Nano33-Trainer`; press its reset button. On Linux, older Chrome needs `chrome://flags` > Experimental Web Platform features.
- *USB port not listed*: close the Serial Monitor and other apps. On Linux add yourself to the `dialout` group.
- *Capture times out over Bluetooth*: Disconnect, Connect again, or use USB.
- *Test on Nano says no model*: press *Send model to Nano*, or bake it.

By Jeremy Ellis, webmcu-ai. MIT license. Use at your own risk.
