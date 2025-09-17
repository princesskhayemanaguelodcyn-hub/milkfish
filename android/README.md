## Android app (Jetpack Compose) for Bangus Freshness

- Open the `android` folder in Android Studio.
- Ensure the backend is running. In Android Emulator, use `http://10.0.2.2:8000` as the base URL. On a physical device, replace with your machine's LAN IP (e.g., `http://192.168.1.x:8000`).

### Run & Preview
- Build and run on the emulator.
- The main screen lets you:
  - Enter the server URL
  - Pick an image from the device
  - Preview the selected image
  - Send to the API and view classification + confidence
- UI includes an `@Preview` composable for preview in Android Studio without running the backend.

### Dependencies
- Compose Material3, Coil for image preview, Retrofit + Moshi for networking.