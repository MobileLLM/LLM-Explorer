# LLM-Explorer

## About
This repository contains the code for the system for the paper: LLM-Explorer: Towards Efficient and Affordable LLM-based Exploration for Mobile Apps.

LLM-Explorer is based on [DroidBot](https://github.com/honeynet/droidbot). By leveraging LLM, the exploration can be more efficient.


## How to install

1. `Python`
2. `Java`
3. `Android SDK`
4. Add `platform_tools` directory in Android SDK to `PATH`

Then clone this repo and install with pip:
```
git clone https://github.com/MobileLLM/LLM-Explorer.git
cd LLM-Explorer
pip install -e .
```

## How to use

1. Make sure you have:

    - `.apk` file path of the app you want to analyze.
    - A device or an emulator connected to your host machine via `adb`.

2. Set `GPT_API_URL` and `GPT_API_KEY` environment variables to your ChatGPT API url and key in `start.py`.
2. Start DroidBot with LLM-guided policy:

    ```
    droidbot -a <path_to_apk> -o output_dir -policy llm_guided
    ```

    - If you are using multiple devices, you may need to use `-d <device_serial>` to specify the target device. The easiest way to determine a device's serial number is calling `adb devices`.
    - On some devices, you may need to manually turn on accessibility service for DroidBot (required by DroidBot to get current view hierarchy).
    - If you want to test a large scale of apps, you may want to add `-keep_env` option to avoid re-installing the test environment every time.
    - You may find other useful features in `droidbot -h`.
