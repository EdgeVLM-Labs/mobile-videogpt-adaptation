# WSL2 Webcam Streaming Issue

## Problem Summary

USB webcams passed through to WSL2 via USBIPD **cannot stream video data**, even though the camera device is visible and appears to be working.

## Symptoms

When using **Direct Webcam** mode on WSL2:

```log
INFO  | Attempting to open webcam with index: 0
INFO  | WSL2 detected - using ffmpeg subprocess for camera capture
INFO  | Starting ffmpeg: ffmpeg -f v4l2 -framerate 30 -video_size 640x480 -i /dev/video0 ...
INFO  | ffmpeg capture started for /dev/video0, FPS: 30.00
INFO  | FFmpeg capture loop started: target FPS=30, frame size=921600 bytes
WARNING | Poll #1: No frames extracted (buffer size: 0), skipping
WARNING | Poll #2: No frames extracted (buffer size: 0), skipping
...
WARNING | FFmpeg: incomplete frame read (0/921600 bytes)
ERROR | FFmpeg process died:
INFO  | FFmpeg capture loop ended after 0 frames
```

The camera:

- ✅ Is detected by Linux (`/dev/video0`, `/dev/video4`, etc.)
- ✅ Opens successfully (reports 640x480 @ 30 FPS)
- ❌ Cannot actually stream frame data (0 bytes read)

## Root Cause

### The Technical Issue

When a USB camera is passed through USBIPD to WSL2, it appears under the `vhci_hcd` (Virtual Host Controller Interface) driver:

```bash
$ lsusb -t
/:  Bus 001.Port 001: Dev 001, Class=root_hub, Driver=vhci_hcd/8, 480M
    |__ Port 001: Dev 002, Class=Video, Driver=uvcvideo, 480M
```

The `vhci_hcd` is a **virtual USB controller** that tunnels USB packets between Windows and WSL2 over a network-like interface. While this works for:

- USB storage devices
- Keyboards/mice
- Serial devices

It **fails for USB video** because:

1. Video streaming requires **isochronous transfers** with strict timing guarantees
2. The `vhci_hcd` adds latency that breaks real-time video streaming
3. USB 2.0 video (480 Mbps) saturates the virtual USB bus capacity

### Why the Camera "Opens" But Can't Stream

The UVC (USB Video Class) driver can:

- Query camera capabilities (GET requests work)
- Set format/resolution (control transfers work)

But when streaming starts:

- Isochronous transfers timeout
- `select()` calls block indefinitely
- Zero bytes are read from the video buffer

## Solutions

### ✅ Recommended: Browser Webcam Mode

Use Gradio's browser-based webcam capture. The browser runs on Windows and has direct access to the camera, then streams frames to the WSL2 backend.

**How to use:**

1. Select **"Browser Webcam"** in Input Mode
2. Allow camera access when browser prompts
3. Click "Start Inference"

This completely bypasses WSL2's USB limitations.

### Alternative: OBS Virtual Camera

1. Install OBS Studio on Windows
2. Add your USB webcam as a source
3. Start Virtual Camera (Tools → Start Virtual Camera)
4. The virtual camera may work better through USBIPD (software-based)

### Alternative: Run on Native Linux

If you have a dual-boot setup, boot into Linux natively where the USB camera works directly.

### Alternative: Run on Windows

Run the Python environment directly on Windows where the camera works natively.

## Verification Commands

### Check if camera is detected:

```bash
ls -la /dev/video*
v4l2-ctl --list-devices
```

### Check camera capabilities:

```bash
v4l2-ctl -d /dev/video0 --all
```

### Test if camera can actually stream:

```bash
# This will likely fail on WSL2 with USB passthrough
ffmpeg -f v4l2 -i /dev/video0 -frames:v 1 -y test_frame.jpg

# If it says "Output file is empty, nothing was encoded" - streaming doesn't work
```

### Check USB controller type:

```bash
lsusb -t
# If you see "Driver=vhci_hcd" - you're using USBIPD passthrough
```

## Related Links

- [USBIPD-WIN GitHub](https://github.com/dorssel/usbipd-win)
- [WSL2 USB Limitations Discussion](https://github.com/dorssel/usbipd-win/issues/292)
- [Microsoft WSL2 USB Support](https://learn.microsoft.com/en-us/windows/wsl/connect-usb)

## Conclusion

This is a **fundamental limitation of WSL2's USB passthrough mechanism**, not a bug in the code. The Browser Webcam mode is the recommended workaround for WSL2 users who need live webcam functionality.
