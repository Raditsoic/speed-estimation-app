## **Speed Estimation App w/ YOLOv8**

To start please install the dependencies by:
`pip install -r requirements.txt`

if you got this error
> OpenCV(4.5.1) C:\Users\appveyor\AppData\Local\Temp\1\pip-req-build-1drr4hl0\opencv\modules\highgui\src\window.cpp:651: error: (-2:Unspecified error) The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Cocoa support. If you are on Ubuntu or Debian, install libgtk2.0-dev and pkg-config, then re-run cmake or configure script in function 'cvShowImage',
 
Run:
```pwsh
pip uninstall opencv-python-headless -y 

pip install opencv-python --upgrade
```

## Camera Installation

1. Connect the power cable and ethernet for the camera
2. Download SADP for the camera
3. Grab the IPv4 Address and type it in to your browser
4. You will see the live preview of the camera and the advanced settings to see some port and other setting and information
5. To get the RTSP, use this line for your code 'rtsp://{username}:{password}@{IP}:{port}/stream'
