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