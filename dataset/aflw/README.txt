################################################################
## README: Annotated Facial Landmarks in the Wild Database (AFLW)
################################################################

The AFLW database consists of three main parts. The images, the annotations and additional tools.
Note that the image data is not property of Graz University of Technology. Graz University of Technology is not responsible for the content nor the meaning of the images. 
Any use of the images other than for not-for-profit research purposes must be negociated with the respective picture owners!
Annotations, code and tools are provided without any warranty.

Directory structure of AFLW:
----------------------------

aflw/data/aflw.sqlite (annoatations - SQLITE database)
aflw/data/flickr (image data)
aflw/doc/ (README, paper)
aflw/facedbsql/ (C++ AFLW database wrapper library and tools)
aflw/gui/ (C++ annotation gui)
aflw/matlab/ (matlab scripts)

################################################################
## 
##	FIRST STEPS
##

(1) Download and install an sqlite management gui such as SQLite2009Pro for windows or Sqliteman under linux and explore the SQLite database (aflw.sqlite) file.
(2) Explore the matlab scripts or the gui binaries
(3) Build facedbsql library and tools
(4) Build the gui

For the gui binaries under Windows install the Microsoft Visual C++ 2010 Redistributable Package (x86).
To build facedbsql and the gui knowledge of CMake (http://www.cmake.org/) is strongly strongly encouraged!
For the dependencies we report also the library or software versions we used.

################################################################
##
##   INSTALL
##

----------------------------------------------------------
* Matlab scripts

Source: aflw/matlab/
Dependencies: mksqlite 1.1 (http://mksqlite.berlios.de/)

Matlab scripts make use of mksqlite. 
Compile and add it to your matlab path.

----------------------------------------------------------
* FaceDBSql Library 

Source: aflw/facedbsql/src/
Dependencies: Boost 1.46 (http://www.boost.org), OpenCV 2.3.2 (opencv.willowgarage.com/)

Install Boost and OpenCV. Set the respective environment variables (BOOST_ROOT, OpenCV_DIR).

- Windows:

In CMake gui set the CMAKE_INSTALL_PATH to your choice or leave it default. 
Check BUILD_SQLITE.
Build and install solution in VS.

- Linux: 

Set CMAKE_INSTALL_PATH (or leave default (/usr/local/)) and 
install my calling

 > sudo make install

----------------------------------------------------------
* PoseAnnotationGUI

Source: aflw/gui/src/
Dependencies: 

Boost 1.46 (http://www.boost.org)
OpenCV 2.3.2 (opencv.willowgarage.com/)
Coin3D 3.1.3 (http://www.coin3d.org)
GLEW 1.5.2 (http://glew.sourceforge.net)
QUARTER 1.0.0 (http://www.coin3d.org/lib/quarter)
QT 4.7 (qt.nokia.com)

- Windows:

Install the libraries. Ensure that the environment variables (COINDIR, GLEWROOT, QUARTERDIR, QTDIR) are properly set.
For facedbsql set the FaceDBSql_DIR environment variable to your install directory
(where there are two subdirectories lib/ and include/facedbsql/ are located).

- Linux: 

If you didn't install the FaceDBSql Library to a default location (/usr/local/)
set the environment variable FaceDBSql_DIR to your install directory
(where there are two subdirectories lib/ and include/facedbsql/)

################################################################
##
##   Known Problems
##

Linux:

----------------------------------------------------------
* "unrecognized option '-Bsymbolic-functions"

On some systems

  > coin-config --ldflags

which is used in FindCoin.cmake, returns

  "-L/usr/lib -Wl,-Bsymbolic-functions"

this may lead to the following linker error:

  /usr/bin/ld: unrecognized option '-Bsymbolic-functions

If you happen to have this problem, remove "-Bsymbolic-functions" from
the COIN_LINK_DIRECTORIES entry in cmake (to be found under "advanced view").
