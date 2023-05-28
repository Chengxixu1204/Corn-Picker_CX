TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    map_drawer.cpp


HEADERS += \
    map_drawer.h


INCLUDEPATH+=/usr/include/eigen3

INCLUDEPATH+=/usr/include/opencv \
             /usr/include   \
              /usr/include/opencv2

LIBS+= -lopencv_core -lopencv_highgui -lopencv_imgproc

