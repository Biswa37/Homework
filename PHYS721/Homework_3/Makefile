ROOTLIBS	= $(shell root-config --libs)
INCLUDE		= $(shell root-config --incdir)
ROOTSYS		= $(shell root-config --exec-prefix)
CXX = clang++
CXXFLAGS =      -O3 -fPIC -w -g $(shell root-config --cflags) 
INCS =          -I$(shell root-config --incdir)
LIBS =          $(shell root-config --glibs)
TARGET =		Problem_1
FILENAME=		Problem_1

all:	clean Problem_1

Problem_1:	$(FILENAME).o 
	$(CXX) $(FILENAME).o -L. $(CXXFLAGS) $(LIBS) -o $(TARGET) 

clean:
	-rm -f $(OBJS) $(LIBOUT) $(TARGET) $(FILENAME).o 
