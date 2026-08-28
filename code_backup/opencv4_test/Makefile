# Compiler settings
CXX = g++
CXXFLAGS = -Wall -O2 `pkg-config --cflags opencv4`
LIBS = `pkg-config --libs opencv4`

# Target executable name
TARGET = opencv_test

# Source files
SRCS = main.cpp

# Default rule
all: $(TARGET)

# Compile target
$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET) $(LIBS)

# Clean build artifacts
clean:
	rm -f $(TARGET) test_output.jpg

.PHONY: all clean

