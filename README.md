# pd-perceptron









# Build
> [!NOTE]
To build `perceptron` on Linux, Mac and Windows (using Mingw64):

1. `git clone https://github.com/oviniciuscesar/pd-perceptron/ --recursive`;
2. `cd pd-perceptron`;
3. `git submodule add https://github.com/ampl/gsl   gsl`;
4. `cmake . -B build`;
5. `cmake --build build`;


Binaries for macOS, Windows, and Linux are provided here: https://github.com/oviniciuscesar/pd-perceptron/releases. 
If they are not working in some specific architecture or system, please contact me (they was tested on macOS (Sonoma 14.5), Windows 10 64 bits and Ubuntu 22.04.2 LTS 64 bits runing Pure Data Vanilla 0.54-1).

# License

`perceptron` uses `GSL - GNU Scientific Library`. `GSL - GNU Scientific Library` is licensed under GNU GPL version 3.0. 
