# pd-perceptron



The `perceptron` is an artificial neuron model, created by Frank Rosenblatt in 1958, capable of automatically “learning” which are the ideal values ​​of weights `𝑊` that, when weighting the inputs `𝑋`, transmit the signal (returning the value 1) or not (returning the value 0) depending on the nature of the input data (input signal). In the context of machine learning, such an algorithm is used to solve binary classification and linear regression.

This implementation is the result of a study of the `creative appropriation` of `machine learning` algorithms in the context of `live-electornics` music `composition`. Therefore, it presents some functions that are not in the original algorithm created by Rosenblat.



# Build
> [!NOTE]
To build `perceptron` on Linux, Mac and Windows (using Mingw64):

1. `git clone https://github.com/oviniciuscesar/pd-perceptron/ --recursive`;
2. `cd pd-perceptron`;
3. `git submodule add https://github.com/ampl/gsl   gsl`;
4. `cmake . -B build`;
5. `cmake --build build`;


Binaries for macOS, Windows, and Linux are provided here: https://github.com/oviniciuscesar/pd-perceptron/releases. 
If they are not working in some specific architecture or system, please contact me. 
`perceptron` was tested on macOS (Sonoma 14.5), Windows 10 64 bits, and Ubuntu 22.04.2 LTS 64 bits running Pure Data Vanilla 0.54-1).

# License

`perceptron` uses `GSL - GNU Scientific Library` licensed under GNU GPL version 3.0. 
