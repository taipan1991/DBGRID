The program has the following parameters to pass.
DBGRID [Dataset filename] [EPS] [MinPTS] [#Objects in dataset] [Mode]
For example, 

DBGRID t48k.txt 8.5 15 8000 3

*Notes:
1. The code was set to run on 2D dataset. If you would like to run on the other dimension please modify the DIM in the code on line 22.
2. The code was written in C++.
3. Since it is an experimental code, the mode should be set to 3 or 4 so that the program can report the reliable output.