CC=C:\mingw64\bin\gcc
cpu1:cpu1.c
	$(CC) cpu1.c -o cpu1.exe 

cpu2:cpu2.c
	$(CC) -mavx cpu2.c -o cpu2.exe

cpu3_4x4:cpu3_4x4.c
	$(CC) -mavx -march=haswell cpu3_4x4.c -o cpu3_4x4.exe

cpu3_8x8:cpu3_8x8.c
	$(CC) -mavx -march=haswell cpu3_8x8.c -o cpu3_8x8.exe

cpu3_16x16:cpu3_16x16.c
	$(CC) -mavx -march=haswell cpu3_16x16.c -o cpu3_16x16.exe

cpu3_32x32:cpu3_32x32.c
	$(CC) -mavx -march=haswell cpu3_32x32.c -o cpu3_32x32.exe

cpu3_64x64:cpu3_64x64.c
	$(CC) -mavx -march=haswell cpu3_64x64.c -o cpu3_64x64.exe