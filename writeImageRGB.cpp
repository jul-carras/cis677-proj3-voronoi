// program that uses a function to write a simple 2D image file
// demonstrates structure of the '.ppm' RGB file format

#include <iostream>
#include <fstream>
#include <cstring>
using namespace std;

void writeImage(const char *fileName, int dim, unsigned char *data)
{
	ofstream f;
	f.open(fileName, fstream::out | fstream::binary);

	f << "P6" << endl;
	f << dim << " " << dim << endl;
	f << "255" << endl;

	for (int x = 0; x < dim; x++) {
		for (int y = 0; y < dim * 3; y++) {
			f << data[(x * 3 * dim) + y];	
		}
	}

	f.close();
}

int main()
{
	int size = 768;
	unsigned char *test = new unsigned char[size * size * 3];
	unsigned char color[3];

	// generates data (a stream of bytes) representing a series of RED gradients
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size * 3; j++) {
			color[0] = j % 256;
			color[1] = 0;
			color[2] = 0;
			test[(i * 3 * size) + j] = color[j % 3];
		}
	}

	// pass data to function for writing to file
	writeImage("test.ppm", size, test);
	delete test;

	return 0;
}

