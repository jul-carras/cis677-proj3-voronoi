// Voronoi Generator

#include <iostream>
#include <random>
#include <fstream>
#include <cstring>

using namespace std;

class v_point
{
	public:
	
	long position;
	int red;
	int blue;
	int green;
};

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

// declarations
bool is_a_center(long, v_point[], long);
v_point closest_center(v_point, v_point[], long, long);

int main (int argc, char *argv[])
{	
	v_point* img;
	v_point* v_locations;
	
	v_point* 
	v_point closest;
	random_device dice;
	mt19937 generator(dice());
	
	if (argc != 3) {
		cerr << "usage: dim voronoi_points" << endl;
		exit(-1);
    }
	
	long dim = atol (argv[1]);
	long v_pts = atol(argv[2]);
	uniform_int_distribution<int> distr(0, dim*dim - 1);
	uniform_int_distribution<int> color(0, 255);
	
	img = new v_point[dim*dim];
	v_locations = new v_point[v_pts];
	
	cout << "Image Dim: " << dim << endl;
	cout << "Voronoi Points: " << v_pts << endl;
	// place points using a uniform distribution
	for (long i = 0; i < v_pts; i++){
		v_locations[i].position = distr(generator);
		v_locations[i].red = color(generator);
		v_locations[i].blue = color(generator);
		v_locations[i].green = color(generator);
	}
	
	// calling function to call kernel.
	calc_distance(img, v_locations, dim*dim, v_pts);
	

	// Write out using Wolffe's function
	unsigned char *test = new unsigned char[dim * dim * 3];
	unsigned char rgb[3];
	
	for (int i = 0; i < dim; i++) {
		for (int j = 0; j < dim * 3; j++) {
			rgb[0] = img[i*dim + j].red;
			rgb[1] = img[i*dim + j].blue;
			rgb[2] = img[i*dim + j].green;
			test[(i * 3 * dim) + j] = rgb[j % 3];
		}
	}
	writeImage("test.ppm", dim, test);
	delete test;
	
	return 0;
}

bool is_a_center(long check, v_point center_list[], long size){
	for(long i = 0; i < size; i++){
		if(center_list[i].position == check) return true;
	}
	return false;
}
