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
	
	// calculate distances
	for (int i = 0; i < dim*dim; i++){
		img[i].position = i;
		// check if pixel is a center
		if (!is_a_center(i, v_locations, v_pts)){
			closest = closest_center(img[i], v_locations, v_pts, dim);
			img[i].red = closest.red;
			img[i].blue = closest.blue;
			img[i].green = closest.green;			
		} else {
			img[i].position = 0;
		}
		
	}

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

v_point closest_center(v_point point, v_point center_list[], long center_list_size, long dim){
	float distance;
	float shortest_dist = 0.0;
	long x_point, y_point;
	long x_center, y_center;
	v_point closest_center;

	// extract x and y components for the checked point
	x_point = point.position / dim;
	y_point = point.position % dim;
	

	// run distances against all listed centers
	for(long i = 0; i < center_list_size; i++){
		// extract x and y components for the center
		x_center = center_list[i].position / dim;
		y_center = center_list[i].position % dim;	
		distance = sqrt(pow(1.0 * x_center - x_point, 2) + pow(1.0 * y_center - y_point, 2));
		// debug
		//cout << "(" << x_point << "," << y_point << ")" << endl;
		//cout << "(" << x_center << "," << y_center << ")" << endl;	
		//cout << "Distance: " << distance << endl;	
		
		// are we in the first iteration? Take that distance
		if(i == 0){	
			shortest_dist = distance;
			closest_center.position = center_list[i].position;
			closest_center.red = center_list[i].red;
			closest_center.blue = center_list[i].blue;
			closest_center.green = center_list[i].green;
			
		// if not, then check to see if the new distance we calculated is smaller
		// note this produces a first calculated point for contested areas
		} else if(distance < shortest_dist){
			shortest_dist = distance;
			closest_center.position = center_list[i].position;
			closest_center.red = center_list[i].red;
			closest_center.blue = center_list[i].blue;
			closest_center.green = center_list[i].green;
		} 
	}
	return closest_center;
}

// Write image - from Dr Wolffe


