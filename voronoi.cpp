#include <iostream>
#include <random>

using namespace std;

int main (int argc, char *argv[])
{	
	int* img;
	int* v_locations;
	int v_local;
	random_device dice;
	mt19937 generator(dice());
	
	if (argc != 3) {
		cerr << "usage: dim voronoi_points" << endl;
		exit(-1);
    }
	
	long dim = atol (argv[1]);
	long v_pts = atol(argv[2]);
	uniform_int_distribution<int> distr(0, dim*dim);
	
	img = new int[dim*dim];
	v_locations = new int[v_pts];
	
	cout << "Image Dim: " << dim << endl;
	cout << "Voronoi Points: " << v_pts << endl;
	// place points using a uniform distribution
	for (int i = 0; i < v_pts; i++){
		v_local = distr(generator);
		
		while(img[v_local] == -1){
			v_local = distr(generator);
		}
		v_locations[i] = v_local;

		img[v_local] = i + 1;
	}
	
	
	
	
	
// viz	
	for (int i = 0; i < dim*dim; i++)
		if((i + 1) % dim != 0){
			cout << img[i] << " ";
		} else {
			cout << img[i] << endl;
		}
	
	for(int i = 0; i < v_pts; i++)
		cout << v_locations[i] << " ";

	
	cout << endl;
	
	return 0;
}