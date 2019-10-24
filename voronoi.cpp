#include <iostream>
#include <random>

using namespace std;

// declarations
bool is_a_center(long, int[], long);
long closest_center(long, int[], long, long);

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
		
		while(img[v_local] != 0){
			v_local = distr(generator);
		}
		v_locations[i] = v_local;

		img[v_local] = i + 1;
	}
	
	// calculate distances
	for (int i = 0; i < dim*dim; i++){
		// check if pixel is a center
		if (!is_a_center(i, v_locations, v_pts)){
			img[i] = closest_center(i, v_locations, v_pts, dim);
		} 
		
	}
	
	
	
	
// viz	
	for (int i = 0; i < dim*dim; i++)
		if((i + 1) % dim != 0){
			cout << img[i] << " ";
		} else {
			cout << img[i] << endl;
		}
	
	for(int i = 0; i < v_pts; i++){
		cout << v_locations[i] << " ";
	}	

	
	cout << endl;
	
	return 0;
}

bool is_a_center(long check, int center_list[], long size){
	for(long i = 0; i < size; i++){
		if(center_list[i] == check) return true;
	}
	return false;
}

long closest_center(long point, int center_list[], long center_list_size, long dim){
	float distance;
	float shortest_dist = 0.0;
	long x_point, y_point;
	long x_center, y_center;
	long closest_center;
	
	// extract x and y components
	x_point = point / dim;
	y_point = point % dim;
	
	for(long i = 0; i < center_list_size; i++){
		x_center = center_list[i] / dim;
		y_center = center_list[i] % dim;
			
		distance = sqrt(pow(1.0 * x_center - x_point, 2) + pow(1.0 * y_center - y_point, 2));
		//cout << "center: (" << x_center << ", " << y_center << ")" << endl;
		//cout << "point:  (" << x_point << ", " << y_point << ")" << endl;
		//cout << "distance: " << distance << endl;
		
		// are we in the first iteration? Take that distance
		if(i == 0){
			cout << "i is " << i << endl;
			cout << "shortest_dist = " << shortest_dist << endl;
			cout << "distance = " << distance << endl;
			
			shortest_dist = distance;
			closest_center = center_list[i];
			
			cout << "shortest_dist = " << shortest_dist << endl;
			cout << "distance = " << distance << endl;
			
		// if not, then check to see if the new distance we calculated is smaller
		} else if(distance < shortest_dist){
			cout << "i is " << i << endl;
			cout << "shortest_dist = " << shortest_dist << endl;
			cout << "distance = " << distance << endl;
			shortest_dist = distance;
			closest_center = center_list[i];
			cout << "shortest_dist = " << shortest_dist << endl;
			cout << "distance = " << distance << endl;
			
		}
	}
	
	return closest_center;
}