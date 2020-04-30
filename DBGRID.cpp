#include <stdio.h>
#include <inttypes.h>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <cstdlib> 
#include <cmath>
#include <vector>
#include <time.h>
#include <unordered_map>
#include <utility>
#include <map>
#include <queue>
#include <cstring>
#include <random>
#include <chrono>
#include "inc/nanoflann.hpp"

using namespace std;
using namespace nanoflann;

#define DIM 2
#define NO_CLUSTER -1
int mode;

#include <boost/unordered_map.hpp>
#include <boost/container/flat_map.hpp>

/* Global Varaible Declaration */
struct node;

/* Type Definition */

typedef double num_t;
typedef double dis_type;
typedef boost::container::flat_map<int, node> hashtype;

struct point{
	num_t x[DIM];
	int cluster;
	bool is_core;
	point() {is_core=false; cluster=NO_CLUSTER;}
};
struct node{
	vector<point> points;
	hashtype hmpts;
	int count;
	int min;
	int max;
	int position[DIM];
	int cluster;
	int id;
	int bitid;
	int dth;
	bool is_core;
	bool processed;
	node(){is_core=false;  count=0; cluster=NO_CLUSTER; min=10000000; max=-1; processed=false;}
};
/* Global Varaible Declaration */

vector<num_t> p_min(DIM, 0);
vector<num_t> p_max(DIM, 0);
vector<int>    seg(DIM, 0);
int max_slot_hori=0;
int max_slot_orth=0;
int gid=0;
int p = ceil(sqrt(DIM));
vector<int> boundary(2*p+1, 0);

int myrandom (int i) { return std::rand()%i;}
void setBoundary(int dim){
	int d = dim;
	int p = ceil(sqrt(d));
	int oth = ceil(sqrt(d/2.0));
	int m = ceil(sqrt(d - pow((p-1), 2)));
	
	if (p == oth){
		for (int i=0; i < 2*p + 1 ; i++)
		{
			boundary[i] = p;
		}
	}else if (oth < p)
	{
		int bindex = 0;
		for (int i = p-1 ; i >= 0 ; i-- )
		{
			int m = ceil(sqrt(d - pow((i), 2)));
			boundary[bindex] = m;
			boundary[2*p+1 - bindex - 1] = m;
			bindex++;
		}
		boundary[bindex] = p;
	}
}
#define TestBit(A,k)( A[(k/32)] & (1 << (k%32)) )
struct bitmap{
	int* bm[DIM];
	int intlen;
	int gnum;
	vector<node*> cc;
	bitmap(int grid_num, vector<node*> core_cells){
		int intsize = 32;	// In bit
		gnum = grid_num;
		int maxcol  = grid_num;
		intlen = (maxcol/intsize) + 1;
		int maxrow  = 200;
		
		for (int i=0 ; i < DIM; i++)
		{
			bm[i] = (int*)malloc((intlen*sizeof(int))*seg[i]);
			for (int j = 0; j < intlen*seg[i]; j++ )
      			bm[i][j] = 0; 
		}
		for (int i=0; i<core_cells.size() ; i++)
		{
			cc.push_back(core_cells[i]);
		}
		
	}
	void set_bit(int d, int pos,int id) 	{bm[d][intlen*pos + (id/32)] |=  (1 << (id%32));}
	void clear_bit(int d, int pos,int id) {bm[d][intlen*pos + (id/32)] &= ~(1 << (id%32));} 
	bool test_bit(int d, int pos,int id) {return bm[d][intlen*pos + (id/32)] & (1 << (id%32));}
	void build_index()
	{
		cout << cc.size() << endl;
		for (int i=0; i<cc.size() ; i++)
		{
			cc[i]->bitid = i;
			
			for (int j=0 ; j<DIM ; j++)
			{
				//cout << j<< ":"<< cc[i]->position[j] << endl;
				set_bit(j, cc[i]->position[j], i);
			}
			
		}
		
	}
	vector<node*> range_query(int position[])
	{
		//int range[DIM][2];
		vector<node*> v;
		int s,e;
		int tmp[DIM][intlen];
		int res[intlen];
		for (int d=0; d<DIM; d++)
		{
			
			for (int k=0; k<intlen ; k++) tmp[d][k] = 0;
			
			s = position[d] - max_slot_hori;
			e = position[d] + max_slot_hori;
			if (s < 0) s = 0;
			if (e > seg[d] - 1) e = seg[d] - 1;
			
			//cout << s << "," << e << endl;
				
			for (int k=s; k <= e; k++)
				for (int p=0; p<intlen ; p++) 
					tmp[d][p] |= bm[d][k*intlen + p];
				
		}
		
		for (int p=0; p<intlen ; p++) res[p] = tmp[0][p];
		for (int d=1; d<DIM; d++)
		{
			for (int p=0; p<intlen ; p++)
				res[p] &= tmp[d][p];
		}
		
		for (int i=0; i < gnum; i++)
		{
			if (TestBit(res, i))
			{
				v.push_back(cc[i]);
			}
		}
		return v;
	}
};

num_t dist(const point& x, const point& y, int dim){
    num_t t, d = 0;
    while (dim--) {
        t = x.x[dim] - y.x[dim];
        d += t * t;
    }
    return d;
}

bool comp(const node *lhs, const node *rhs) 
{
	//return lhs.points.size() < rhs.points.size();
	for (int i=0; i < DIM ; i++)
    {
		 if (lhs->position[i] == rhs->position[i]) continue;
		 return lhs->position[i] < rhs->position[i];
	}
}

template <typename T>
struct PointCloud
{
	std::vector<point>  pts;
	// Must return the number of data points
	inline size_t kdtree_get_point_count() const { return pts.size(); }
	// Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the class:
	inline num_t kdtree_distance(const T *p1, const size_t idx_p2,size_t size) const
	{
		num_t sum=0, t;
		for (int i=0; i < DIM ; i++)
		{
			t=(p1[i]-pts[idx_p2].x[i]);
			sum += t*t;
		}
		return sum;
	}
	// Returns the dim'th component of the idx'th point in the class:
	// Since this is inlined and the "dim" argument is typically an immediate value, the
	//  "if/else's" are actually solved at compile time.
	inline num_t kdtree_get_pt(const size_t idx, int dim) const
	{
		return pts[idx].x[dim];
	}
	// Optional bounding-box computation: return false to default to a standard bbox computation loop.
	//   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
	//   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
	template <class BBOX>
	bool kdtree_get_bbox(BBOX &bb) const { return false; }

};	

typedef KDTreeSingleIndexAdaptor<
		L2_Simple_Adaptor<num_t, PointCloud<num_t> > ,
		PointCloud<num_t>,
		DIM /* dim */
		> my_kd_tree_t;
		
void read_dataset(const char* path, vector<point> &ds,int n,int mode){
     ifstream infile(path, ios::in);
     if (!infile.is_open())
     {
			cout << "File cannot be open." << endl;
			cout << "Exit program with code -1" << endl;
			system("pause");
			exit(-1);
	 }
	 int i = 0;
     num_t x, y, z;
     for (int k=0 ; k < n ; k++)
     {
           num_t in;
           point n;
           for (int j = 0; j < DIM ; j++)
           {
				infile >> in;
				
				if (i == 0)
	            {
	             	p_min[j] = in;
	             	p_max[j] = in;
	             	n.x[j] = in;
	            }
	            else
	            {
					n.x[j] = in;
	            	if (p_min[j] > in) p_min[j] = in;
	            	if (p_max[j] < in) p_max[j] = in;
				}
		   }
		   if (mode == 3)
		   infile >> z;
		   ds.push_back(n);
           i++;
     }
     infile.close();
}
vector<node*> get_all_core_cells(node* root,vector<vector<int>>& positions){
	vector<node*> v;
	for (int i=0 ; i < positions.size() ;i++)
	{
		node*r = root;
		for (int j=0 ; j < DIM ; j++)
		{
			r=&r->hmpts.find(positions[i][j])->second;
		}
		v.push_back(r);
	}
	return v;
}
void build_tree(vector<point> &ds, node* root, const dis_type b_width, const int MINPTS)
{
	vector<node*> grids;
	for (int i = 0; i < ds.size(); i++)
    {
		node* r=root;
		int temp[DIM];
		for (int j = 0; j < DIM ; j++)
		{
			int ind = (int)(((ds[i].x[j] - p_min[j])*1.)/b_width);
			temp[j] = ind;
			auto tmp=r->hmpts.find(ind);
			if (tmp == r->hmpts.end())
			{
				node n;
				n.count = 0;
				n.dth = ind;
				n.id = gid++;
				if (ind > r->max) r->max = ind;
				if (ind < r->min) r->min = ind;
				r->hmpts.insert(make_pair(ind,n));
			}
			r->count++;
			r = &(r->hmpts.find(ind)->second);
		}
		
		r->points.push_back(ds[i]);
		if (r->points.size() >= MINPTS) r->is_core = true;
		memcpy (r->position, temp, sizeof(int)*DIM);
	}
}
vector<node*> get_nodes(node* root){
	vector<node*> v;
	vector<pair<node*,int>> stack;
	stack.push_back(make_pair(root, 0));
	pair<node*,int> tmp;
	hashtype::iterator tr;
	
	while (stack.size() > 0)
	{
		tmp = stack[stack.size() - 1];
		stack.pop_back();
		int dim = tmp.second;
		node* r = tmp.first;
		
		if (r->points.size() > 0)
		{
			v.push_back(r);
		}
		
		for (tr = r->hmpts.begin(); tr != r->hmpts.end() ; tr++)
		{
			stack.push_back(make_pair(&tr->second, dim+1));
		}
	}
	return v;
}
void label_core_cell(bitmap& bm, int MINPTS, dis_type EPS)
{
	vector<node*> G;
	for (int i=0 ; i< bm.cc.size() ; i++)
	{

		if (bm.cc[i]->points.size() >= MINPTS) {
			bm.cc[i]->is_core = true;
			for (int j=0 ; j<bm.cc[i]->points.size(); j++)
				bm.cc[i]->points[j].is_core = true;
		}else
		{
			G = bm.range_query(bm.cc[i]->position);
			int count = 0;
			for (int k=0; k < bm.cc[i]->points.size() ; k++){
				count = 0;
				for (int j=0; j < G.size() ; j++){
					for (int m=0; m < G[j]->points.size() ; m++)
					{
						if (dist(bm.cc[i]->points[k], G[j]->points[m], DIM) <= EPS*EPS)
						{
							count++;
							if (count >= MINPTS)
							{
								bm.cc[i]->points[k].is_core = true;
								bm.cc[i]->is_core = true;
								break;
							}
						}
					}
					if (count >= MINPTS) break;
				}
			}
		}
	}
}

bool is_mergeable(my_kd_tree_t& kd, const node& n, dis_type EPS)
{
     
     if (n.points.size() <= 0) return false;
     const size_t num_results = 1;
	 std::vector<size_t>   ret_index(num_results);
	 std::vector<num_t> out_dist_sqr(num_results);
     int indices[num_results];
     dis_type dists[num_results];
     for (int i = 0 ; i < n.points.size(); i++)
     {
		  if (!n.points[i].is_core) continue;
		  kd.knnSearch(&n.points[i].x[0], num_results, &ret_index[0], &out_dist_sqr[0], 1);
		  if (out_dist_sqr[0] <= EPS*EPS) return true;
     }
     return false;
}
int rooted_cluster(map<int,int> &cluster_references, int c){
	
	if (c == NO_CLUSTER) return NO_CLUSTER;
	int cl = c;
	int tmp = cluster_references.find(cl)->second;
	while (tmp != cl)
	{
		cl = tmp;
		tmp = cluster_references.find(tmp)->second;
	}
	return tmp;
}
void merge_core_cells(node* root, vector<node> &ds, double EPS, map<int,int> &cluster_references, 
 bitmap &bm){
	
	node* n;
	int C=0;
	int merge_times=0, iskip_times=0, imerge_times=0;
	map<int,vector<node*>> watching_lists;
	clock_t stt, end;
	double getnodes = 0.0;
	double clster_forest = 0.0;
	double build_tree = 0.0;
	double get_nb = 0.0;
	double find_cluster = 0.0;
	
	vector<node*> priority_stack;
	vector<node*> normal_queue;
	vector<node*> core_cells;
	
	int count_cc = 0;
	for (int i=0 ; i < bm.cc.size() ; i++)
	{
		if (bm.cc[i]->is_core) {
			count_cc++;
			normal_queue.push_back(bm.cc[i]);
			core_cells.push_back(bm.cc[i]);
		}
	}
	cout << "#core cells = " << count_cc << endl;
	
	if (mode==1 || mode==2){
		cout << "NEXT-ONE!" << endl;
	}
	if(mode==3 || mode==4){
		cout << "ALL-NEIGH!" << endl;
	}
	cout << "#core cells to be processed: " << core_cells.size() << endl;
	//for (int i=0 ; i < ds.size() ; i++)
	
	if (mode == 2 || mode == 4 || mode == 5){
		std::sort(normal_queue.begin(), normal_queue.end(), comp);
		cout << "SORTED." << endl;
	}
	if (mode == 1 || mode == 3 || mode == 6 ){
		random_shuffle(normal_queue.begin(), normal_queue.end(), myrandom);
		cout << "SHUFFLED." << endl;
	}
	
	int i = 0;
	while (priority_stack.size() + normal_queue.size() > 0)
	{
		//cout << i << endl;
		iskip_times=0;
		imerge_times=0;
		if (priority_stack.size() > 0)
		{
			n = priority_stack[priority_stack.size() - 1];
			priority_stack.pop_back();
		}
		else
		{
			n = normal_queue[normal_queue.size() - 1];
			normal_queue.pop_back();
		}
		
		if (n->processed) continue;
		n->processed = true;
		i++;	
stt = clock();

		vector<node*> A;
		A.push_back(n);
		
		vector<node*> G;
		vector<node*> H;
stt = clock();
		G = bm.range_query(n->position);
get_nb += (clock() - stt);

		vector<node*> F;

		int cur_cluster;
			
		if(mode==3 || mode==4 || mode == 5 || mode == 6){
			for (int j=0; j < G.size() ; j++)
			{
				if (G[j]->is_core)
					F.push_back(G[j]);
			}
			
		}
getnodes += (clock() - stt);


stt = clock();
		
		if (n->cluster == NO_CLUSTER){
			C++;
			cluster_references[C] = C;
			n->cluster = C;
			cur_cluster = C;
			
		}else{
			
			cur_cluster = rooted_cluster(cluster_references, n->cluster);
		}
clster_forest += (clock() - stt);
		
stt = clock();
		PointCloud<num_t> cloud;
		
		for (int k=0 ; k < n->points.size() ; k++)
		{
			if (!n->points[k].is_core) continue;
			cloud.pts.push_back(n->points[k]);
		}
	   	
	   	my_kd_tree_t   index(DIM, cloud, KDTreeSingleIndexAdaptorParams(100) );
		index.buildIndex();
build_tree += (clock() - stt);
		
stt = clock();

		for (int j=0; j < F.size() ; j++)
		{
			if (rooted_cluster(cluster_references, F[j]->cluster) == cur_cluster)
			{
				iskip_times++;
				continue;
			}
			else
			{
				imerge_times++;
				merge_times++;
				if (is_mergeable(index, *F[j], EPS))
				{
					
					if (F[j]->cluster == NO_CLUSTER)
						F[j]->cluster = cur_cluster;
					else 
						A.push_back(F[j]);
						
					if (mode==5 || mode == 6){
						if (!F[j]->processed)
							priority_stack.push_back(F[j]);	
					}	
					
				}
				
			}
		}
		
		int min = cur_cluster;
		int c;
		for (int j=0 ; j < A.size() ; j++)
		{
			c = rooted_cluster(cluster_references, A[j]->cluster);
			if (c < min) min = c;
		}
	
		for (int j=0 ; j < A.size() ; j++)

		{
			//A[j]->cluster = min;
			c = rooted_cluster(cluster_references, A[j]->cluster);
			cluster_references[c] = min;
		}
		
		cluster_references[cur_cluster] = min;
	}
	cout << "Mode : " << mode << endl;
clster_forest += (clock() - stt);
stt = clock();	
	map<int,int> cc;
	for (int i=0 ; i < bm.cc.size() ; i++)
	{
		
		n = bm.cc[i];
		if (!n->is_core) continue;
		int cl = n->cluster;
		int tmp = cluster_references.find(cl)->second;
		while (tmp != cl)
		{
				cl = tmp;
				tmp = cluster_references.find(tmp)->second;
		}
		cc.insert(make_pair(tmp,1));		
	}
	
find_cluster += (clock() - stt);

	
	cout << "get_node():" << (double)getnodes/CLOCKS_PER_SEC << endl;
	cout << "clster_forest:" << (double)clster_forest/CLOCKS_PER_SEC << endl;
	cout << "build_tree():" << (double)build_tree/CLOCKS_PER_SEC << endl;
	cout << "get_nb():" << (double)get_nb/CLOCKS_PER_SEC << endl;
	cout << "find_cluster():" << (double)find_cluster/CLOCKS_PER_SEC << endl;
	
	cout << "#cluster=" << cc.size() << endl;
	cout << "#merging times=" << merge_times << endl;
}

void label_noise_and_border(bitmap &bm, const dis_type EPS)
{
	for (int i=0 ; i<bm.cc.size() ; i++)
	{
		vector<node*> bn_list;
		if (bm.cc[i]->is_core) continue;
		
	 	bn_list = bm.range_query(bm.cc[i]->position);
	
		for (int j=0 ; j < bm.cc[i]->points.size() ; j++)
		{
			if (bm.cc[i]->points[j].is_core) continue; 
			int best_grid = -1;
			dis_type best_grid_dist = 5000000.0;
		 	for (int z=0; z < bn_list.size() ; z++)
		    {
				if (!bn_list[z]->is_core) continue;
				
				for (int m=0; m < bn_list[z]->points.size() ; m++)
				{
					if (!bn_list[z]->points[m].is_core) continue;

					dis_type d = dist(bm.cc[i]->points[j], bn_list[z]->points[m], DIM);
	                     
                    if ((j==0 || d < best_grid_dist) && d < EPS*EPS )
                    {
						 best_grid = j;
						 best_grid_dist = d;
						 bm.cc[i]->points[j].cluster = bn_list[z]->cluster;
					}
				}
			}
		}
	}
}


int main(int argc, char* argv[])
{
	
	const dis_type EPS = atof(argv[2]);
  const int    MINPTS = atof(argv[3]);
	//DIM = atof(argv[4]);
  const dis_type b_width = EPS/sqrt(DIM);
	
	max_slot_hori = (int)(ceil(sqrt(DIM)));
	max_slot_orth = (int)(ceil(sqrt(DIM)/sqrt(2)));
	
	setBoundary(DIM);
	
	//std::srand ( unsigned ( std::time(0) ) );
	
	cout << "----- Bitmap DBSCAN -----" << endl;
	cout << "File: " << argv[1] << endl;
	cout << "N : " << argv[4]  << " objects"<< endl;
	cout << "Dimension: " << DIM << endl;
	cout << "b_width: " << b_width << endl;
	cout << "EPS: " << EPS << endl;
	cout << "MINPTS: " << MINPTS << endl;
	cout << "MODE: " << argv[5] << endl;
	cout << "max_slot_hori : " << max_slot_hori  << endl;
	cout << "------------------------------------------" << endl;
	vector<point> ds;
	
	mode = atoi(argv[5]);
	read_dataset(argv[1], ds, atoi(argv[4]), 2);
	
	clock_t tStart2 = clock();
	cout << ds.size() << endl;
	
	for (int i=0; i < DIM; i++){
		seg[i] = (int) ((p_max[i] - p_min[i])/b_width) +1;
	}
	
	cout << "MAX/MIN:" << endl;
	for (int i=0 ; i < DIM ; i++)
	{
		cout << "Dim " << i+1 << ": MIN/MAX: " << p_min[i] << ", " << p_max[i] << endl;
	}
	
	cout << "We are partitioning the matrix into :" ;
	for (int i=0 ; i < DIM ; i++)
	{
		cout << seg[i] << "x" ;
	}

	node root;
	clock_t tStart = clock();
	build_tree(ds,&root,b_width, MINPTS);
	printf("Time taken [Build tree]: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
	
	cout << "#points="<< ds.size() << endl;
	vector<vector<int>> positions, bn_positions;
	std::vector<node> core_cells, borders_noises;
	tStart = clock();
	
	vector<node*> v = get_nodes(&root);
	
	cout << "#all cells=" << v.size() << endl;
	
	tStart = clock();
	
	
	
	printf("Time. taken [Sort Core Cells]: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
	
	cout << "Build bitmap index..." << endl;
	
	random_shuffle(v.begin(), v.end(), myrandom);
	bitmap bm(v.size(), v);
	bm.build_index();
	
	
	
	label_core_cell(bm, MINPTS, EPS);
	
	printf("Time taken [Label Core Cells]: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);

	map<int,int> cluster_references;
	tStart = clock();
	merge_core_cells(&root, core_cells, EPS, cluster_references, bm);
	printf("Time taken [Merge Cells]: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
	
	
	tStart = clock();
	label_noise_and_border(bm, EPS);
	printf("Time taken [Border/Noise]: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
	printf("Time taken: %.2fs\n", (double)(clock() - tStart2)/CLOCKS_PER_SEC);
	//write_result("ghmout.txt", core_cells, borders_noises, cluster_references);
	
	
	system("pause");
	return 0;
}
