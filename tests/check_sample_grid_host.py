from pathlib import Path
import subprocess
import tempfile
# Executes the actual device math as C++ with CUDA qualifiers erased. This
# validates traversal against exhaustive search without requiring a GPU; use
# `make test` separately to validate GPU sampling, scans, and atomic scatter.
root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix='sample-grid-') as temp:
    d = Path(temp)
    (d/'curand_kernel.h').write_text('struct curandStateXORWOW_t {}; inline void curand_init(unsigned,int,int,curandStateXORWOW_t*) {} inline float curand_uniform(curandStateXORWOW_t*) { return .5f; }\n')
    (d/'cuda_runtime.h').write_text('')
    source = '''
#define __device__
#define __host__
#define __global__
#define __forceinline__ inline
#include "utils.cuh"
#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>
using std::isfinite;
struct {int x;} blockIdx, blockDim, threadIdx;
struct Primitive {Triangle triangle; int object, id, padding;};
unsigned long long atomicAdd(unsigned long long* p, unsigned long long x) {auto old=*p; *p+=x; return old;}
'''
    source += (root/'sampled_grid.cuh').read_text().split('struct OwnedSampleGrid')[0]
    source += '''
int main() {
 auto require=[](bool ok) { if(!ok)throw std::runtime_error("Host rule/sampling/CLI check failed"); };
 {
  RenderSettings settings{};std::string out,ref,error;
  const char* args[]={"--intersection","bvh","--sample-spacing",".13","--acceptance-radius",".23","--cube-size",".1"};
  require(rtscene_detail::parse_cli_args(8,const_cast<char**>(args),settings,out,ref,error));
  require(!settings.sampled_intersections && settings.sample_spacing==.13f && settings.acceptance_radius==.23f && settings.cube_size==.1f);
  const char* bad[]={"--sample-spacing","0"};
  require(!rtscene_detail::parse_cli_args(2,const_cast<char**>(bad),settings,out,ref,error));
  const char* invalid[]={"--intersection","invalid"};
  require(!rtscene_detail::parse_cli_args(2,const_cast<char**>(invalid),settings,out,ref,error));
 }
 {
  Primitive triangles[]={{{Vec3(0,0,0),Vec3(1,0,0),Vec3(0,1,0)},2,0,0},
                          {{Vec3(0,0,0),Vec3(0,0,0),Vec3(0,0,0)},1,1,0}};
  int divisions[2];GridOffset counts[3], offsets[3];
  blockIdx.x=0;blockDim.x=1;
  for(int i=0;i<3;++i){threadIdx.x=i;sampling_counts(triangles,2,.13f,divisions,counts);}
  require(divisions[0]==11 && counts[0]==78 && counts[1]==0);
  offsets[0]=0;offsets[1]=78;offsets[2]=78;
  SurfaceSample samples[78];
  for(int i=0;i<78;++i){threadIdx.x=i;sample_surfaces(triangles,2,divisions,offsets,samples,78);}
  for(const auto& p:samples)require(p.position.x>=0 && p.position.y>=0 && p.position.x+p.position.y<=1.000001f && p.normal.z==1 && p.object==2 && p.triangle==0);
  require(samples[0].position.x==0 && samples[66].position.x==1 && samples[77].position.y==1);
 }
 {
  const SurfaceSample points[]={
   {Vec3(.9f,0,1),0,Vec3(),0},{Vec3(0,0,1.1f),1,Vec3(),0},
   {Vec3(0,0,-1),2,Vec3(),0},{Vec3(0,0,EPSILON),3,Vec3(),0},
   {Vec3(1.01f,0,.01f),4,Vec3(),0},{Vec3(0,0,.1f),42,Vec3(),0},
   {Vec3(1,0,1),6,Vec3(),0},{Vec3(1,0,1),7,Vec3(),0}};
  SampleGrid g;g.samples=points;g.radius=1;Ray ray{Vec3(),Vec3(0,0,1)};SampleClosest best;
  for(int i=0;i<6;++i)sample_candidate(ray,g,i,42,best);
  require(best.index==1);best=SampleClosest();
  for(int i:{7,6,7})sample_candidate(ray,g,i,-1,best);
  require(best.index==6);
 }
 std::mt19937 rng(81); std::uniform_real_distribution<float> pos(-2,2);
 long long tested=0;
 for(float radius : {.003f,.07f,.23f}) for(float cell : {.04f,.1f,.2f}) {
  SampleGrid g; g.lo=Vec3(-1,-1,-1); g.cell=cell; g.radius=radius;
  for(int a=0;a<3;++a) g.dims[a]=int(2/cell)+1;
  g.hi=g.lo+Vec3(g.dims[0]*cell,g.dims[1]*cell,g.dims[2]*cell);
  std::vector<SurfaceSample> samples;
  for(int i=0;i<120;++i) samples.push_back({Vec3(pos(rng)*.3f,pos(rng)*.3f,pos(rng)*.3f), i%5, Vec3(0,0,1), 0});
  samples.push_back({Vec3(0,0,0),0,Vec3(0,0,1),0});
  g.sample_count=samples.size();g.samples=samples.data();
  size_t cells=size_t(g.dims[0])*g.dims[1]*g.dims[2];
  std::vector<std::vector<int>> buckets(cells);
  for(int i=0;i<g.sample_count;++i) {
   int low[3],high[3]; overlapped_cells(g,samples[i].position,low,high);
   for(int z=low[2];z<=high[2];++z) for(int y=low[1];y<=high[1];++y) for(int x=low[0];x<=high[0];++x)
    buckets[cell_index(g,x,y,z)].push_back(i);
  }
  std::vector<GridOffset> offsets(cells+1);std::vector<int> refs;
  for(size_t c=0;c<cells;++c) {offsets[c]=refs.size(); refs.insert(refs.end(),buckets[c].begin(),buckets[c].end());}
  offsets[cells]=refs.size();g.offsets=offsets.data();g.references=refs.data();
  std::vector<Ray> rays;
  for(int i=0;i<20000;++i) {
   Vec3 o(pos(rng),pos(rng),pos(rng)),dir(pos(rng),pos(rng),pos(rng));
   if(i%4==0)dir=Vec3(0,0,i%8?1:-1);
   if(i%4==1)dir=Vec3(0,i%8?1:-1,0);
   if(i%4==2)dir=Vec3(i%8?1:-1,0,0);
   rays.push_back({o,normalize(dir)});
  }
  for(const auto& p:samples)for(float f:{.9999f,1.f,1.0001f})
   rays.push_back({p.position+Vec3(radius*f,0,-2),Vec3(0,0,1)});
  for(int k=0;k<g.dims[0];++k) for(Vec3 dir:{Vec3(0,0,1),normalize(Vec3(1,1,1)),Vec3(-1,0,0)})
   rays.push_back({g.lo+Vec3(k*cell,k*cell,k*cell),dir});
  for(const Ray& ray:rays){
   SampleClosest expected;for(int i=0;i<g.sample_count;++i)sample_candidate(ray,g,i,-1,expected);
   auto actual=sampled_hit(ray,g,-1);
   if(expected.index!=actual.index){std::cerr<<"Mismatch R="<<radius<<" cell="<<cell<<" expected="<<expected.index<<" actual="<<actual.index<<"\\n";return 1;}
   ++tested;
  }
 }
 std::cout<<"PASS: "<<tested<<" actual DDA searches agree with exhaustive search\\n";
}
'''
    (d/'test.cpp').write_text(source)
    subprocess.run(['clang++','-std=c++17','-O2','-Wno-unknown-pragmas','-I'+str(d),'-I'+str(root),str(d/'test.cpp'),'-o',str(d/'test')],check=True)
    subprocess.run([str(d/'test')],check=True)
