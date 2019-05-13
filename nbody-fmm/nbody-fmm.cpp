#include <stdio.h>
#include <string>
#include <mpi.h>
#include <pvfmm.h>
#include "utils.h"

struct Particle {
  double coord[3];
  double velocity[3];
  double mass;
};

void force_kernel(std::vector<double>& force, const std::vector<Particle>& X, void* ctx) {
  long N = X.size();
  force.resize(N*3);
  std::vector<double> pos(N*3), sl_den(N);
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) {
    pos[i*3+0] = X[i].coord[0];
    pos[i*3+1] = X[i].coord[1];
    pos[i*3+2] = X[i].coord[2];

    sl_den[i] = X[i].mass;

    force[i*3+0] = 0;
    force[i*3+1] = 0;
    force[i*3+2] = 0;
  }

  int setup = 1;
  PVFMMEvalD(&pos[0], &sl_den[0], nullptr, N, &pos[0], &force[0], N, ctx, setup);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;

  int rank, np;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &np);
  srand48(rank);

  long N = read_option<long>("-N", argc, argv)/np;
  std::vector<Particle> pts(N);
  for (long i = 0; i < N; i++) { // Initialize pts (coordinates, velocity, mass)
    double X = 0.5 + (drand48()-0.5)*0.1;
    double Y = 0.5 + (drand48()-0.5)*0.1;
    double Z = 0.5 + (drand48()-0.5)*0.1;

    pts[i].coord[0] = X;
    pts[i].coord[1] = Y;
    pts[i].coord[2] = Z;

    pts[i].velocity[0] = ((drand48()-0.5)*2*0.2+(Y-0.5)*100);
    pts[i].velocity[1] = ((drand48()-0.5)*2*0.2-(X-0.5)*100);
    pts[i].velocity[2] = ((drand48()-0.5)*2*0.2);

    pts[i].mass = 14.0/N;
  }

  // Create FMM context with Laplace-gradient kernel
  void* ctx = PVFMMCreateContextD(-1, 250, 8, PVFMMLaplaceGradient, comm);

  double dt = 0.00001;
  std::vector<double> force;
  for (double t = 0; t < 1; t+=dt) {

    // Compute gravitational forces
    force_kernel(force, pts, ctx);

    #pragma omp parallel for schedule(static)
    for (long k = 0; k < N; k++) {

      // Update velocity: v = v + f/m * dt
      pts[k].velocity[0] += force[k*3+0] * dt;
      pts[k].velocity[1] += force[k*3+1] * dt;
      pts[k].velocity[2] += force[k*3+2] * dt;

      // Update position: x = x + v * dt
      pts[k].coord[0] += pts[k].velocity[0]*dt;
      pts[k].coord[1] += pts[k].velocity[1]*dt;
      pts[k].coord[2] += pts[k].velocity[2]*dt;
    }

    if ((int)(t*1000) != (int)((t+dt)*1000)) { // Write VTK
      VTUData vtk_data;
      for (long i = 0; i < N; i++) {
        vtk_data.coord.push_back(pts[i].coord[0]);
        vtk_data.coord.push_back(pts[i].coord[1]);
        vtk_data.coord.push_back(pts[i].coord[2]);
        vtk_data.connect.push_back(i);
      }
      vtk_data.offset.push_back(vtk_data.connect.size());
      vtk_data.types.push_back(2);

      vtk_data.WriteVTK((std::string("pts")+std::to_string((int)(t*1000))).c_str(), comm);
    }
    if (!rank) printf("Time = %f\n", t*1000);
  }

  PVFMMDestroyContextF(&ctx);
  MPI_Finalize();
  return 0;
}


