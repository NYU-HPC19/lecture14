#include <stdio.h>
#include <string>
#include <mpi.h>
#include "utils.h"

struct Particle {
  double coord[3];
  double velocity[3];
  double mass;
};

void force_kernel(std::vector<double>& force, const std::vector<Particle>& X, MPI_Comm comm) {
  int rank, np;
  MPI_Comm_size(comm, &np);
  MPI_Comm_rank(comm, &rank);

  const std::vector<Particle>& Xt = X;
  long Nt = Xt.size();

  force.resize(Nt*3);
  for (long i = 0; i < Nt*3; i++) force[i] = 0;

  std::vector<Particle> Xs;
  for (int p = 0; p < np; p++) {
    if (p) { // Set Xs
      int sendcount = X.size(), recvcount;
      MPI_Sendrecv(&sendcount, 1, MPI_INT, (rank+np-p)%np, p, &recvcount, 1, MPI_INT, (rank+p)%np, p, comm, MPI_STATUS_IGNORE);
      Xs.resize(recvcount);
      MPI_Sendrecv(&X[0], sendcount, CommDatatype<Particle>::value(), (rank+np-p)%np, p, &Xs[0], recvcount, CommDatatype<Particle>::value(), (rank+p)%np, p, comm, MPI_STATUS_IGNORE);
    } else {
      Xs = X;
    }
    long Ns = Xs.size();

    #pragma omp parallel for schedule(static)
    for (long t = 0; t < Nt; t++) {
      double Xt_[3];
      Xt_[0] = Xt[t].coord[0];
      Xt_[1] = Xt[t].coord[1];
      Xt_[2] = Xt[t].coord[2];

      double f[3] = {0,0,0};
      for (long s = 0; s < Ns; s++) {
        double Xs_[3];
        Xs_[0] = Xs[s].coord[0];
        Xs_[1] = Xs[s].coord[1];
        Xs_[2] = Xs[s].coord[2];

        double dX[3];
        dX[0] = Xs_[0] - Xt_[0];
        dX[1] = Xs_[1] - Xt_[1];
        dX[2] = Xs_[2] - Xt_[2];

        double r2 = dX[0]*dX[0] + dX[1]*dX[1] + dX[2]*dX[2];
        double rinv = (r2>0 ? 1.0 / sqrt(r2) : 0.0);
        double rinv3 = rinv*rinv*rinv;

        f[0] += dX[0] * rinv3 * Xs[s].mass * Xt[t].mass;
        f[1] += dX[1] * rinv3 * Xs[s].mass * Xt[t].mass;
        f[2] += dX[2] * rinv3 * Xs[s].mass * Xt[t].mass;
      }
      force[t*3+0] += f[0]/(4*M_PI);
      force[t*3+1] += f[1]/(4*M_PI);
      force[t*3+2] += f[2]/(4*M_PI);
    }
  }
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

  double dt = 0.00001;
  std::vector<double> force;
  for (double t = 0; t < 1; t+=dt) {

    // Compute gravitational forces
    force_kernel(force, pts, comm);

    #pragma omp parallel for schedule(static)
    for (long k = 0; k < N; k++) {

      // Update velocity: v = v + f/m * dt
      pts[k].velocity[0] += force[k*3+0] / pts[k].mass * dt;
      pts[k].velocity[1] += force[k*3+1] / pts[k].mass * dt;
      pts[k].velocity[2] += force[k*3+2] / pts[k].mass * dt;

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

  MPI_Finalize();
  return 0;
}


