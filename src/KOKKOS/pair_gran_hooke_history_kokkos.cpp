/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "pair_gran_hooke_history_kokkos.h"
#include "kokkos.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "memory_kokkos.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "error.h"
#include "modify.h"
#include "fix_neigh_history_kokkos.h"
#include "update.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairGranHookeHistoryKokkos<DeviceType>::PairGranHookeHistoryKokkos(LAMMPS *lmp) : PairGranHookeHistory(lmp)
{
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | V_MASK | OMEGA_MASK | F_MASK | TORQUE_MASK | TYPE_MASK | MASK_MASK | ENERGY_MASK | VIRIAL_MASK | RMASS_MASK | RADIUS_MASK;
  datamask_modify = F_MASK | TORQUE_MASK | ENERGY_MASK | VIRIAL_MASK;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairGranHookeHistoryKokkos<DeviceType>::~PairGranHookeHistoryKokkos()
{
  if (copymode) return;

  if (allocated) {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->destroy_kokkos(k_vatom,vatom);
    eatom = NULL;
    vatom = NULL;
  }
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

template<class DeviceType>
void PairGranHookeHistoryKokkos<DeviceType>::init_style()
{
  if (history && fix_history == NULL) {
    char dnumstr[16];
    sprintf(dnumstr,"%d",3);
    char **fixarg = new char*[4];
    fixarg[0] = (char *) "NEIGH_HISTORY";
    fixarg[1] = (char *) "all";
    if (execution_space == Device)
      fixarg[2] = (char *) "NEIGH_HISTORY/KK/DEVICE";
    else
      fixarg[2] = (char *) "NEIGH_HISTORY/KK/HOST";
    fixarg[3] = dnumstr;
    modify->add_fix(4,fixarg,1);
    delete [] fixarg;
    fix_history = (FixNeighHistory *) modify->fix[modify->nfix-1];
    fix_history->pair = this;
    fix_historyKK = (FixNeighHistoryKokkos<DeviceType> *)fix_history;
  }

  PairGranHookeHistory::init_style();

  // irequest = neigh request made by parent class

  neighflag = lmp->kokkos->neighflag;
  int irequest = neighbor->nrequest - 1;

  neighbor->requests[irequest]->
    kokkos_host = Kokkos::Impl::is_same<DeviceType,LMPHostType>::value &&
    !Kokkos::Impl::is_same<DeviceType,LMPDeviceType>::value;
  neighbor->requests[irequest]->
    kokkos_device = Kokkos::Impl::is_same<DeviceType,LMPDeviceType>::value;

  if (neighflag == HALF || neighflag == HALFTHREAD) {
    neighbor->requests[irequest]->full = 0;
    neighbor->requests[irequest]->half = 1;
  } else {
    error->all(FLERR,"Cannot use chosen neighbor list style with gran/hooke/history/kk");
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairGranHookeHistoryKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{
  copymode = 1;

  eflag = eflag_in;
  vflag = vflag_in;

  if (neighflag == FULL) no_virial_fdotr_compute = 1;

  if (eflag || vflag) ev_setup(eflag,vflag,0);
  else evflag = vflag_fdotr = 0;

  int shearupdate = 1;
  if (update->setupflag) shearupdate = 0;

  // reallocate per-atom arrays if necessary

  if (eflag_atom) {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->create_kokkos(k_eatom,eatom,maxeatom,"pair:eatom");
    d_eatom = k_eatom.view<DeviceType>();
  }
  if (vflag_atom) {
    memoryKK->destroy_kokkos(k_vatom,vatom);
    memoryKK->create_kokkos(k_vatom,vatom,maxvatom,6,"pair:vatom");
    d_vatom = k_vatom.view<DeviceType>();
  }

  atomKK->sync(execution_space,datamask_read);
  if (eflag || vflag) atomKK->modified(execution_space,datamask_modify);
  else atomKK->modified(execution_space,F_MASK | TORQUE_MASK);

  x = atomKK->k_x.view<DeviceType>();
  v = atomKK->k_v.view<DeviceType>();
  omega = atomKK->k_omega.view<DeviceType>();
  c_x = atomKK->k_x.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  torque = atomKK->k_torque.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();
  tag = atomKK->k_tag.view<DeviceType>();
  rmass = atomKK->k_rmass.view<DeviceType>();
  radius = atomKK->k_radius.view<DeviceType>();
  nlocal = atom->nlocal;
  nall = atom->nlocal + atom->nghost;
  newton_pair = force->newton_pair;
  special_lj[0] = force->special_lj[0];
  special_lj[1] = force->special_lj[1];
  special_lj[2] = force->special_lj[2];
  special_lj[3] = force->special_lj[3];

  int inum = list->inum;
  NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
  d_numneigh = k_list->d_numneigh;
  d_neighbors = k_list->d_neighbors;
  d_ilist = k_list->d_ilist;

  d_firsttouch = fix_historyKK->d_firstflag;
  d_firstshear = fix_historyKK->d_firstvalue;
  
  EV_FLOAT ev;

  if (lmp->kokkos->neighflag == HALF) {
    if (force->newton_pair) {
      if (evflag) {
	if (shearupdate) {
	  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairGranHookeHistoryCompute<HALF,1,1,1>>(0,inum),*this);
	} else {
	  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairGranHookeHistoryCompute<HALF,1,1,0>>(0,inum),*this);
	}
      } else {
	if (shearupdate) {
	  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairGranHookeHistoryCompute<HALF,1,0,1>>(0,inum),*this);
	} else {
	  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairGranHookeHistoryCompute<HALF,1,0,0>>(0,inum),*this);
	}
      }
    } else {
      if (evflag) {
	if (shearupdate) {
	  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairGranHookeHistoryCompute<HALF,0,1,1>>(0,inum),*this);
	} else {
	  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairGranHookeHistoryCompute<HALF,0,1,0>>(0,inum),*this);
	}
      } else {
	if (shearupdate) {
	  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairGranHookeHistoryCompute<HALF,0,0,1>>(0,inum),*this);
	} else {
	  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairGranHookeHistoryCompute<HALF,0,0,0>>(0,inum),*this);
	}
      }
    }
  } else { // HALFTHREAD
    if (force->newton_pair) {
      if (evflag) {
	if (shearupdate) {
	  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairGranHookeHistoryCompute<HALFTHREAD,1,1,1>>(0,inum),*this);
	} else {
	  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairGranHookeHistoryCompute<HALFTHREAD,1,1,0>>(0,inum),*this);
	}
      } else {
	if (shearupdate) {
	  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairGranHookeHistoryCompute<HALFTHREAD,1,0,1>>(0,inum),*this);
	} else {
	  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairGranHookeHistoryCompute<HALFTHREAD,1,0,0>>(0,inum),*this);
	}
      }
    } else {
      if (evflag) {
	if (shearupdate) {
	  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairGranHookeHistoryCompute<HALFTHREAD,0,1,1>>(0,inum),*this);
	} else {
	  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairGranHookeHistoryCompute<HALFTHREAD,0,1,0>>(0,inum),*this);
	}
      } else {
	if (shearupdate) {
	  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairGranHookeHistoryCompute<HALFTHREAD,0,0,1>>(0,inum),*this);
	} else {
	  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairGranHookeHistoryCompute<HALFTHREAD,0,0,0>>(0,inum),*this);
	}
      }
    }
  }

  if (eflag_global) eng_vdwl += ev.evdwl;
  if (vflag_global) {
    virial[0] += ev.v[0];
    virial[1] += ev.v[1];
    virial[2] += ev.v[2];
    virial[3] += ev.v[3];
    virial[4] += ev.v[4];
    virial[5] += ev.v[5];
  }

  if (eflag_atom) {
    k_eatom.template modify<DeviceType>();
    k_eatom.template sync<LMPHostType>();
  }

  if (vflag_atom) {
    k_vatom.template modify<DeviceType>();
    k_vatom.template sync<LMPHostType>();
  }

  if (vflag_fdotr) pair_virial_fdotr_compute(this);

  copymode = 0;
}

template<class DeviceType>
template<int NEIGHFLAG, int NEWTON_PAIR, int EVFLAG, int SHEARUPDATE>
KOKKOS_INLINE_FUNCTION
void PairGranHookeHistoryKokkos<DeviceType>::operator()(TagPairGranHookeHistoryCompute<NEIGHFLAG,NEWTON_PAIR,EVFLAG,SHEARUPDATE>, const int &ii) const {

  // The f and torque arrays are atomic for Half/Thread neighbor style
  Kokkos::View<F_FLOAT*[3], typename DAT::t_f_array::array_layout,DeviceType,Kokkos::MemoryTraits<AtomicF<NEIGHFLAG>::value> > a_f = f;
  Kokkos::View<F_FLOAT*[3], typename DAT::t_f_array::array_layout,DeviceType,Kokkos::MemoryTraits<AtomicF<NEIGHFLAG>::value> > a_torque = torque;

  int i = d_ilist[ii];
  double xtmp = x(i,0);
  double ytmp = x(i,1);
  double ztmp = x(i,2);
  int itype = type[i];
  double imass = rmass[i];
  double irad = radius[i];
  int jnum = d_numneigh[i];
  
  double fx_i = 0.0;
  double fy_i = 0.0;
  double fz_i = 0.0;

  double torquex_i = 0.0;
  double torquey_i = 0.0;
  double torquez_i = 0.0;
  
  for (int jj = 0; jj < jnum; jj++) {
    int j = d_neighbors(i,jj);
    j &= NEIGHMASK;

    double delx = xtmp - x(j,0);
    double dely = ytmp - x(j,1);
    double delz = ztmp - x(j,2);
    double rsq = delx*delx + dely*dely + delz*delz;
    int jtype = type[j];
    double jmass = rmass[j];
    double jrad = radius[j];
    double radsum = irad + jrad;

    // check for touching neighbors

    if (rsq >= radsum * radsum) {
      d_firsttouch(i,jj) = 0;
      d_firstshear(i,3*jj) = 0;
      d_firstshear(i,3*jj+1) = 0;
      d_firstshear(i,3*jj+2) = 0;
    } else {
      double r = sqrt(rsq);
      double rinv = 1.0/r;
      double rsqinv = 1/rsq;

      // relative translational velocity

      double vr1 = v(i,0) - v(j,0);
      double vr2 = v(i,1) - v(j,1);
      double vr3 = v(i,2) - v(j,2);
      
      // normal component

      double vnnr = vr1*delx + vr2*dely + vr3*delz;
      double vn1 = delx*vnnr * rsqinv;
      double vn2 = dely*vnnr * rsqinv;
      double vn3 = delz*vnnr * rsqinv;
      
      // tangential component
      
      double vt1 = vr1 - vn1;
      double vt2 = vr2 - vn2;
      double vt3 = vr3 - vn3;

      // relative rotational velocity

      double wr1 = (irad*omega(i,0) + jrad*omega(j,0)) * rinv;
      double wr2 = (irad*omega(i,1) + jrad*omega(j,1)) * rinv;
      double wr3 = (irad*omega(i,2) + jrad*omega(j,2)) * rinv;

      double meff = imass*jmass / (imass+jmass);
      if (mask[i] & freeze_group_bit) meff = jmass;
      if (mask[j] & freeze_group_bit) meff = imass;

      double damp = meff*gamman*vnnr*rsqinv;
      double ccel = kn*(radsum-r)*rinv - damp;

      // relative velocities

      double vtr1 = vt1 - (delz*wr2-dely*wr3);
      double vtr2 = vt2 - (delx*wr3-delz*wr1);
      double vtr3 = vt3 - (dely*wr1-delx*wr2);
      double vrel = vtr1*vtr1 + vtr2*vtr2 + vtr3*vtr3;
      vrel = sqrt(vrel);

      // shear history effects

      d_firsttouch(i,jj) = 1;
      double shear1 = d_firstshear(i,3*jj);
      double shear2 = d_firstshear(i,3*jj+1);
      double shear3 = d_firstshear(i,3*jj+2);
      if (SHEARUPDATE) {
  	shear1 += vtr1*dt;
  	shear2 += vtr2*dt;
  	shear3 += vtr3*dt;
      }
      double shrmag = sqrt(shear1*shear1 + shear2*shear2 +
  			   shear3*shear3);

      // rotate shear displacements

      double rsht = shear1*delx + shear2*dely + shear3*delz;
      rsht *= rsqinv;
      if (SHEARUPDATE) {
  	shear1 -= rsht*delx;
  	shear2 -= rsht*dely;
  	shear3 -= rsht*delz;
      }

      // tangential forces = shear + tangential velocity damping

      double fs1 = - (kt*shear1 + meff*gammat*vtr1);
      double fs2 = - (kt*shear2 + meff*gammat*vtr2);
      double fs3 = - (kt*shear3 + meff*gammat*vtr3);

      // rescale frictional displacements and forces if needed

      double fs = sqrt(fs1*fs1 + fs2*fs2 + fs3*fs3);
      double fn = xmu * fabs(ccel*r);

      if (fs > fn) {
  	if (shrmag != 0.0) {
  	  shear1 = (fn/fs) * (shear1 + meff*gammat*vtr1/kt) -
  	    meff*gammat*vtr1/kt;
  	  shear2 = (fn/fs) * (shear2 + meff*gammat*vtr2/kt) -
  	    meff*gammat*vtr2/kt;
  	  shear3 = (fn/fs) * (shear3 + meff*gammat*vtr3/kt) -
  	    meff*gammat*vtr3/kt;
  	  fs1 *= fn/fs;
  	  fs2 *= fn/fs;
  	  fs3 *= fn/fs;
  	} else fs1 = fs2 = fs3 = 0.0;
      }

      if (SHEARUPDATE) {
  	d_firstshear(i,3*jj) = shear1;
  	d_firstshear(i,3*jj+1) = shear2;
  	d_firstshear(i,3*jj+2) = shear3;
      }

      // forces & torques

      double fx = delx*ccel + fs1;
      double fy = dely*ccel + fs2;
      double fz = delz*ccel + fs3;
      fx_i += fx;
      fy_i += fy;
      fz_i += fz;

      double tor1 = rinv * (dely*fs3 - delz*fs2);
      double tor2 = rinv * (delz*fs1 - delx*fs3);
      double tor3 = rinv * (delx*fs2 - dely*fs1);
      torquex_i -= irad*tor1;
      torquey_i -= irad*tor2;
      torquez_i -= irad*tor3;
      
      if (NEWTON_PAIR || j < nlocal) {
        a_f(j,0) -= fx;
        a_f(j,1) -= fy;
        a_f(j,2) -= fz;
  	a_torque(j,0) -= jrad*tor1;
  	a_torque(j,1) -= jrad*tor2;
  	a_torque(j,2) -= jrad*tor3;
      }
    }
  }

  a_f(i,0) += fx_i;
  a_f(i,1) += fy_i;
  a_f(i,2) += fz_i;
  a_torque(i,0) += torquex_i;
  a_torque(i,1) += torquey_i;
  a_torque(i,2) += torquez_i;
}

namespace LAMMPS_NS {
template class PairGranHookeHistoryKokkos<LMPDeviceType>;
#ifdef KOKKOS_HAVE_CUDA
template class PairGranHookeHistoryKokkos<LMPHostType>;
#endif
}
