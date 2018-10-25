import numpy as np
import sharpy.utils.algebra as algebra
import ctypes as ct


def aero2struct_force_mapping(aero_forces,
                              struct2aero_mapping,
                              zeta,
                              pos_def,
                              psi_def,
                              master,
                              master_elem,
                              cag=np.eye(3)):

    n_node, _ = pos_def.shape
    struct_forces = np.zeros((n_node, 6))

    for i_global_node in range(n_node):
        for mapping in struct2aero_mapping[i_global_node]:
            i_surf = mapping['i_surf']
            i_n = mapping['i_n']
            _, n_m, _ = aero_forces[i_surf].shape

            i_master_elem, master_elem_local_node = master[i_global_node, :]

            crv = psi_def[i_master_elem, master_elem_local_node, :]
            cab = algebra.crv2rot(crv)
            cbg = np.dot(cab.T, cag)

            for i_m in range(n_m):
                chi_g = zeta[i_surf][:, i_m, i_n] - np.dot(cag.T, pos_def[i_global_node, :])
                struct_forces[i_global_node, 0:3] += np.dot(cbg, aero_forces[i_surf][0:3, i_m, i_n])
                struct_forces[i_global_node, 3:6] += np.dot(cbg, aero_forces[i_surf][3:6, i_m, i_n])
                struct_forces[i_global_node, 3:6] += np.dot(cbg, np.cross(chi_g, aero_forces[i_surf][0:3, i_m, i_n]))

    return struct_forces


def map_forces(beam, aero, aero_step, structural_step, ts=None, unsteady=True, steady_coeff=1.0, unsteady_coeff=0.0):
    # set all forces to 0
    structural_step.steady_applied_forces.fill(0.0)

    # aero forces to structural forces
    struct_forces = steady_coeff*aero2struct_force_mapping(
        aero_step.forces,
        aero.struct2aero_mapping,
        aero_step.zeta,
        structural_step.pos,
        structural_step.psi,
        beam.node_master_elem,
        beam.master,
        structural_step.cag())

    structural_step.steady_applied_forces = (
        (struct_forces + beam.ini_info.steady_applied_forces).
            astype(dtype=ct.c_double, order='F', copy=True))

    if unsteady:
        structural_step.unsteady_applied_forces.fill(0.0)
        dynamic_struct_forces = unsteady_coeff*aero2struct_force_mapping(
            aero_step.dynamic_forces,
            aero.struct2aero_mapping,
            aero_step.zeta,
            structural_step.pos,
            structural_step.psi,
            beam.node_master_elem,
            beam.master,
            structural_step.cag())
        if ts is None:
            raise Exception('ts is needed when calling map_forces in an unsteady context!')
        structural_step.unsteady_applied_forces = (
            (dynamic_struct_forces + beam.dynamic_input[max(ts - 1, 0)]['dynamic_forces']).
                astype(dtype=ct.c_double, order='F', copy=True))

















