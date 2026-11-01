#!/usr/bin/env python3
"""Offline geometry from VERIFIED URDF angles; no SDK/network imports."""
import argparse
import json
import math


def calculate(q, beta_degree=None, min_projection=.05):
    if len(q) != 6 or not all(math.isfinite(v) for v in q):
        raise ValueError('Need six finite URDF angles in radians')
    if not math.isfinite(min_projection) or min_projection <= 0:
        raise ValueError('Projection threshold must be positive')
    q1, q2 = q[:2]
    p2 = [0., 0., .1035]
    d = [-.35*math.cos(q2)*math.cos(q1),
         -.35*math.cos(q2)*math.sin(q1), .35*math.sin(q2)]
    length = math.hypot(*d[:2])
    alpha = math.atan2(d[1], d[0]) if length >= min_projection else None
    result = dict(input_convention='URDF angles, user must verify SDK mapping first',
                  joint2_link00_m=p2, joint3_link00_m=[a+b for a,b in zip(p2,d)],
                  direction_joint2_to_joint3_m=d, xy_projection_m=d[:2],
                  projection_length_m=length, minimum_projection_m=min_projection,
                  suitable_for_yaw=alpha is not None,
                  alpha_rad=alpha, alpha_degree=math.degrees(alpha) if alpha is not None else None,
                  beta_measured_degree=beta_degree, installation_yaw_rad=None,
                  R=None, T=None, translation_status='initial estimate, uncalibrated',
                  rotation_assumption='roll=pitch=0 must be independently verified')
    if beta_degree is not None:
        if not math.isfinite(beta_degree):
            raise ValueError('Measured beta must be finite')
        if alpha is None:
            raise ValueError('XY projection too short for yaw measurement')
        psi = (math.radians(beta_degree)-alpha+math.pi) % (2*math.pi)-math.pi
        c, s = math.cos(psi), math.sin(psi)
        result.update(installation_yaw_rad=psi, installation_yaw_degree=math.degrees(psi),
                      R=[[c,-s,0],[s,c,0],[0,0,1]],
                      T=[[c,-s,0,.078],[s,c,0,0],[0,0,1,.09602],[0,0,0,1]])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--q-urdf', nargs=6, type=float, required=True,
                   help='Verified URDF angles in radians; never target/Home values')
    p.add_argument('--beta-degree', type=float,
                   help='Measured directed joint2->joint3 azimuth in B2W XY, left positive')
    p.add_argument('--min-projection', type=float, default=.05)
    args = p.parse_args()
    print(json.dumps(calculate(args.q_urdf, args.beta_degree, args.min_projection), indent=2))


if __name__ == '__main__':
    main()
