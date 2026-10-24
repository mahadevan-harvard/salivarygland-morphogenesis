import json

def generate_params(z0=1, k_P=0.01, k_A=0.1,tau_g=1000000, dt=0.1, kappa_n=5e-6, Nupdate=5000, motility=1e-7, r0=5.0, tau_final=1.5, growth_mode="exponential", seed=0, V=1, inputFolder="input"):
    params = {
        "scaling": {
            "Ey": 1.0,
            "mu": 1.0,
            "d": 1.0
        },

        "geometry": {
            "r0": r0,
            "h0": 1.0,
            "z0_basis": 0.001,
            "z": z0
        },

        "material": {
            "kappa_nucleus": kappa_n,
            "motility": motility,
            "k_A": k_A,
            "k_P": k_P
        },

        "homeostasis": {
            "enabled": False,
            "tau_d": 0.0,
            "tau_c": 0.0
        },

        "discretization": {
            "N": 400
        },

        "time": {
            "dt": dt,
            "tau_g": tau_g, #100000
            "Nsave": 150,
            "Nupdate": Nupdate,
            "tau_final": tau_final,
            "growth_mode": growth_mode,
            "seed": seed,
            "V": V
        }
    }

    filename = f"r_{r0:.2g}_z_{z0:.2g}_kP_{k_P:.2g}_kA_{k_A:.2g}_kn_{kappa_n:.2g}_tg_{tau_g:.2g}_M_{motility:.2g}_dt_{dt}_V{V:03d}.json" 
    path = f"{inputFolder}/{filename}"

    with open(path, "w") as f:
        json.dump(params, f, indent=2)

    print(f"Wrote parameter file to {path}")
    return filename

if __name__ == "__main__":
    generate_params()